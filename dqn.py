"""DQN implementation with an optional CuPy backend for CUDA GPUs."""

import os

import numpy as np


DEVICE_REQUEST = os.environ.get('PIKA_DEVICE', 'auto').strip().lower()
if DEVICE_REQUEST not in ('auto', 'cpu', 'cuda'):
    raise ValueError('PIKA_DEVICE must be one of: auto, cpu, cuda')

cp = None
if DEVICE_REQUEST != 'cpu':
    try:
        import cupy as cp
        device_index = int(os.environ.get('PIKA_CUDA_DEVICE', '0'))
        cp.cuda.Device(device_index).use()
        cp.cuda.runtime.getDeviceCount()
        xp = cp
        BACKEND = 'cupy'
    except Exception as error:
        cp = None
        if DEVICE_REQUEST == 'cuda':
            raise RuntimeError(
                'PIKA_DEVICE=cuda was requested, but CuPy could not initialize a CUDA device: '
                + str(error)
            ) from error
        xp = np
        BACKEND = 'numpy'
else:
    xp = np
    BACKEND = 'numpy'


def to_host(value):
    """Convert a CPU or GPU array to a NumPy array for JSON/NPZ serialization."""
    if BACKEND == 'cupy':
        return cp.asnumpy(value)
    return np.asarray(value)


class DQN:
    def __init__(self):
        # Fine-tuning a previously competent policy needs smaller, bounded
        # steps; the old 1e-4 MSE updates repeatedly erased its behavior.
        self.lr = 0.00002
        # Rewards arrive when a rally ends, often dozens of frames after an action.
        self.gamma = 0.99
        self.epsilon = 1.0
        self.training_steps = 0
        self.episodes = 0
        # Keep simulator seeds independent from the champion's rally count so
        # rollback does not replay the same training rallies on every restart.
        self.rollout_seed = 0
        # Exploration age is independent of inherited model rallies and RNG seeds.
        self.exploration_rallies = 0

        # Add the physics engine's predicted ball landing point. This gives the
        # policy an explicit trajectory feature instead of making it infer the
        # landing from position and velocity alone.
        self.input_dim = 11
        self.hidden_dim = 64
        # 3 horizontal directions x 3 vertical directions x power-hit on/off.
        self.output_dim = 18

        self.rng = xp.random.default_rng()
        self.W1 = (self.rng.standard_normal(size=(self.input_dim, self.hidden_dim))
                   * np.sqrt(2 / self.input_dim)).astype(xp.float32)
        self.W2 = (self.rng.standard_normal(size=(self.hidden_dim, self.hidden_dim))
                   * np.sqrt(2 / self.hidden_dim)).astype(xp.float32)
        self.W3 = (self.rng.standard_normal(size=(self.hidden_dim, self.output_dim))
                   * np.sqrt(2 / self.hidden_dim)).astype(xp.float32)

        self.b1 = xp.zeros(self.hidden_dim, dtype=xp.float32)
        self.b2 = xp.zeros(self.hidden_dim, dtype=xp.float32)
        self.b3 = xp.zeros(self.output_dim, dtype=xp.float32)

        self.target_W1 = self.W1.copy()
        self.target_W2 = self.W2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b1 = self.b1.copy()
        self.target_b2 = self.b2.copy()
        self.target_b3 = self.b3.copy()
        self.target_update_interval = 2000
        self.max_grad_norm = 5.0
        # GPU training can bound bootstrap estimates for its terminal +/-1
        # objective. Leave the legacy trainer's behavior unchanged by default.
        self.bootstrap_clip = None
        # Explicitly enabled only by the GPU trainer. These constraints preserve
        # demonstrated winning decisions while their value estimates are fitted.
        self.winner_margin = 0.0
        self.winner_margin_weight = 0.0

    def _forward(self, X):
        X = xp.asarray(X, dtype=xp.float32)

        Z1 = xp.dot(X, self.W1) + self.b1
        A1 = xp.maximum(Z1, 0)
        Z2 = xp.dot(A1, self.W2) + self.b2
        A2 = xp.maximum(Z2, 0)
        Z3 = xp.dot(A2, self.W3) + self.b3
        return {'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'A1': A1, 'A2': A2}

    def predict(self, X):
        return self._forward(X)['Z3']

    def predict_target(self, X):
        X = xp.asarray(X, dtype=xp.float32)
        Z1 = xp.maximum(xp.dot(X, self.target_W1) + self.target_b1, 0)
        Z2 = xp.maximum(xp.dot(Z1, self.target_W2) + self.target_b2, 0)
        return xp.dot(Z2, self.target_W3) + self.target_b3

    def sync_target_network(self):
        for name in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
            getattr(self, 'target_' + name)[...] = getattr(self, name)

    def valid_action_mask(self, states):
        """Mask only horizontal movement that would push the player out of court."""
        states = xp.asarray(states, dtype=xp.float32)
        if states.ndim == 1:
            states = states[None, :]

        player_x = (1.0 - states[:, 0]) * 432.0
        direction_slots = xp.arange(9, dtype=xp.int32)[None, :]
        x_direction = -((direction_slots // 3) - 1)
        valid = ~(
            ((player_x[:, None] >= 400.0) & (x_direction > 0)) |
            ((player_x[:, None] <= 248.0) & (x_direction < 0))
        )
        # Power-hit actions use the same directional slots. Keep every dive,
        # including ones made before the ball is close enough to connect.
        return xp.concatenate((valid, valid), axis=1)

    def td_targets(self, batch):
        """Compute detached Double DQN targets for one-step or n-step replay.

        ``discount`` already includes gamma**horizon and the terminal mask.
        Older callers without it keep the original one-step target.
        """
        # Double DQN: choose the next action with the online network, but
        # evaluate it with the target network to limit optimistic Q drift.
        online_next = self.predict(batch['next_state'])
        online_next = xp.where(
            self.valid_action_mask(batch['next_state']), online_next, -xp.inf,
        )
        next_actions = xp.argmax(online_next, axis=1)
        target_next = self.predict_target(batch['next_state'])
        next_values = target_next[xp.arange(len(next_actions)), next_actions]
        if self.bootstrap_clip is not None:
            bounds = ((-float(self.bootstrap_clip), float(self.bootstrap_clip))
                      if np.isscalar(self.bootstrap_clip) else self.bootstrap_clip)
            next_values = xp.clip(next_values, *bounds)
        if 'discount' in batch:
            discounts = xp.asarray(batch['discount'], dtype=xp.float32)
        else:
            discounts = self.gamma * (1 - xp.asarray(batch['done'], dtype=xp.float32))
        # where also prevents a terminal zero discount from multiplying a
        # non-finite estimate and turning an otherwise valid reward into NaN.
        bootstrap = xp.where(discounts != 0, next_values, 0.0)
        targets = xp.asarray(batch['reward'], dtype=xp.float32) + discounts * bootstrap
        if 'return_mask' in batch:
            targets = xp.where(
                xp.asarray(batch['return_mask']) > 0,
                xp.asarray(batch['return_target'], dtype=xp.float32), targets,
            )
        if self.bootstrap_clip is not None:
            targets = xp.clip(targets, *bounds)
        return targets

    def _get_Y(self, batch):
        Y = self.predict(batch['state'])
        targets = self.td_targets(batch)
        Y[xp.arange(len(batch['action'])), batch['action'].astype(xp.int32)] = targets
        return Y

    def train_step(self, batch, return_loss=True, return_metrics=False):
        """Run weighted Huber SGD; diagnostics are opt-in to avoid GPU stalls."""
        X = xp.asarray(batch['state'], dtype=xp.float32)
        targets = self.td_targets(batch)

        forwarded = self._forward(X)
        A1 = forwarded['A1']
        A2 = forwarded['A2']
        Z1 = forwarded['Z1']
        Z2 = forwarded['Z2']
        Z3 = forwarded['Z3']

        # Huber loss on the selected action avoids large TD errors producing
        # unstable updates. Unselected actions have zero loss/gradient.
        rows = xp.arange(X.shape[0])
        actions = xp.asarray(batch['action'], dtype=xp.int32)
        selected_values = Z3[rows, actions]
        td_error = selected_values - targets
        weights = (xp.asarray(batch['weight'], dtype=xp.float32) if 'weight' in batch
                   else xp.ones(X.shape[0], dtype=xp.float32))
        # The sampler supplies exact mixture corrections. Normalizing their
        # minibatch mean keeps the optimizer's effective step size stable.
        weights = weights / xp.maximum(xp.mean(weights), 1e-8)
        dZ3 = xp.zeros_like(Z3)
        dZ3[rows, actions] = xp.clip(td_error, -1.0, 1.0) * weights / X.shape[0]
        margin_loss = xp.asarray(0.0, dtype=xp.float32)
        margin_violation_fraction = xp.asarray(0.0, dtype=xp.float32)
        if (self.winner_margin > 0 and self.winner_margin_weight > 0
                and 'margin_mask' in batch):
            eligible = (xp.asarray(batch['margin_mask']) > 0).astype(xp.float32)
            eligible_count = xp.maximum(xp.sum(eligible), 1.0)
            other_valid = (self.valid_action_mask(X)
                           & (xp.arange(self.output_dim)[None, :] != actions[:, None]))
            other_values = xp.where(other_valid, Z3, -xp.inf)
            competitors = xp.argmax(other_values, axis=1)
            violation = xp.maximum(
                0.0, self.winner_margin + Z3[rows, competitors] - selected_values,
            )
            active = eligible * (violation > 0)
            # This is a separate action-ranking objective, deliberately not
            # reweighted by the replay TD importance corrections.
            margin_gradient = self.winner_margin_weight * active / eligible_count
            dZ3[rows, actions] -= margin_gradient
            dZ3[rows, competitors] += margin_gradient
            margin_loss = (self.winner_margin_weight
                           * xp.sum(violation * eligible) / eligible_count)
            margin_violation_fraction = xp.sum(active) / eligible_count
        dW3 = xp.dot(A2.T, dZ3)
        dZ2 = xp.dot(dZ3, self.W3.T) * (Z2 > 0)
        dW2 = xp.dot(A1.T, dZ2)
        dZ1 = xp.dot(dZ2, self.W2.T) * (Z1 > 0)
        dW1 = xp.dot(X.T, dZ1)
        db1 = xp.sum(dZ1, axis=0)
        db2 = xp.sum(dZ2, axis=0)
        db3 = xp.sum(dZ3, axis=0)

        gradients = (dW1, dW2, dW3, db1, db2, db3)
        squared_norm = sum(xp.sum(gradient * gradient) for gradient in gradients)
        grad_norm = xp.sqrt(squared_norm)
        scale = xp.minimum(1.0, self.max_grad_norm / (grad_norm + 1e-8))
        for gradient in gradients:
            gradient *= scale

        self.W1 -= self.lr * dW1
        self.W2 -= self.lr * dW2
        self.W3 -= self.lr * dW3
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2
        self.b3 -= self.lr * db3
        self.training_steps += 1
        if self.training_steps % self.target_update_interval == 0:
            self.sync_target_network()

        if return_loss or return_metrics:
            absolute_error = xp.abs(td_error)
            loss = xp.mean(weights * xp.where(
                absolute_error <= 1.0,
                0.5 * td_error * td_error,
                absolute_error - 0.5,
            ))
            if return_metrics:
                return_count = (xp.count_nonzero(xp.asarray(batch['return_mask']) > 0)
                                if 'return_mask' in batch else xp.asarray(0))
                values = xp.stack((
                    loss, xp.mean(absolute_error), xp.mean(selected_values),
                    xp.mean(targets), xp.min(selected_values), xp.max(selected_values),
                    grad_norm, xp.mean((absolute_error > 1.0).astype(xp.float32)),
                    xp.mean(weights * absolute_error), margin_loss,
                    margin_violation_fraction, return_count,
                ))
                names = ('loss', 'mean_abs_td', 'mean_q', 'mean_target',
                         'min_q', 'max_q', 'grad_norm', 'huber_clipped_fraction',
                         'weighted_mean_abs_td', 'margin_loss',
                         'margin_violation_fraction', 'return_label_count')
                return dict(zip(names, map(float, to_host(values))))
            return float(to_host(loss))
        return None

    def get_action(self, X):
        state = xp.asarray(X, dtype=xp.float32)
        if state.ndim == 1:
            state = state[None, :]
        valid = self.valid_action_mask(state)[0]
        if float(to_host(self.rng.random())) <= self.epsilon:
            valid_actions = xp.nonzero(valid)[0]
            index = self.rng.integers(0, len(valid_actions))
            return int(to_host(valid_actions[index]))
        values = xp.where(valid, self.predict(state)[0], -xp.inf)
        return int(to_host(xp.argmax(values)))


class ReplayBuffer:
    def __init__(self):
        # A single GPU rollout can produce more than 100k frame transitions.
        # Keep multiple rollout batches so each new batch does not erase the
        # previous policy's experience and trigger rapid forgetting.
        self.capacity = 1_000_000
        self.state_dim = 11
        self.counter = 0
        self.pointer = 0
        # Preserve examples collected while the current champion is acting.
        # The recent ring below is allowed to roll over; this anchor partition
        # is append-only so later policies cannot erase the champion's behavior.
        self.champion_capacity = 200_000
        self.champion_counter = 0
        self.rng = xp.random.default_rng()

        self.state_buffer = xp.zeros((self.capacity, self.state_dim), dtype=xp.float32)
        self.next_state_buffer = xp.zeros((self.capacity, self.state_dim), dtype=xp.float32)
        self.action_buffer = xp.zeros(self.capacity, dtype=xp.int32)
        self.reward_buffer = xp.zeros(self.capacity, dtype=xp.float32)
        self.done_buffer = xp.zeros(self.capacity, dtype=xp.float32)
        self.champion_state_buffer = xp.zeros(
            (self.champion_capacity, self.state_dim), dtype=xp.float32,
        )
        self.champion_next_state_buffer = xp.zeros_like(self.champion_state_buffer)
        self.champion_action_buffer = xp.zeros(self.champion_capacity, dtype=xp.int32)
        self.champion_reward_buffer = xp.zeros(self.champion_capacity, dtype=xp.float32)
        self.champion_done_buffer = xp.zeros(self.champion_capacity, dtype=xp.float32)

    @property
    def size(self):
        return min(self.counter, self.capacity) + min(
            self.champion_counter, self.champion_capacity,
        )

    def push_batch(self, transitions):
        """Transfer one JSON batch to the selected backend in a few array copies."""
        count = len(transitions)
        if not count:
            return

        host_indices = (np.arange(count, dtype=np.int64) + self.pointer) % self.capacity
        indices = xp.asarray(host_indices)
        states = np.asarray([item['state'] for item in transitions], dtype=np.float32)
        next_states = np.asarray([item['next_state'] for item in transitions], dtype=np.float32)
        actions = np.fromiter((int(item['action']) for item in transitions), dtype=np.int32, count=count)
        rewards = np.fromiter((float(item['reward']) for item in transitions), dtype=np.float32, count=count)
        dones = np.fromiter((bool(item['done']) for item in transitions), dtype=np.float32, count=count)

        self._push_champion(states, actions, rewards, next_states, dones)

        self.state_buffer[indices] = xp.asarray(states)
        self.next_state_buffer[indices] = xp.asarray(next_states)
        self.action_buffer[indices] = xp.asarray(actions)
        self.reward_buffer[indices] = xp.asarray(rewards)
        self.done_buffer[indices] = xp.asarray(dones)

        self.counter += count
        self.pointer = self.counter % self.capacity

    def push_arrays(self, states, actions, rewards, next_states, dones):
        """Append already-device-resident rollout arrays without a host round trip."""
        count = int(states.shape[0])
        if count == 0:
            return

        self._push_champion(states, actions, rewards, next_states, dones)

        if count > self.capacity:
            states = states[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            next_states = next_states[-self.capacity:]
            dones = dones[-self.capacity:]
            first_index = (self.pointer + count - self.capacity) % self.capacity
            write_count = self.capacity
        else:
            first_index = self.pointer
            write_count = count
        indices = (xp.arange(write_count, dtype=xp.int64) + first_index) % self.capacity
        self.state_buffer[indices] = xp.asarray(states, dtype=xp.float32)
        self.next_state_buffer[indices] = xp.asarray(next_states, dtype=xp.float32)
        self.action_buffer[indices] = xp.asarray(actions, dtype=xp.int32)
        self.reward_buffer[indices] = xp.asarray(rewards, dtype=xp.float32)
        self.done_buffer[indices] = xp.asarray(dones, dtype=xp.float32)
        self.counter += count
        self.pointer = self.counter % self.capacity

    def _push_champion(self, states, actions, rewards, next_states, dones):
        """Keep an immutable sample of the policy active at training start."""
        count = int(states.shape[0])
        champion_count = min(count, self.champion_capacity - self.champion_counter)
        if champion_count <= 0:
            return
        first = self.champion_counter
        last = first + champion_count
        self.champion_state_buffer[first:last] = xp.asarray(
            states[:champion_count], dtype=xp.float32,
        )
        self.champion_next_state_buffer[first:last] = xp.asarray(
            next_states[:champion_count], dtype=xp.float32,
        )
        self.champion_action_buffer[first:last] = xp.asarray(
            actions[:champion_count], dtype=xp.int32,
        )
        self.champion_reward_buffer[first:last] = xp.asarray(
            rewards[:champion_count], dtype=xp.float32,
        )
        self.champion_done_buffer[first:last] = xp.asarray(
            dones[:champion_count], dtype=xp.float32,
        )
        self.champion_counter = last

    def push(self, state, action, reward, next_state, done):
        self.push_batch([{
            'state': state, 'action': action, 'reward': reward,
            'next_state': next_state, 'done': done,
        }])

    def clear_recent(self):
        """Discard recent data from a regressed policy; keep champion anchors."""
        discarded = min(self.counter, self.capacity)
        self.counter = 0
        self.pointer = 0
        return discarded

    def clear_champion(self):
        """Discard anchors whose behavior policy is unknown or outdated."""
        discarded = min(self.champion_counter, self.champion_capacity)
        self.champion_counter = 0
        return discarded

    def push_champion_arrays(self, states, actions, rewards, next_states, dones):
        """Add transitions from a deterministic champion-policy rollout only."""
        self._push_champion(states, actions, rewards, next_states, dones)

    def sample(self, batch_size):
        recent_available = min(self.counter, self.capacity)
        champion_available = min(self.champion_counter, self.champion_capacity)
        available = recent_available + champion_available
        if batch_size > available:
            raise ValueError(f'Need {batch_size} replay items, only {available} available')

        # Keep one quarter of each batch anchored to champion play. The remaining
        # samples track recent behavior, without relying on CuPy choice().
        champion_batch = min(batch_size // 4, champion_available)
        recent_batch = batch_size - champion_batch
        if recent_batch > recent_available:
            champion_batch = min(batch_size, champion_available)
            recent_batch = batch_size - champion_batch
        champion_indices = self.rng.integers(
            0, champion_available, size=champion_batch,
        ) if champion_batch else None
        recent_indices = self.rng.integers(
            0, recent_available, size=recent_batch,
        ) if recent_batch else None

        def combine(name, champion_name):
            parts = []
            if champion_indices is not None:
                parts.append(getattr(self, champion_name)[champion_indices])
            if recent_indices is not None:
                parts.append(getattr(self, name)[recent_indices])
            return xp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

        return {
            'state': combine('state_buffer', 'champion_state_buffer'),
            'action': combine('action_buffer', 'champion_action_buffer'),
            'reward': combine('reward_buffer', 'champion_reward_buffer'),
            'next_state': combine('next_state_buffer', 'champion_next_state_buffer'),
            'done': combine('done_buffer', 'champion_done_buffer'),
        }


def restore_replay_buffer(checkpoint, buffer):
    """Restore a replay ring across capacity changes and a 10→11 state upgrade."""
    required = ('state_buffer', 'next_state_buffer', 'action_buffer',
                'reward_buffer', 'done_buffer', 'counter', 'pointer')
    if not all(name in checkpoint.files for name in required):
        return 0

    states = checkpoint['state_buffer']
    next_states = checkpoint['next_state_buffer']
    actions = checkpoint['action_buffer']
    rewards = checkpoint['reward_buffer']
    dones = checkpoint['done_buffer']
    if (states.ndim != 2 or states.shape != next_states.shape or
            states.shape[1] not in (buffer.state_dim - 1, buffer.state_dim)):
        return 0
    old_capacity = states.shape[0]
    if (old_capacity < 1 or actions.shape != (old_capacity,) or
            rewards.shape != (old_capacity,) or dones.shape != (old_capacity,)):
        return 0

    total_written = max(0, int(checkpoint['counter']))
    available = min(total_written, old_capacity)
    copy_count = min(available, buffer.capacity)
    if copy_count == 0:
        return 0

    old_pointer = int(checkpoint['pointer']) % old_capacity
    start = (old_pointer - copy_count) % old_capacity if total_written >= old_capacity else 0
    indices = (start + np.arange(copy_count, dtype=np.int64)) % old_capacity
    old_state_dim = states.shape[1]
    buffer.state_buffer[:copy_count, :old_state_dim] = xp.asarray(
        states[indices], dtype=buffer.state_buffer.dtype,
    )
    buffer.next_state_buffer[:copy_count, :old_state_dim] = xp.asarray(
        next_states[indices], dtype=buffer.next_state_buffer.dtype,
    )
    if old_state_dim < buffer.state_dim:
        buffer.state_buffer[:copy_count, old_state_dim:] = 0
        buffer.next_state_buffer[:copy_count, old_state_dim:] = 0
    buffer.action_buffer[:copy_count] = xp.asarray(actions[indices], dtype=xp.int32)
    buffer.reward_buffer[:copy_count] = xp.asarray(rewards[indices], dtype=xp.float32)
    buffer.done_buffer[:copy_count] = xp.asarray(dones[indices], dtype=xp.float32)
    buffer.counter = copy_count
    buffer.pointer = copy_count % buffer.capacity

    # Newer checkpoints persist a separate, immutable champion partition.
    # Old checkpoints simply start filling it from the restored champion.
    anchor_required = (
        'champion_state_buffer', 'champion_next_state_buffer',
        'champion_action_buffer', 'champion_reward_buffer',
        'champion_done_buffer', 'champion_counter',
    )
    if not all(name in checkpoint.files for name in anchor_required):
        buffer.champion_counter = 0
    if all(name in checkpoint.files for name in anchor_required):
        anchor_states = checkpoint['champion_state_buffer']
        anchor_next = checkpoint['champion_next_state_buffer']
        anchor_actions = checkpoint['champion_action_buffer']
        anchor_rewards = checkpoint['champion_reward_buffer']
        anchor_dones = checkpoint['champion_done_buffer']
        anchor_capacity = anchor_states.shape[0] if anchor_states.ndim == 2 else 0
        if (anchor_capacity > 0 and anchor_states.shape == anchor_next.shape and
                anchor_states.shape[1] in (buffer.state_dim - 1, buffer.state_dim) and
                anchor_actions.shape == (anchor_capacity,) and
                anchor_rewards.shape == (anchor_capacity,) and
                anchor_dones.shape == (anchor_capacity,)):
            anchor_written = max(0, min(int(checkpoint['champion_counter']), anchor_capacity))
            anchor_copy = min(anchor_written, buffer.champion_capacity)
            old_anchor_dim = anchor_states.shape[1]
            if anchor_copy:
                buffer.champion_state_buffer[:anchor_copy, :old_anchor_dim] = xp.asarray(
                    anchor_states[:anchor_copy], dtype=xp.float32,
                )
                buffer.champion_next_state_buffer[:anchor_copy, :old_anchor_dim] = xp.asarray(
                    anchor_next[:anchor_copy], dtype=xp.float32,
                )
                if old_anchor_dim < buffer.state_dim:
                    buffer.champion_state_buffer[:anchor_copy, old_anchor_dim:] = 0
                    buffer.champion_next_state_buffer[:anchor_copy, old_anchor_dim:] = 0
                buffer.champion_action_buffer[:anchor_copy] = xp.asarray(
                    anchor_actions[:anchor_copy], dtype=xp.int32,
                )
                buffer.champion_reward_buffer[:anchor_copy] = xp.asarray(
                    anchor_rewards[:anchor_copy], dtype=xp.float32,
                )
                buffer.champion_done_buffer[:anchor_copy] = xp.asarray(
                    anchor_dones[:anchor_copy], dtype=xp.float32,
                )
                buffer.champion_counter = anchor_copy
    return copy_count
