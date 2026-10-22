"""Versioned n-step replay used only by the GPU training pipeline.

Rollouts, rather than this buffer, assemble returns within each rally. This
module never concatenates neighboring rallies or imports legacy one-step data.
Reference transitions are ordinary TD examples, not behavior-cloning labels.
"""

import numpy as np

from dqn import to_host, xp


class NStepReplayBuffer:
    SCHEMA_VERSION = 1
    FIELDS = ('state', 'action', 'reward', 'next_state', 'done', 'discount', 'outcome')

    def __init__(self, capacity=1_000_000, state_dim=11,
                 reference_capacity=50_000, n_step=5, gamma=0.99, seed=None):
        if min(capacity, state_dim, n_step) < 1 or reference_capacity < 0:
            raise ValueError('Replay dimensions and n_step must be positive.')
        if not 0.0 < gamma <= 1.0:
            raise ValueError('Replay gamma must be in (0, 1].')
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.reference_capacity = int(reference_capacity)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.counter = 0
        self.pointer = 0
        self.reference_counter = 0
        self.rng = xp.random.default_rng(seed)
        self.last_load_error = None
        self._reference_keys = set()
        self._signal_indices = xp.empty(0, dtype=xp.int64)
        for prefix, count in (('', self.capacity), ('reference_', self.reference_capacity)):
            for field in self.FIELDS:
                shape = ((count, self.state_dim)
                         if field in ('state', 'next_state') else (count,))
                dtype = xp.int32 if field == 'action' else xp.float32
                setattr(self, prefix + field + '_buffer', xp.zeros(shape, dtype=dtype))

    @property
    def recent_size(self):
        return min(self.counter, self.capacity)

    @property
    def reference_size(self):
        return self.reference_counter

    @property
    def size(self):
        return self.recent_size + self.reference_size

    def _arrays(self, batch):
        missing = set(self.FIELDS) - set(batch)
        if missing:
            raise ValueError('Missing n-step replay fields: ' + ', '.join(sorted(missing)))
        count = int(batch['state'].shape[0])
        result = {}
        for field in self.FIELDS:
            shape = ((count, self.state_dim)
                     if field in ('state', 'next_state') else (count,))
            if batch[field].shape != shape:
                raise ValueError(f'Invalid {field} shape: expected {shape}.')
            dtype = xp.int32 if field == 'action' else xp.float32
            result[field] = xp.asarray(batch[field], dtype=dtype)
        return result, count

    def _refresh_signal_indices(self):
        # Rebuilt once per rollout/load, rather than scanning a million slots
        # for each optimizer minibatch. A nonzero n-step reward marks the
        # frames close enough to the terminal result to carry direct signal.
        self._signal_indices = xp.flatnonzero(
            self.reward_buffer[:self.recent_size] != 0,
        )

    def push_arrays(self, batch):
        """Append already-computed, episode-safe n-step transitions."""
        arrays, count = self._arrays(batch)
        if not count:
            return 0
        written = min(count, self.capacity)
        first = (self.pointer + count - written) % self.capacity
        indices = (xp.arange(written, dtype=xp.int64) + first) % self.capacity
        for field, values in arrays.items():
            getattr(self, field + '_buffer')[indices] = values[-written:]
        self.counter += count
        self.pointer = self.counter % self.capacity
        self._refresh_signal_indices()
        return count

    @classmethod
    def _reference_key_rows(cls, host_arrays):
        """Encode every TD field, retaining distinct stochastic continuations."""
        count = len(host_arrays['action'])
        columns = [np.asarray(host_arrays[field], dtype=np.float32).reshape(count, -1)
                   for field in cls.FIELDS]
        packed = np.ascontiguousarray(np.concatenate(columns, axis=1))
        row_dtype = np.dtype((np.void, packed.dtype.itemsize * packed.shape[1]))
        return packed.view(row_dtype).reshape(-1)

    def push_reference_arrays(self, batch):
        """Add explicit reference-policy data, deduplicated by full transition.

        The fixed initial states can yield thousands of identical rallies.
        Keeping their unique TD transitions avoids presenting repetition as
        broad reference coverage. Identical observations/actions can still
        have different future returns, and those examples must remain distinct.
        Training rollouts never call this implicitly.
        """
        arrays, count = self._arrays(batch)
        room = self.reference_capacity - self.reference_counter
        if not count or room <= 0:
            return 0
        host_arrays = {field: np.asarray(to_host(array))
                       for field, array in arrays.items()}
        key_rows = self._reference_key_rows(host_arrays)
        # Eliminate identical rows in NumPy before the cross-batch set lookup,
        # preserving first-occurrence order and avoiding a Python loop over
        # many thousands of duplicate deterministic trajectories.
        _, first_indices = np.unique(key_rows, return_index=True)
        selected = []
        for index in np.sort(first_indices):
            key = key_rows[index].tobytes()
            if key in self._reference_keys:
                continue
            self._reference_keys.add(key)
            selected.append(index)
            if len(selected) >= room:
                break
        if not selected:
            return 0
        first = self.reference_counter
        last = first + len(selected)
        indices = xp.asarray(selected, dtype=xp.int64)
        for field, values in arrays.items():
            getattr(self, 'reference_' + field + '_buffer')[first:last] = values[indices]
        self.reference_counter = last
        return len(selected)

    def clear_recent(self):
        discarded = self.recent_size
        self.counter = self.pointer = 0
        self._signal_indices = xp.empty(0, dtype=xp.int64)
        return discarded

    def clear_reference(self):
        discarded = self.reference_counter
        self.reference_counter = 0
        self._reference_keys.clear()
        return discarded

    def sample(self, batch_size):
        """Sample reference, direct-signal, and uniform recent examples.

        Direct-signal oversampling is corrected to the same objective as a
        25% uniform-reference / 75% uniform-recent mixture. The returned weight
        is the desired probability divided by the actual mixture probability.
        DQN uses self-normalized importance sampling, dividing by the sampled
        minibatch's mean weight. This preserves scale but introduces finite-
        sample bias; it is not an exactly unbiased gradient estimator.
        Sampling uses replacement, including when a reference set is small.
        """
        batch_size = int(batch_size)
        if batch_size < 1 or self.size == 0:
            raise ValueError('Sampling requires positive batch_size and nonempty replay.')
        reference_count = batch_size // 4 if self.reference_size else 0
        if not self.recent_size:
            reference_count = batch_size
        recent_count = batch_size - reference_count
        signal_count = min(batch_size // 4, recent_count)
        if not len(self._signal_indices):
            signal_count = 0
        uniform_count = recent_count - signal_count
        # Keep support for every recent example, even with a tiny batch.
        if recent_count and not uniform_count:
            signal_count -= 1
            uniform_count = 1

        reference_indices = (self.rng.integers(0, self.reference_size, size=reference_count)
                             if reference_count else None)
        recent_parts = []
        if signal_count:
            positions = self.rng.integers(0, len(self._signal_indices), size=signal_count)
            recent_parts.append(self._signal_indices[positions])
        if uniform_count:
            recent_parts.append(self.rng.integers(0, self.recent_size, size=uniform_count))
        recent_indices = (xp.concatenate(recent_parts) if len(recent_parts) > 1
                          else recent_parts[0] if recent_parts else None)

        result = {}
        for field in self.FIELDS:
            pieces = []
            if reference_indices is not None:
                pieces.append(getattr(self, 'reference_' + field + '_buffer')[reference_indices])
            if recent_indices is not None:
                pieces.append(getattr(self, field + '_buffer')[recent_indices])
            result[field] = xp.concatenate(pieces) if len(pieces) > 1 else pieces[0]

        weights = []
        if reference_count:
            weights.append(xp.ones(reference_count, dtype=xp.float32))
        if recent_count:
            # Conditional on drawing recent data: q(i)=u/R+s*I(signal)/S.
            uniform_fraction = uniform_count / recent_count
            signal_fraction = signal_count / recent_count
            is_signal = (self.reward_buffer[recent_indices] != 0).astype(xp.float32)
            density_ratio = (uniform_fraction + signal_fraction * is_signal
                             * self.recent_size / max(len(self._signal_indices), 1))
            weights.append(1.0 / density_ratio)
        result['weight'] = (xp.concatenate(weights) if len(weights) > 1 else weights[0])
        return result

    def export_state(self):
        """Return host arrays and strict schema metadata for NPZ serialization."""
        values = {
            'nstep_schema_version': self.SCHEMA_VERSION,
            'nstep_capacity': self.capacity,
            'nstep_state_dim': self.state_dim,
            'nstep_n_step': self.n_step,
            'nstep_gamma': self.gamma,
            'nstep_counter': self.counter,
            'nstep_pointer': self.pointer,
            'reference_capacity': self.reference_capacity,
            'reference_counter': self.reference_counter,
        }
        for field in self.FIELDS:
            values['nstep_' + field + '_buffer'] = to_host(getattr(self, field + '_buffer'))
            values['reference_' + field + '_buffer'] = to_host(
                getattr(self, 'reference_' + field + '_buffer'),
            )
        return values

    def load_state(self, values):
        """Restore only an exactly compatible schema; reject legacy replay.

        Incompatible metadata/data clears both partitions and returns False.
        ``last_load_error`` explains the rejection for the training log.
        """
        self.clear_recent()
        self.clear_reference()
        self.last_load_error = None
        try:
            expected = {
                'nstep_schema_version': self.SCHEMA_VERSION,
                'nstep_capacity': self.capacity,
                'nstep_state_dim': self.state_dim,
                'nstep_n_step': self.n_step,
                'reference_capacity': self.reference_capacity,
            }
            for key, expected_value in expected.items():
                if key not in values or int(values[key]) != expected_value:
                    raise ValueError(f'Incompatible replay metadata: {key}.')
            if 'nstep_gamma' not in values or not np.isclose(
                    float(values['nstep_gamma']), self.gamma, rtol=0, atol=1e-8):
                raise ValueError('Incompatible replay gamma.')
            counter = int(values['nstep_counter'])
            pointer = int(values['nstep_pointer'])
            reference_count = int(values['reference_counter'])
            if counter < 0 or pointer != counter % self.capacity:
                raise ValueError('Invalid replay ring position.')
            if not 0 <= reference_count <= self.reference_capacity:
                raise ValueError('Invalid reference replay count.')
            validated = {}
            for prefix, count in (('nstep_', self.capacity),
                                  ('reference_', self.reference_capacity)):
                for field in self.FIELDS:
                    key = prefix + field + '_buffer'
                    array = np.asarray(values[key])
                    shape = ((count, self.state_dim)
                             if field in ('state', 'next_state') else (count,))
                    expected_dtype = np.dtype(np.int32 if field == 'action' else np.float32)
                    if array.shape != shape or array.dtype != expected_dtype:
                        raise ValueError(f'Invalid replay array schema: {key}.')
                    if not np.all(np.isfinite(array)):
                        raise ValueError(f'Non-finite replay array: {key}.')
                    if field == 'discount' and np.any((array < 0) | (array > 1)):
                        raise ValueError(f'Invalid bootstrap discount: {key}.')
                    if field == 'done' and np.any((array != 0) & (array != 1)):
                        raise ValueError(f'Invalid terminal flags: {key}.')
                    if field == 'action' and np.any((array < 0) | (array >= 18)):
                        raise ValueError(f'Invalid action IDs: {key}.')
                    validated[key] = array
            # No payload is installed until every field has passed validation.
            for field in self.FIELDS:
                getattr(self, field + '_buffer')[...] = xp.asarray(
                    validated['nstep_' + field + '_buffer'],
                )
                getattr(self, 'reference_' + field + '_buffer')[...] = xp.asarray(
                    validated['reference_' + field + '_buffer'],
                )
            self.counter = counter
            self.pointer = pointer
            self.reference_counter = reference_count
            if reference_count:
                reference_arrays = {
                    field: validated['reference_' + field + '_buffer'][:reference_count]
                    for field in self.FIELDS
                }
                for row in self._reference_key_rows(reference_arrays):
                    self._reference_keys.add(row.tobytes())
            self._refresh_signal_indices()
            return True
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            self.clear_recent()
            self.clear_reference()
            self.last_load_error = str(error)
            return False

    def diagnostics(self):
        """Small counters for reporting; no per-frame arrays leave the GPU."""
        count = self.recent_size
        used = self.reward_buffer[:count]
        stats = xp.stack((xp.count_nonzero(used > 0), xp.count_nonzero(used < 0)))
        positive, negative = map(int, to_host(stats))
        allocated = sum(getattr(self, prefix + field + '_buffer').nbytes
                        for prefix in ('', 'reference_') for field in self.FIELDS)
        return {
            'schema_version': self.SCHEMA_VERSION,
            'recent': count,
            'reference': self.reference_size,
            'positive_signal': positive,
            'negative_signal': negative,
            'signal_fraction': (positive + negative) / max(count, 1),
            'allocated_bytes': int(allocated),
            'n_step': self.n_step,
            'gamma': self.gamma,
        }
