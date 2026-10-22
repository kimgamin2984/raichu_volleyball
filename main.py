"""Offline DQN worker. Gameplay never connects to this process."""
import json
import os
import sys
import tempfile

import numpy as np

from dqn import BACKEND, DQN, ReplayBuffer, restore_replay_buffer, to_host, xp

BATCH_SIZE = 64
EPSILON_MIN = 0.05
EPSILON_DECAY_RALLIES = 300000
TRAINING_VERSION = 9
dqn = DQN()
buffer = ReplayBuffer()


def save_checkpoint(path, include_replay=True):
    """Write a resumable checkpoint atomically."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.checkpoint-', suffix='.npz', dir=parent)
    os.close(fd)
    values = {
        'W1': to_host(dqn.W1), 'W2': to_host(dqn.W2), 'W3': to_host(dqn.W3),
        'b1': to_host(dqn.b1), 'b2': to_host(dqn.b2), 'b3': to_host(dqn.b3),
        'target_W1': to_host(dqn.target_W1), 'target_W2': to_host(dqn.target_W2),
        'target_W3': to_host(dqn.target_W3), 'target_b1': to_host(dqn.target_b1),
        'target_b2': to_host(dqn.target_b2), 'target_b3': to_host(dqn.target_b3),
        'training_steps': dqn.training_steps, 'episodes': dqn.episodes,
        'epsilon': dqn.epsilon, 'training_version': TRAINING_VERSION,
    }
    if include_replay:
        values.update({
            'counter': buffer.counter, 'pointer': buffer.pointer,
            'state_buffer': to_host(buffer.state_buffer),
            'next_state_buffer': to_host(buffer.next_state_buffer),
            'action_buffer': to_host(buffer.action_buffer),
            'reward_buffer': to_host(buffer.reward_buffer),
            'done_buffer': to_host(buffer.done_buffer),
            'champion_counter': buffer.champion_counter,
            'champion_state_buffer': to_host(buffer.champion_state_buffer),
            'champion_next_state_buffer': to_host(buffer.champion_next_state_buffer),
            'champion_action_buffer': to_host(buffer.champion_action_buffer),
            'champion_reward_buffer': to_host(buffer.champion_reward_buffer),
            'champion_done_buffer': to_host(buffer.champion_done_buffer),
        })
    try:
        np.savez_compressed(temp_path, **values)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        for name in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
            value = checkpoint[name]
            target = getattr(dqn, name)
            if name == 'W1' and value.shape == (target.shape[0] - 1, target.shape[1]):
                target[:-1] = xp.asarray(value, dtype=target.dtype)
                target[-1] = 0
            elif value.shape == target.shape:
                target[...] = xp.asarray(value, dtype=target.dtype)
            else:
                raise ValueError('Invalid checkpoint shape for ' + name)
        target_names = ('target_W1', 'target_W2', 'target_W3',
                        'target_b1', 'target_b2', 'target_b3')
        if all(name in checkpoint.files for name in target_names):
            for name in target_names:
                target = getattr(dqn, name)
                value = checkpoint[name]
                if name == 'target_W1' and value.shape == (target.shape[0] - 1, target.shape[1]):
                    target[:-1] = xp.asarray(value, dtype=target.dtype)
                    target[-1] = 0
                elif value.shape == target.shape:
                    target[...] = xp.asarray(value, dtype=target.dtype)
                else:
                    raise ValueError('Invalid checkpoint shape for ' + name)
        else:
            dqn.sync_target_network()
        dqn.training_steps = int(checkpoint['training_steps'])
        dqn.episodes = int(checkpoint['episodes'])
        dqn.epsilon = float(checkpoint['epsilon'])
        replay_count = restore_replay_buffer(checkpoint, buffer)
        if replay_count:
            print(json.dumps({'status': 'replay_restored', 'replay_size': replay_count}),
                  file=sys.stderr)


def export_model():
    return {
        'input_dim': dqn.input_dim,
        'hidden_dim': dqn.hidden_dim,
        'output_dim': dqn.output_dim,
        'training_steps': dqn.training_steps,
        'episodes': dqn.episodes,
        'epsilon': dqn.epsilon,
        'backend': BACKEND,
        'training_version': TRAINING_VERSION,
        'weights': {name: to_host(getattr(dqn, name)).tolist()
                    for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')},
    }


def reply(payload):
    print(json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    sys.stdout.flush()


for line in sys.stdin:
    try:
        message = json.loads(line)
        kind = message.get('type')
        if kind == 'status':
            reply({'episodes': dqn.episodes, 'training_steps': dqn.training_steps,
                   'epsilon': dqn.epsilon, 'replay_size': buffer.size,
                   'backend': BACKEND})
        elif kind == 'load':
            load_checkpoint(message['path'])
            reply({'status': 'loaded', 'episodes': dqn.episodes,
                   'training_steps': dqn.training_steps, 'epsilon': dqn.epsilon,
                   'backend': BACKEND,
                   'model': export_model()})
        elif kind == 'train_batch':
            transitions = message['transitions']
            for item in transitions:
                if len(item['state']) == dqn.input_dim - 1:
                    item['state'] = list(item['state']) + [0.0]
                if len(item['next_state']) == dqn.input_dim - 1:
                    item['next_state'] = list(item['next_state']) + [0.0]
            buffer.push_batch(transitions)
            finished_rallies = sum(1 for item in transitions if item['done'])
            dqn.episodes += finished_rallies
            dqn.epsilon = max(
                EPSILON_MIN,
                1.0 - (1.0 - EPSILON_MIN) *
                min(dqn.episodes / EPSILON_DECAY_RALLIES, 1.0),
            )
            available = buffer.size
            updates = max(1, len(transitions) // 64) if available >= BATCH_SIZE else 0
            losses = []
            for _ in range(updates):
                losses.append(float(dqn.train_step(buffer.sample(BATCH_SIZE))))
            reply({'status': 'trained', 'episodes': dqn.episodes,
                   'training_steps': dqn.training_steps, 'epsilon': dqn.epsilon,
                   'replay_size': available,
                   'backend': BACKEND,
                   'loss': sum(losses) / len(losses) if losses else None,
                   })
        elif kind == 'save':
            save_checkpoint(message['path'], bool(message.get('include_replay', True)))
            reply({'status': 'saved', 'path': message['path'], 'episodes': dqn.episodes,
                   'training_steps': dqn.training_steps})
        elif kind == 'export':
            reply({'model': export_model()})
        else:
            raise ValueError('Unknown message type: ' + str(kind))
    except Exception as error:
        reply({'error': str(error)})
