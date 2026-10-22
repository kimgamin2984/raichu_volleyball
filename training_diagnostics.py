"""Read-only value and paired-policy diagnostics for volleyball training."""

import json

import numpy as np

from atomic_files import atomic_write
from dqn import to_host, xp


def _host(value):
    return np.asarray(to_host(value))


def _finite_stat(values, operation):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(operation(finite)) if finite.size else None


def _quantiles(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    names = ('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max')
    if not finite.size:
        return dict.fromkeys(names)
    return dict(zip(names, map(float, np.quantile(
        finite, (0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0),
    ))))


def value_diagnostics(dqn, replay, sample_count=8192, batch=None):
    """Measure Q values and Bellman errors without updating the policy.

    ``batch`` can be supplied to examine a particular state distribution. Without
    it, sampling advances only the replay RNG; weights and stored transitions
    are unchanged. An explicit ``discount`` is the complete bootstrap factor,
    including terminal handling, and remains nonzero for time-limit truncation.
    """
    if sample_count < 1:
        raise ValueError('sample_count must be positive')
    if batch is None:
        count = min(int(replay.size), int(sample_count))
        if count == 0:
            return {'sample_count': 0, 'reason': 'Replay is empty.'}
        batch = replay.sample(count)
    else:
        count = min(len(batch['state']), int(sample_count))
        if count == 0:
            return {'sample_count': 0, 'reason': 'Diagnostic batch is empty.'}
        batch = {name: value[:count] for name, value in batch.items()}

    states = xp.asarray(batch['state'], dtype=xp.float32)
    q_values = dqn.predict(states)
    valid = dqn.valid_action_mask(states)
    masked = xp.where(valid, q_values, -xp.inf)
    q_host = _host(q_values)
    ranked = np.sort(_host(masked), axis=1)
    top = ranked[:, -1]
    gaps = top - ranked[:, -2]

    if hasattr(dqn, 'td_targets'):
        targets = dqn.td_targets(batch)
    else:
        # Keep the diagnostic usable with older saved-policy tooling as well.
        next_states = xp.asarray(batch['next_state'], dtype=xp.float32)
        next_online = xp.where(
            dqn.valid_action_mask(next_states), dqn.predict(next_states), -xp.inf,
        )
        next_actions = xp.argmax(next_online, axis=1)
        next_q = dqn.predict_target(next_states)[xp.arange(count), next_actions]
        if 'discount' in batch:
            discount = xp.asarray(batch['discount'], dtype=xp.float32)
        else:
            discount = dqn.gamma * (
                1.0 - xp.asarray(batch['done'], dtype=xp.float32)
            )
        targets = xp.asarray(batch['reward'], dtype=xp.float32) + discount * next_q

    target_host = _host(targets).reshape(-1)
    actions = _host(batch['action']).astype(np.int64)
    selected_q = q_host[np.arange(count), actions]
    td = target_host - selected_q
    abs_td = np.abs(td)
    finite_top = np.isfinite(top)
    return {
        'sample_count': count,
        'q_min': _finite_stat(q_host, np.min),
        'q_max': _finite_stat(q_host, np.max),
        'greedy_q_min': _finite_stat(top, np.min),
        'greedy_q_max': _finite_stat(top, np.max),
        'greedy_q_above_one_count': int(np.count_nonzero(finite_top & (top > 1.0))),
        'greedy_q_above_one_fraction': (
            float(np.mean(top[finite_top] > 1.0)) if finite_top.any() else None
        ),
        'nonfinite_q_count': int(np.count_nonzero(~np.isfinite(q_host))),
        'action_gap': _quantiles(gaps),
        'td_error_abs_mean': _finite_stat(abs_td, np.mean),
        'td_error_abs': _quantiles(abs_td),
        'td_error_mean': _finite_stat(td, np.mean),
        'td_target_min': _finite_stat(target_host, np.min),
        'td_target_max': _finite_stat(target_host, np.max),
        'nonfinite_td_count': int(np.count_nonzero(~np.isfinite(td))),
    }


def _trace_host(trace, env_count, max_frames):
    """Normalize trace arrays without reading unused, uninitialized frames."""
    result = {name: _host(value) for name, value in trace.items()}
    for name in ('states', 'next_states'):
        if name in result and result[name].ndim == 2:
            result[name] = result[name].reshape(env_count, max_frames, -1)
    for name in ('actions', 'rewards', 'dones'):
        if name in result and result[name].ndim == 1:
            result[name] = result[name].reshape(env_count, max_frames)
    result['lengths'] = result['lengths'].astype(np.int64).reshape(env_count)
    if np.any((result['lengths'] < 1) | (result['lengths'] > max_frames)):
        raise ValueError('Trace lengths must be between 1 and max_frames')
    prefix = np.asarray(result.get('prefix_frames', 0), dtype=np.int64)
    result['prefix_frames'] = np.broadcast_to(prefix, (env_count,))
    return result


def _action_values(dqn, state, cuda_action):
    states = xp.asarray(state, dtype=xp.float32)[None, :]
    values = _host(dqn.predict(states))[0]
    valid = _host(dqn.valid_action_mask(states))[0].astype(bool)
    ids = np.flatnonzero(valid)
    ranked = ids[np.argsort(-values[ids], kind='stable')][:2]
    top = [{'action': int(action), 'q': (
        float(values[action]) if np.isfinite(values[action]) else None
    )} for action in ranked]
    gap = float(values[ranked[0]] - values[ranked[1]]) if len(ranked) > 1 else None
    return {
        'cuda_action': int(cuda_action),
        'top_two': top,
        'action_gap': gap if gap is not None and np.isfinite(gap) else None,
    }


def first_policy_divergence(candidate, reference, alphachu_weights,
                            max_frames=1500, env_count=32,
                            seed_base=1_500_000_000, random_prefix_frames=0):
    """Trace two policies and locate their first action difference on shared states.

    The first difference is chosen by earliest frame, then environment index.
    ``first_divergence`` is ``None`` when no comparable action differs. A state
    mismatch before any action difference is reported separately and is never
    presented as an identified policy decision. CUDA actions come from the full
    rollout kernel, while the top-two values use the training prediction path.
    """
    if max_frames < 1 or env_count < 1:
        raise ValueError('max_frames and env_count must be positive')
    # Import lazily so value diagnostics also work on the NumPy backend.
    from gpu_physics import rollout_trace

    options = dict(
        env_count=env_count, max_frames=max_frames, epsilon_override=0.0,
        opponent_kind='alphachu', alphachu_weights=alphachu_weights,
        random_prefix_frames=random_prefix_frames,
    )
    candidate_trace = _trace_host(
        rollout_trace(candidate, seed_base, **options), env_count, max_frames,
    )
    reference_trace = _trace_host(
        rollout_trace(reference, seed_base, **options), env_count, max_frames,
    )
    differences = []
    mismatches = []
    outcomes = []
    for env in range(env_count):
        candidate_length = int(candidate_trace['lengths'][env])
        reference_length = int(reference_trace['lengths'][env])
        outcomes.append({
            'env': env,
            'prefix_frames': int(candidate_trace['prefix_frames'][env]),
            'candidate': {
                'length': candidate_length,
                'reward': float(candidate_trace['rewards'][env, candidate_length - 1]),
            },
            'reference': {
                'length': reference_length,
                'reward': float(reference_trace['rewards'][env, reference_length - 1]),
            },
        })
        common_length = min(candidate_length, reference_length, max_frames)
        candidate_prefix = int(candidate_trace['prefix_frames'][env])
        reference_prefix = int(reference_trace['prefix_frames'][env])
        if candidate_prefix != reference_prefix:
            mismatches.append({
                'env': env, 'frame': 0,
                'reason': 'Random prefix lengths differ between the paired traces.',
            })
            continue
        for frame in range(common_length):
            candidate_state = candidate_trace['states'][env, frame]
            reference_state = reference_trace['states'][env, frame]
            if not np.allclose(candidate_state, reference_state,
                               rtol=1e-5, atol=1e-6, equal_nan=False):
                mismatches.append({
                    'env': env, 'frame': frame,
                    'reason': 'States differ before an action disagreement was identified.',
                })
                break
            candidate_action = int(candidate_trace['actions'][env, frame])
            reference_action = int(reference_trace['actions'][env, frame])
            if candidate_action != reference_action:
                if frame < candidate_prefix:
                    mismatches.append({
                        'env': env, 'frame': frame,
                        'reason': 'Actions differ inside the shared random prefix.',
                    })
                    break
                differences.append((frame, env, candidate_state, reference_state,
                                    candidate_action, reference_action))
                break

    divergence = None
    if differences:
        frame, env, candidate_state, reference_state, candidate_action, \
            reference_action = min(differences, key=lambda item: item[:2])
        divergence = {
            'env': env, 'frame': frame,
            'prefix_frames': int(candidate_trace['prefix_frames'][env]),
            'state': candidate_state.astype(float).tolist(),
            'reference_state': reference_state.astype(float).tolist(),
            'candidate': _action_values(candidate, candidate_state, candidate_action),
            'reference': _action_values(reference, reference_state, reference_action),
            'candidate_outcome': outcomes[env]['candidate'],
            'reference_outcome': outcomes[env]['reference'],
        }
    return {
        'first_divergence': divergence,
        'compared_rallies': env_count,
        'rallies_with_action_divergence': len(differences),
        'state_prefix_mismatches': mismatches,
        'outcomes': outcomes,
    }


def write_diagnostic_report(path, report):
    """Publish a JSON report with the checkpoint writer's lock recovery behavior."""
    def write(temp_path):
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write('\n')
    return atomic_write(path, write)
