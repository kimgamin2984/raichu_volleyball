"""Train DQN from batched GPU simulations of independent volleyball rallies."""

import argparse
import json
import os
import shutil
import time

import numpy as np

from alphachu import MODEL_CARD_URL, MODEL_ENV_CONTRACT, MODEL_SIDE, load_weights
from atomic_files import atomic_write, resolve_saved_path
from dqn import BACKEND, DQN, cp, to_host
from gpu_physics import rollout_training, rollout_summary
from training_replay import NStepReplayBuffer
from training_diagnostics import value_diagnostics, first_policy_divergence

TRAINING_VERSION = 14
EVALUATION_SEED = 1_500_000_000
MAX_BOT_REGRESSION_PP = 5.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=5_000_000)
    parser.add_argument('--env-count', type=int, default=2048)
    parser.add_argument('--max-frames', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--samples-per-transition', type=float, default=1.0)
    parser.add_argument('--target-update-interval', type=int, default=250)
    parser.add_argument('--learning-rate', type=float, default=0.00002)
    parser.add_argument('--random-prefix-frames', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=25000)
    parser.add_argument('--report-every', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--eval-env-count', type=int, default=2048)
    parser.add_argument('--builtin-eval-every', type=int, default=50000)
    parser.add_argument('--epsilon-decay-rallies', type=int, default=300000)
    parser.add_argument('--epsilon-start', type=float, default=0.05)
    parser.add_argument('--epsilon-min', type=float, default=0.01)
    parser.add_argument('--checkpoint-dir', default='checkpoints-v6-alphachu')
    parser.add_argument('--model-dir', default='src/resources/assets/ai')
    parser.add_argument('--alphachu-model', default='checkpoints/alphachu-v1/model.onnx')
    return parser.parse_args()


def save_checkpoint(path, dqn, buffer, include_replay=True):
    values = {
        'W1': to_host(dqn.W1), 'W2': to_host(dqn.W2), 'W3': to_host(dqn.W3),
        'b1': to_host(dqn.b1), 'b2': to_host(dqn.b2), 'b3': to_host(dqn.b3),
        'target_W1': to_host(dqn.target_W1), 'target_W2': to_host(dqn.target_W2),
        'target_W3': to_host(dqn.target_W3), 'target_b1': to_host(dqn.target_b1),
        'target_b2': to_host(dqn.target_b2), 'target_b3': to_host(dqn.target_b3),
        'training_steps': dqn.training_steps, 'episodes': dqn.episodes,
        'epsilon': dqn.epsilon, 'training_version': TRAINING_VERSION,
        'exploration_rallies': dqn.exploration_rallies,
        'gamma': dqn.gamma, 'learning_rate': dqn.lr,
        'target_update_interval': dqn.target_update_interval,
        'n_step': getattr(dqn, 'n_step', 1),
        'value_calibrated': getattr(dqn, 'value_calibrated', False),
    }
    if include_replay:
        values['rollout_seed'] = dqn.rollout_seed
        values.update(buffer.export_state())
    return atomic_write(path, lambda temp_path: np.savez_compressed(temp_path, **values))


def load_checkpoint(path, dqn, buffer):
    requested_path = os.path.abspath(path)
    path = resolve_saved_path(requested_path)
    if path != requested_path:
        print(f'Loading newer recovery checkpoint from {path}.', flush=True)
    with np.load(path, allow_pickle=False) as checkpoint:
        for name in ('W1', 'W2', 'W3', 'b1', 'b2', 'b3'):
            value = checkpoint[name]
            target = getattr(dqn, name)
            if name == 'W1' and value.shape == (target.shape[0] - 1, target.shape[1]):
                target[:-1] = cp.asarray(value, dtype=target.dtype)
                target[-1] = 0
            elif value.shape == target.shape:
                target[...] = cp.asarray(value, dtype=target.dtype)
            else:
                raise ValueError('Invalid checkpoint shape for ' + name)
        target_names = ('target_W1', 'target_W2', 'target_W3',
                        'target_b1', 'target_b2', 'target_b3')
        if all(name in checkpoint.files for name in target_names):
            for name in target_names:
                target = getattr(dqn, name)
                value = checkpoint[name]
                if name == 'target_W1' and value.shape == (target.shape[0] - 1, target.shape[1]):
                    target[:-1] = cp.asarray(value, dtype=target.dtype)
                    target[-1] = 0
                elif value.shape == target.shape:
                    target[...] = cp.asarray(value, dtype=target.dtype)
                else:
                    raise ValueError('Invalid checkpoint shape for ' + name)
        else:
            dqn.sync_target_network()
        dqn.training_steps = int(checkpoint['training_steps'])
        dqn.episodes = int(checkpoint['episodes'])
        dqn.epsilon = float(checkpoint['epsilon'])
        if 'rollout_seed' in checkpoint.files:
            dqn.rollout_seed = int(checkpoint['rollout_seed'])
        elif 'counter' in checkpoint.files:
            dqn.rollout_seed = max(dqn.rollout_seed, dqn.episodes)
        if 'exploration_rallies' in checkpoint.files:
            exploration_age = int(checkpoint['exploration_rallies'])
        elif ('rollout_seed' in checkpoint.files and 'counter' in checkpoint.files
              and int(checkpoint.get('training_version', 0)) == 12):
            # v12 used rollout_seed as the real exploration clock, even while
            # its console epsilon was computed from inherited model rallies.
            exploration_age = int(checkpoint['rollout_seed'])
        else:
            exploration_age = 0
        # Restoring champion weights must not rewind exploration progress.
        dqn.exploration_rallies = max(dqn.exploration_rallies, exploration_age)
        dqn.value_calibrated = bool(checkpoint.get('value_calibrated', False))
        # Inference-only champions deliberately have no replay. Loading their
        # weights during rollback must not silently replace the current buffer.
        if 'nstep_schema_version' in checkpoint.files or 'state_buffer' in checkpoint.files:
            if not buffer.load_state(checkpoint):
                print('Replay schema or n-step settings differ; using fresh replay.', flush=True)


def export_model(dqn):
    return {
        'input_dim': dqn.input_dim,
        'hidden_dim': dqn.hidden_dim,
        'output_dim': dqn.output_dim,
        'training_steps': dqn.training_steps,
        'episodes': dqn.episodes,
        'epsilon': dqn.epsilon,
        'exploration_rallies': dqn.exploration_rallies,
        'backend': BACKEND,
        'training_version': TRAINING_VERSION,
        'value_calibrated': getattr(dqn, 'value_calibrated', False),
        'weights': {name: to_host(getattr(dqn, name)).tolist()
                    for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')},
    }


def write_json_atomic(path, value):
    def write(temp_path):
        with open(temp_path, 'w', encoding='utf-8') as file:
            json.dump(value, file, separators=(',', ':'))
    return atomic_write(path, write)


def epsilon_at(episode, decay_rallies, minimum, start=1.0):
    progress = min(max(episode / decay_rallies, 0.0), 1.0)
    return start - (start - minimum) * progress


def load_inference_model(path):
    try:
        path = resolve_saved_path(path)
        with open(path, 'r', encoding='utf-8') as file:
            model = json.load(file)
        candidate = DQN()
        if (model.get('input_dim') not in (candidate.input_dim, candidate.input_dim - 1) or
                model.get('hidden_dim') != candidate.hidden_dim or
                model.get('output_dim') != candidate.output_dim):
            return None
        for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            value = cp.asarray(model['weights'][name], dtype=cp.float32)
            target = getattr(candidate, name)
            if name == 'W1' and value.shape == (target.shape[0] - 1, target.shape[1]):
                target[:-1] = value
                target[-1] = 0
            elif value.shape == target.shape:
                target[...] = value
            else:
                return None
        candidate.sync_target_network()
        candidate.episodes = int(model.get('episodes', model.get('trained_rallies', 0)))
        candidate.training_steps = int(model.get('training_steps', 0))
        candidate.exploration_rallies = int(model.get('exploration_rallies', 0))
        candidate.value_calibrated = bool(model.get('value_calibrated', False))
        candidate.epsilon = 0.0
        return candidate
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _evaluate(dqn, max_frames, env_count, seed_base, opponent_kind,
              opponent_label, alphachu_weights, random_prefix_frames=0):
    if seed_base is None:
        # Keep the evaluation set fixed so checkpoint comparisons are paired
        # and aren't decided by a lucky batch of initial ball placements.
        seed_base = EVALUATION_SEED
    seed_base %= 2_000_000_000
    started = time.monotonic()
    summary = rollout_summary(
        dqn, seed_base, env_count=env_count, max_frames=max_frames,
        opponent_difficulty=2, epsilon_override=0.0,
        opponent_kind=opponent_kind, alphachu_weights=alphachu_weights,
        random_prefix_frames=random_prefix_frames,
    )
    summary = {name: to_host(value) for name, value in summary.items()}
    rewards = summary['rewards']
    played = len(rewards)
    wins = int(np.count_nonzero(rewards > 0))
    losses = int(np.count_nonzero(rewards < 0))
    ties = max(played - wins - losses, 0)
    win_rate = 100.0 * wins / max(played, 1)
    start_label = 'standard starts' if random_prefix_frames == 0 else 'varied prefixes'
    print(f'Greedy rally evaluation vs {opponent_label} [{start_label}]: '
          f'{wins} rally-points for / {losses} against / {ties} tied '
          f'({win_rate:.1f}% rally win rate) · '
          f'{time.monotonic() - started:.2f}s', flush=True)
    learner_serves = summary['learner_serves'].astype(bool)
    for label, mask in (('Bot serves', learner_serves),
                        (f'{opponent_label} serves', ~learner_serves)):
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        side_wins = int(np.count_nonzero(rewards[mask] > 0))
        side_losses = int(np.count_nonzero(rewards[mask] < 0))
        print(f'  {label}: {side_wins}W/{side_losses}L '
              f'({100.0 * side_wins / count:.1f}%) · '
              f'{np.mean(summary["lengths"][mask]):.1f} frames/rally · '
              f'touches bot {np.mean(summary["learner_hits"][mask]):.1f}, '
              f'opponent {np.mean(summary["opponent_hits"][mask]):.1f} · '
              f'net crossings {np.mean(summary["net_crossings"][mask]):.1f}',
              flush=True)
    return win_rate


def evaluate_against_alphachu(dqn, max_frames, env_count, alphachu_weights,
                              seed_base=None, random_prefix_frames=0):
    return _evaluate(dqn, max_frames, env_count, seed_base, 'alphachu',
                     'AlphaChu-v1', alphachu_weights, random_prefix_frames)


def evaluate_against_builtin(dqn, max_frames, env_count, alphachu_weights,
                             seed_base=None):
    return _evaluate(dqn, max_frames, env_count, seed_base, 'builtin',
                     'original hard AI', alphachu_weights)


def evaluate_suite(dqn, args, alphachu_weights):
    standard = evaluate_against_alphachu(
        dqn, args.max_frames, args.eval_env_count, alphachu_weights,
    )
    varied = evaluate_against_alphachu(
        dqn, args.max_frames, args.eval_env_count, alphachu_weights,
        seed_base=EVALUATION_SEED + 1_000_000,
        random_prefix_frames=args.random_prefix_frames,
    )
    return dict(standard=standard, varied=varied, score=(standard + varied) / 2)


def save_best_bot(dqn, buffer, checkpoint_dir, evaluation, eval_env_count):
    checkpoint_path = os.path.join(checkpoint_dir, 'best-bot.npz')
    save_checkpoint(checkpoint_path, dqn, buffer, include_replay=False)
    model = export_model(dqn)
    model.update({
        'model': 'single-bot',
        'trained_rallies': dqn.episodes,
        'best_opponent_win_rate': evaluation['standard'],
        'varied_start_win_rate': evaluation['varied'],
        'selection_score': evaluation['score'],
        'evaluation': evaluation,
        'evaluation_rallies': eval_env_count,
        'evaluation_unit': 'single-rally outcome (not full-game wins)',
        'training_opponent': 'alphachu-volleyball/alphachu-v1',
        'evaluation_opponent': 'alphachu-volleyball/alphachu-v1',
        'opponent_model_card': MODEL_CARD_URL,
        'opponent_model_side': MODEL_SIDE,
        'opponent_model_contract': MODEL_ENV_CONTRACT,
        'training_version': TRAINING_VERSION,
        'inference_only': True,
    })
    write_json_atomic(os.path.join(checkpoint_dir, 'best-bot.json'), model)
    print(f'New best bot checkpoint at rally {dqn.episodes:,} '
          f'({evaluation["standard"]:.1f}% standard / '
          f'{evaluation["varied"]:.1f}% varied vs AlphaChu-v1).', flush=True)


def promote_best_bot(checkpoint_dir, model_dir, trained_rallies):
    best_checkpoint = resolve_saved_path(os.path.join(checkpoint_dir, 'best-bot.npz'))
    best_model_path = resolve_saved_path(os.path.join(checkpoint_dir, 'best-bot.json'))
    if not (os.path.exists(best_checkpoint) and os.path.exists(best_model_path)):
        return False

    with open(best_model_path, 'r', encoding='utf-8') as file:
        model = json.load(file)
    # Interrupted writes can leave the two best files from different saves.
    # Do not label one policy's NPZ with another policy's score or JSON weights.
    with np.load(best_checkpoint, allow_pickle=False) as checkpoint:
        paired = (
            int(checkpoint['episodes']) == int(model['episodes'])
            and int(checkpoint['training_steps']) == int(model['training_steps'])
            and all(np.array_equal(checkpoint[name], np.asarray(
                model['weights'][name], dtype=checkpoint[name].dtype,
            )) for name in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))
        )
    if not paired:
        print('Best checkpoint and JSON do not describe the same saved policy; '
              'keeping the current gameplay export until a matching pair is saved.',
              flush=True)
        return False

    bot_checkpoint = os.path.join(checkpoint_dir, 'bot.npz')
    atomic_write(bot_checkpoint, lambda temp_path: shutil.copyfile(best_checkpoint, temp_path))

    model['model'] = 'single-bot'
    model['trained_rallies'] = int(model.get('episodes', trained_rallies))
    model['training_run_completed_at_rally'] = trained_rallies
    model['selected_at_rally'] = model['episodes']
    model['inference_only'] = True
    gameplay_path = os.path.abspath(os.path.join(model_dir, 'bot.json'))
    saved_gameplay_path = write_json_atomic(gameplay_path, model)
    if saved_gameplay_path != gameplay_path:
        print(f'Gameplay bot.json is still locked. The selected model is preserved '
              f'at {saved_gameplay_path}; the browser export has not been replaced.',
              flush=True)
        return False
    print(f'Bot selected from its best evaluation at rally '
          f"{model['selected_at_rally']:,} ({model['best_opponent_win_rate']:.1f}% vs AlphaChu-v1).",
          flush=True)
    return True



def configure_learner(dqn, args):
    if dqn is None:
        raise RuntimeError('Could not load the saved inference policy.')
    dqn.lr = args.learning_rate
    dqn.target_update_interval = args.target_update_interval
    dqn.bootstrap_clip = 1.0
    dqn.n_step = args.n_step
    return dqn


def collect_reference(champion, dqn, buffer, args, alphachu_weights):
    """Collect ordinary TD data from the actual selected reference policy."""
    buffer.clear_reference()
    for _ in range(3):
        batch = rollout_training(
            champion, dqn.rollout_seed, env_count=args.env_count,
            max_frames=args.max_frames, epsilon_override=0.0,
            opponent_kind='alphachu', alphachu_weights=alphachu_weights,
            n_step=args.n_step, gamma=dqn.gamma,
            random_prefix_frames=args.random_prefix_frames,
        )
        dqn.rollout_seed += args.env_count
        added = buffer.push_reference_arrays(batch)
        print(f'Reference TD examples: {buffer.reference_size:,} unique transitions '
              f'(+{added:,}); these are replay data, not imitation constraints.', flush=True)
        if added == 0 or buffer.reference_size >= buffer.reference_capacity:
            break


def calibrate_values(dqn, buffer, args, alphachu_weights, evaluation):
    """Shrink an inherited output scale without intentionally changing rankings."""
    if getattr(dqn, 'value_calibrated', False) or not buffer.reference_size:
        return evaluation
    states = buffer.reference_state_buffer[:buffer.reference_size]
    maximum = float(to_host(cp.max(cp.abs(dqn.predict(states)))))
    if not np.isfinite(maximum):
        raise ValueError('Inherited model contains non-finite Q values.')
    factor = min(1.0, 0.95 / max(maximum, 1e-8))
    if factor < 1.0:
        previous_w = dqn.W3.copy()
        previous_b = dqn.b3.copy()
        dqn.W3 *= factor
        dqn.b3 *= factor
        print(f'Calibrating inherited Q output scale by {factor:.6f} '
              f'(observed absolute maximum {maximum:.3f}).', flush=True)
        recalibrated = evaluate_suite(dqn, args, alphachu_weights)
        if (recalibrated['standard'] < evaluation['standard'] or
                recalibrated['varied'] < evaluation['varied']):
            dqn.W3[...] = previous_w
            dqn.b3[...] = previous_b
            print('Output rescaling changed a rollout action near a tie; '
                  'kept the original actor weights. TD targets remain bounded.', flush=True)
        else:
            evaluation = recalibrated
    dqn.sync_target_network()
    dqn.value_calibrated = True
    return evaluation


def print_value_report(dqn, buffer):
    values = value_diagnostics(dqn, buffer)
    if values['sample_count']:
        def display(value, digits):
            return 'non-finite' if value is None else f'{value:.{digits}f}'
        print(f'  Value diagnostics: Q [{display(values["q_min"], 3)}, '
              f'{display(values["q_max"], 3)}] '
              f'· mean |TD| {display(values["td_error_abs_mean"], 4)} '
              f'· minimum action gap {display(values["action_gap"]["min"], 6)}', flush=True)
    return values


def preserve_regression(dqn, champion, buffer, args, alphachu_weights,
                        evaluation, best_evaluation):
    """Keep the failing actor and its experience before any rollback."""
    stem = os.path.join(args.checkpoint_dir,
                        f'regression-rally-{dqn.episodes}-{time.time_ns()}')
    snapshot = save_checkpoint(stem + '.npz', dqn, buffer, include_replay=True)
    report = {
        'training_version': TRAINING_VERSION,
        'candidate_checkpoint': snapshot,
        'candidate_rallies': dqn.episodes,
        'candidate_updates': dqn.training_steps,
        'candidate_evaluation': evaluation,
        'reference_evaluation': best_evaluation,
        'reference_model': export_model(champion),
        'values': value_diagnostics(dqn, buffer),
        'replay': buffer.diagnostics(),
        'training_config': vars(args),
    }
    # The full candidate has already been preserved, even if tracing fails.
    try:
        report['standard_trace'] = first_policy_divergence(
            dqn, champion, alphachu_weights, max_frames=args.max_frames,
            env_count=32, seed_base=EVALUATION_SEED,
        )
        report['varied_trace'] = first_policy_divergence(
            dqn, champion, alphachu_weights, max_frames=args.max_frames,
            env_count=32, seed_base=EVALUATION_SEED + 1_000_000,
            random_prefix_frames=args.random_prefix_frames,
        )
    except Exception as error:
        report['trace_error'] = f'{type(error).__name__}: {error}'
    report_path = write_json_atomic(stem + '.json', report)
    print(f'Preserved regressed model and replay: {snapshot}', flush=True)
    print(f'First-decision and Q diagnostics: {report_path}', flush=True)


def main():
    args = parse_args()
    if BACKEND != 'cupy':
        raise RuntimeError('GPU simulation requires CuPy. Install requirements-gpu.txt in .venv.')
    if min(args.episodes, args.env_count, args.max_frames, args.batch_size,
           args.checkpoint_every, args.report_every, args.eval_every,
           args.eval_env_count, args.builtin_eval_every,
           args.epsilon_decay_rallies, args.n_step, args.target_update_interval) < 1:
        raise ValueError('Batch, frame, learning, and reporting intervals must be positive.')
    if not 0.0 < args.epsilon_min <= args.epsilon_start <= 1.0:
        raise ValueError('Require 0 < --epsilon-min <= --epsilon-start <= 1.')
    if (not np.isfinite(args.samples_per_transition) or args.samples_per_transition <= 0 or
            not np.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Learning rate and samples per transition must be finite and positive.')
    if not 0 <= args.random_prefix_frames < args.max_frames:
        raise ValueError('--random-prefix-frames must be nonnegative and below max-frames.')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    alphachu_weights = load_weights(args.alphachu_model, cp)
    print(f'Loaded pinned AlphaChu-v1 ONNX opponent from {args.alphachu_model} '
          f'· side {MODEL_SIDE} · contract {MODEL_ENV_CONTRACT}.', flush=True)
    latest_path = os.path.join(args.checkpoint_dir, 'latest.npz')
    dqn = configure_learner(DQN(), args)
    buffer = NStepReplayBuffer(state_dim=dqn.input_dim, n_step=args.n_step, gamma=dqn.gamma)
    resumed = os.path.exists(resolve_saved_path(latest_path))
    if resumed:
        load_checkpoint(latest_path, dqn, buffer)
        print(f'Resumed at rally {dqn.episodes:,}; exploration age '
              f'{dqn.exploration_rallies:,} rallies.', flush=True)

    best_model_path = os.path.join(args.checkpoint_dir, 'best-bot.json')
    candidates = [('resumed checkpoint', dqn)] if resumed else []
    for name, path in (
        ('previous v6 champion', best_model_path),
        ('previous AlphaChu champion', 'checkpoints-v5-alphachu/best-bot.json'),
        ('saved gameplay bot', os.path.join(args.model_dir, 'bot.json')),
        ('legacy high model', os.path.join(args.model_dir, 'high.json')),
        ('legacy medium model', os.path.join(args.model_dir, 'medium.json')),
        ('legacy low model', os.path.join(args.model_dir, 'low.json')),
    ):
        candidate = load_inference_model(path)
        if candidate is not None:
            candidates.append((name, configure_learner(candidate, args)))

    # Identical actors need one paired evaluation. Consider the resumed actor
    # first so a tie retains its matching replay and targets.
    seen = set()
    selected = None
    best_evaluation = None
    for name, candidate in candidates:
        fingerprint = tuple(to_host(getattr(candidate, key)).tobytes()
                            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        print(f'Evaluating candidate: {name}.', flush=True)
        evaluation = evaluate_suite(candidate, args, alphachu_weights)
        if best_evaluation is None or evaluation['score'] > best_evaluation['score']:
            best_evaluation = evaluation
            selected = candidate
            print(f'Selected candidate: {name} · combined score '
                  f'{evaluation["score"]:.2f}%.', flush=True)
    if selected is None:
        selected = dqn
        print('No saved actor was found; initializing one bot from fresh weights.', flush=True)
        best_evaluation = evaluate_suite(dqn, args, alphachu_weights)

    if selected is not dqn:
        # Copy an inference actor while preserving monotonic simulator/exploration
        # progress from this run. Old one-step data is never reinterpreted.
        for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            getattr(dqn, key)[...] = getattr(selected, key)
        dqn.sync_target_network()
        dqn.episodes = selected.episodes
        dqn.training_steps = selected.training_steps
        dqn.value_calibrated = getattr(selected, 'value_calibrated', False)
        discarded = buffer.clear_recent()
        buffer.clear_reference()
        if discarded:
            print(f'Changed actor; discarded {discarded:,} recent transitions.', flush=True)
        if not resumed:
            dqn.exploration_rallies = 0
        print(f'Bootstrapped actor from rally {dqn.episodes:,}; '
              'v5 checkpoints remain untouched.', flush=True)

    dqn.epsilon = epsilon_at(dqn.exploration_rallies, args.epsilon_decay_rallies,
                             args.epsilon_min, args.epsilon_start)
    # Always collect from the selected actor, rather than an unrelated actor
    # that happened to obtain the same normal-start score.
    if dqn.episodes < args.episodes:
        collect_reference(dqn, dqn, buffer, args, alphachu_weights)
        best_evaluation = calibrate_values(
            dqn, buffer, args, alphachu_weights, best_evaluation,
        )
    save_best_bot(dqn, buffer, args.checkpoint_dir, best_evaluation, args.eval_env_count)
    champion = configure_learner(load_inference_model(best_model_path), args)
    if champion is None:
        raise RuntimeError('Could not reload the selected champion.')
    # Save the new schema, selected actor, and exploration clock before updates.
    save_checkpoint(latest_path, dqn, buffer, include_replay=True)
    print_value_report(dqn, buffer)

    device_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors='replace')
    print(f'GPU rollout backend: {device_name} · {args.env_count:,} parallel rallies '
          f'· opponent AlphaChu-v1 · {args.n_step}-step Double DQN '
          f'· target {args.episodes:,} rallies.', flush=True)
    print(f'Learner: batch {args.batch_size:,} · {args.samples_per_transition:g} sampled '
          f'items/new transition · target copy every {args.target_update_interval:,} updates '
          f'· SGD lr {dqn.lr:g} · epsilon {dqn.epsilon:.3f} '
          f'({args.epsilon_start:g} → {args.epsilon_min:g}) '
          f'· random legal prefixes up to {args.random_prefix_frames} frames.', flush=True)
    print(f'Evaluation every {args.eval_every:,} rallies: standard and varied starts '
          f'are reported separately; checkpoint every {args.checkpoint_every:,}.', flush=True)

    session_rallies = 0
    session_transitions = 0
    session_samples = 0
    last_saved = last_reported = last_evaluated = last_builtin = 0
    last_report_time = time.monotonic()
    last_report_rallies = 0
    phase_seconds = dict(rollout=0.0, learning=0.0, evaluation=0.0, saving=0.0)
    batch_state_dirty = False
    stopped_for_regression = False
    last_metrics = None
    try:
        while dqn.episodes < args.episodes:
            env_count = min(args.env_count, args.episodes - dqn.episodes)
            phase_started = time.monotonic()
            batch = rollout_training(
                dqn, dqn.rollout_seed, env_count=env_count, max_frames=args.max_frames,
                epsilon_override=dqn.epsilon, opponent_kind='alphachu',
                alphachu_weights=alphachu_weights, n_step=args.n_step, gamma=dqn.gamma,
                random_prefix_frames=args.random_prefix_frames,
            )
            cp.cuda.get_current_stream().synchronize()
            rollout_seconds = time.monotonic() - phase_started
            phase_seconds['rollout'] += rollout_seconds
            transition_count = batch['transition_count']

            batch_state_dirty = True
            phase_started = time.monotonic()
            buffer.push_arrays(batch)
            dqn.episodes += env_count
            dqn.rollout_seed += env_count
            dqn.exploration_rallies += env_count
            session_rallies += env_count
            session_transitions += transition_count
            dqn.epsilon = epsilon_at(dqn.exploration_rallies, args.epsilon_decay_rallies,
                                     args.epsilon_min, args.epsilon_start)

            # Carry fractional minibatches between rollouts instead of silently
            # discarding the sample budget at each rollout boundary.
            wanted_samples = int(session_transitions * args.samples_per_transition)
            updates = max(0, (wanted_samples - session_samples) // args.batch_size)
            for index in range(updates):
                last_update = index == updates - 1
                metrics = dqn.train_step(
                    buffer.sample(args.batch_size), return_loss=False,
                    return_metrics=last_update,
                )
                if last_update:
                    last_metrics = metrics
                session_samples += args.batch_size
            cp.cuda.get_current_stream().synchronize()
            phase_seconds['learning'] += time.monotonic() - phase_started
            batch_state_dirty = False

            if (session_rallies - last_evaluated >= args.eval_every or
                    dqn.episodes >= args.episodes):
                phase_started = time.monotonic()
                evaluation = evaluate_suite(dqn, args, alphachu_weights)
                values = print_value_report(dqn, buffer)
                phase_seconds['evaluation'] += time.monotonic() - phase_started
                standard_drop = best_evaluation['standard'] - evaluation['standard']
                score_drop = best_evaluation['score'] - evaluation['score']
                invalid = (values.get('nonfinite_q_count', 0) > 0 or
                           values.get('nonfinite_td_count', 0) > 0 or
                           (last_metrics is not None and
                            not all(np.isfinite(value) for value in last_metrics.values())))
                if invalid or max(standard_drop, score_drop) >= MAX_BOT_REGRESSION_PP:
                    preserve_regression(
                        dqn, champion, buffer, args, alphachu_weights,
                        evaluation, best_evaluation,
                    )
                    print('Regression guard: restoring the best actor after preserving '
                          'the failing actor, replay, and divergence report; stopping.', flush=True)
                    load_checkpoint(os.path.join(args.checkpoint_dir, 'best-bot.npz'),
                                    dqn, buffer)
                    discarded = buffer.clear_recent()
                    collect_reference(champion, dqn, buffer, args, alphachu_weights)
                    dqn.epsilon = epsilon_at(
                        dqn.exploration_rallies, args.epsilon_decay_rallies,
                        args.epsilon_min, args.epsilon_start,
                    )
                    print(f'Discarded {discarded:,} recent transitions; reference now '
                          'matches the restored champion.', flush=True)
                    stopped_for_regression = True
                elif (evaluation['score'] > best_evaluation['score'] and
                      evaluation['standard'] >= best_evaluation['standard']):
                    best_evaluation = evaluation
                    save_best_bot(dqn, buffer, args.checkpoint_dir,
                                  evaluation, args.eval_env_count)
                    champion = configure_learner(load_inference_model(best_model_path), args)
                    collect_reference(champion, dqn, buffer, args, alphachu_weights)
                last_evaluated = session_rallies

            if session_rallies - last_builtin >= args.builtin_eval_every:
                phase_started = time.monotonic()
                evaluate_against_builtin(
                    dqn, args.max_frames, args.eval_env_count, alphachu_weights,
                )
                phase_seconds['evaluation'] += time.monotonic() - phase_started
                last_builtin = session_rallies

            if (session_rallies - last_saved >= args.checkpoint_every or
                    stopped_for_regression):
                phase_started = time.monotonic()
                save_checkpoint(latest_path, dqn, buffer, include_replay=True)
                phase_seconds['saving'] += time.monotonic() - phase_started
                last_saved = session_rallies

            if (session_rallies - last_reported >= args.report_every or
                    dqn.episodes >= args.episodes or stopped_for_regression):
                now = time.monotonic()
                rate = (session_rallies - last_report_rallies) / max(now - last_report_time, 1e-6)
                print(f'Rallies {dqn.episodes:,}/{args.episodes:,} · run +{session_rallies:,} '
                      f'· updates {dqn.training_steps:,} · epsilon {dqn.epsilon:.3f} '
                      f'· replay {buffer.recent_size:,} recent + '
                      f'{buffer.reference_size:,} unique reference · {rate:,.1f} rallies/s overall '
                      f'· {env_count / max(rollout_seconds, 1e-6):,.1f} rollout rallies/s '
                      f'· {transition_count / env_count:.1f} frames/rally '
                      f'· sample ratio {session_samples / max(session_transitions, 1):.3f}.',
                      flush=True)
                if last_metrics:
                    print(f'  TD loss {last_metrics["loss"]:.5f} '
                          f'· |TD| {last_metrics["mean_abs_td"]:.4f} '
                          f'· batch Q [{last_metrics["min_q"]:.3f}, '
                          f'{last_metrics["max_q"]:.3f}]', flush=True)
                print('  Time since report: '
                      + ' · '.join(f'{name} {seconds:.2f}s'
                                   for name, seconds in phase_seconds.items()), flush=True)
                last_reported = last_report_rallies = session_rallies
                last_report_time = now
                phase_seconds = dict.fromkeys(phase_seconds, 0.0)
            if stopped_for_regression:
                break
    except KeyboardInterrupt:
        if batch_state_dirty:
            print('\nInterrupted during an update batch; retaining the last complete '
                  'resumable checkpoint.', flush=True)
        else:
            save_checkpoint(latest_path, dqn, buffer, include_replay=True)
            print('\nSaved a resumable checkpoint. Run npm run train to continue.', flush=True)
        promote_best_bot(args.checkpoint_dir, args.model_dir, dqn.episodes)
        return

    save_checkpoint(latest_path, dqn, buffer, include_replay=True)
    promote_best_bot(args.checkpoint_dir, args.model_dir, dqn.episodes)


if __name__ == '__main__':
    main()
