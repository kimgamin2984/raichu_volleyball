'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { spawn } = require('child_process');
const { PikaPhysics, PikaUserInput, GROUND_HALF_WIDTH } = require('./dist-train/physics.cjs');

const ROOT = __dirname;
const TRAINING_VERSION = 9;
const args = process.argv.slice(2);
function option(name, fallback) {
  const prefix = '--' + name + '=';
  const found = args.find((arg) => arg.startsWith(prefix));
  if (found) return found.slice(prefix.length);
  const index = args.indexOf('--' + name);
  return index >= 0 && args[index + 1] ? args[index + 1] : fallback;
}

const targetEpisodes = Number(option('episodes', 1000000));
const syncRallies = Math.max(1, Number(option('sync-rallies', 10)));
const policySyncRallies = Math.max(syncRallies, Number(option('policy-sync-rallies', 100)));
const checkpointEvery = Math.max(1, Number(option('checkpoint-every', 5000)));
const reportEvery = Math.max(1, Number(option('report-every', 1000)));
const maxFrames = Math.max(200, Number(option('max-frames', 1500)));
const checkpointDir = path.resolve(ROOT, option('checkpoint-dir', 'checkpoints-v3'));
const modelDir = path.resolve(ROOT, option('model-dir', 'src/resources/assets/ai'));
const latestPath = path.join(checkpointDir, 'latest.npz');

if (!Number.isInteger(targetEpisodes) || targetEpisodes < 1) {
  throw new Error('Use --episodes N with N >= 1. Default: 1000000.');
}

const localPython = path.join(ROOT, process.platform === 'win32'
  ? '.venv/Scripts/python.exe'
  : '.venv/bin/python');
const python = process.env.PIKA_PYTHON ||
  (fs.existsSync(localPython) ? localPython : (process.platform === 'win32' ? 'py' : 'python3'));
const pythonArgs = python === 'py'
  ? ['-3', '-u', 'main.py']
  : ['-u', 'main.py'];
const worker = spawn(python, pythonArgs, { cwd: ROOT, stdio: ['pipe', 'pipe', 'inherit'] });
const lines = readline.createInterface({ input: worker.stdout });
const pendingReplies = [];
let model = null;
let policyWeights = null;
let epsilon = 1;
let currentEpisodes = 0;
let lastPolicySyncEpisodes = 0;
let stopRequested = false;
let queuedTransitions = [];
let queuedRallies = 0;
let lastReportedEpisodes = -1;
let lastSavedEpisodes = -1;
let requestSequence = Promise.resolve();

process.once('SIGINT', () => {
  stopRequested = true;
  console.log('\nStop requested. Finishing the current rally, then saving a resumable checkpoint...');
});

lines.on('line', (line) => {
  const pending = pendingReplies.shift();
  if (!pending) {
    process.stderr.write('Unexpected Python worker output: ' + line + '\n');
    return;
  }
  try {
    const response = JSON.parse(line);
    if (response.error) pending.reject(new Error(response.error));
    else pending.resolve(response);
  } catch (error) {
    pending.reject(new Error('Invalid DQN worker response: ' + line));
  }
});

function request(payload) {
  const current = new Promise((resolve, reject) => {
    pendingReplies.push({ resolve, reject });
    worker.stdin.write(JSON.stringify(payload) + '\n');
  });
  requestSequence = requestSequence.then(() => current);
  return requestSequence;
}

worker.on('error', (error) => {
  process.stderr.write('Could not start Python (' + python + '): ' + error.message + '\n');
  process.exitCode = 1;
});

function setModel(nextModel) {
  model = nextModel;
  const weights = model.weights;
  const hidden = model.hidden_dim;
  policyWeights = {
    W1: Float32Array.from(weights.W1.flat()),
    W2: Float32Array.from(weights.W2.flat()),
    W3: Float32Array.from(weights.W3.flat()),
    b1: Float32Array.from(weights.b1),
    b2: Float32Array.from(weights.b2),
    b3: Float32Array.from(weights.b3),
    first: new Float32Array(hidden),
    second: new Float32Array(hidden),
    output: new Float32Array(18),
  };
}

function isActionAllowed(state, action) {
  const playerX = (1 - state[0]) * 432;
  const direction = action % 9;
  const xDirection = -(Math.floor(direction / 3) - 1);
  if (playerX >= 400 && xDirection > 0) return false;
  if (playerX <= 248 && xDirection < 0) return false;
  return true;
}

function policyAction(state) {
  const validActions = [];
  for (let action = 0; action < 18; action++) {
    if (isActionAllowed(state, action)) validActions.push(action);
  }
  if (Math.random() < epsilon) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }
  const weights = policyWeights;
  const first = weights.first;
  const second = weights.second;
  const output = weights.output;
  first.set(weights.b1);
  for (let i = 0; i < model.input_dim; i++) {
    const input = state[i];
    const offset = i * first.length;
    for (let j = 0; j < first.length; j++) first[j] += input * weights.W1[offset + j];
  }
  for (let j = 0; j < first.length; j++) first[j] = Math.max(0, first[j]);
  second.set(weights.b2);
  for (let i = 0; i < first.length; i++) {
    const input = first[i];
    const offset = i * second.length;
    for (let j = 0; j < second.length; j++) second[j] += input * weights.W2[offset + j];
  }
  for (let j = 0; j < second.length; j++) second[j] = Math.max(0, second[j]);
  output.set(weights.b3);
  for (let i = 0; i < second.length; i++) {
    const input = second[i];
    const offset = i * output.length;
    for (let j = 0; j < output.length; j++) output[j] += input * weights.W3[offset + j];
  }
  let best = -1;
  for (let i = 0; i < output.length; i++) {
    if (!isActionAllowed(state, i)) continue;
    if (best < 0 || output[i] > output[best]) best = i;
  }
  return best;
}

function stateFor(physics) {
  const player = physics.player2;
  const opponent = physics.player1;
  const ball = physics.ball;
  const normalized = (value, scale) => Math.max(-1, Math.min(1, value / scale));
  return [
    (432 - player.x) / 432,
    player.y / 304,
    (432 - ball.x) / 432,
    ball.y / 304,
    normalized(-ball.xVelocity, 20),
    normalized(ball.yVelocity, 20),
    (432 - opponent.x) / 432,
    opponent.y / 304,
    normalized(player.yVelocity, 12),
    player.state / 6,
    (432 - ball.expectedLandingPointX) / 432,
  ];
}

function actionInput(action, input) {
  const direction = action % 9;
  input.xDirection = -(Math.floor(direction / 3) - 1);
  input.yDirection = (direction % 3) - 1;
  input.powerHit = action >= 9 ? 1 : 0;
}

function playRally(episode) {
  const physics = new PikaPhysics(true, false);
  const opponentDifficulty = 2;
  physics.player1.computerDifficulty = opponentDifficulty;
  physics.player1.initializeForNewRound();
  physics.player2.initializeForNewRound();
  physics.ball.initializeForNewRound(Math.random() < 0.5);
  const opponentInput = new PikaUserInput();
  const agentInput = new PikaUserInput();
  const userInput = [opponentInput, agentInput];
  const transitions = [];

  for (let frame = 0; frame < maxFrames; frame++) {
    const state = stateFor(physics);
    const action = policyAction(state);
    actionInput(action, agentInput);
    const ended = physics.runEngineForNextFrame(userInput);
    const nextState = stateFor(physics);
    const done = ended || frame === maxFrames - 1;
    const reward = ended
      ? (physics.ball.punchEffectX < GROUND_HALF_WIDTH ? 1 : -1)
      : 0;
    transitions.push({ state, action, reward, next_state: nextState, done });
    if (done) return { transitions, frames: frame + 1, reward };
  }
  return { transitions, frames: maxFrames, reward: 0 };
}

async function flushTraining() {
  if (queuedTransitions.length === 0) return;
  const batch = queuedTransitions;
  queuedTransitions = [];
  queuedRallies = 0;
  const result = await request({ type: 'train_batch', transitions: batch });
  epsilon = result.epsilon;
  currentEpisodes = result.episodes;
  const stageBoundary = lastPolicySyncEpisodes < targetEpisodes &&
    currentEpisodes >= targetEpisodes;
  if (stageBoundary || currentEpisodes - lastPolicySyncEpisodes >= policySyncRallies) {
    const exported = await request({ type: 'export' });
    setModel(exported.model);
    lastPolicySyncEpisodes = currentEpisodes;
  }
}

function writeJsonAtomic(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const temporary = filePath + '.tmp';
  fs.writeFileSync(temporary, JSON.stringify(value));
  fs.renameSync(temporary, filePath);
}

async function saveBot() {
  await flushTraining();
  const checkpointPath = path.join(checkpointDir, 'bot.npz');
  await request({ type: 'save', path: checkpointPath, include_replay: false });
  const exported = await request({ type: 'export' });
  setModel(exported.model);
  const inferenceModel = Object.assign({}, model, {
    model: 'single-bot',
    trained_rallies: model.episodes,
    training_opponent: 'original-source-hard',
    training_version: TRAINING_VERSION,
    inference_only: true,
  });
  writeJsonAtomic(path.join(modelDir, 'bot.json'), inferenceModel);
  console.log('Saved single-bot checkpoint at rally ' + currentEpisodes +
    ' (' + model.training_steps + ' training updates).');
}

async function main() {
  fs.mkdirSync(checkpointDir, { recursive: true });
  if (fs.existsSync(latestPath)) {
    const restored = await request({ type: 'load', path: latestPath });
    setModel(restored.model);
    currentEpisodes = restored.episodes;
    lastPolicySyncEpisodes = currentEpisodes;
    epsilon = restored.epsilon;
    lastSavedEpisodes = currentEpisodes;
    console.log('Resumed at rally ' + currentEpisodes + ', epsilon ' + epsilon.toFixed(3) + '.');
  } else {
    const initial = await request({ type: 'export' });
    setModel(initial.model);
    epsilon = model.epsilon;
  }
  lastSavedEpisodes = currentEpisodes;
  console.log('Training backend: ' + model.backend + ' · one bot · target ' + targetEpisodes +
    ' rallies · opponent: original-source hard AI.');

  while (currentEpisodes < targetEpisodes && !stopRequested) {
    const episode = currentEpisodes + queuedRallies + 1;
    const rally = playRally(episode);
    queuedTransitions.push(...rally.transitions);
    queuedRallies++;

    const reachedTarget = episode >= targetEpisodes;
    let didFlush = false;
    if (queuedRallies >= syncRallies || reachedTarget) {
      await flushTraining();
      didFlush = true;
    }
    if ((currentEpisodes % reportEvery === 0 && currentEpisodes !== lastReportedEpisodes) ||
        currentEpisodes >= targetEpisodes) {
      console.log('Rallies ' + currentEpisodes + '/' + targetEpisodes +
        ' · updates ' + model.training_steps + ' · epsilon ' + epsilon.toFixed(3) +
        ' · last point ' + rally.reward + ' in ' + rally.frames + ' frames.');
      lastReportedEpisodes = currentEpisodes;
    }
    if (didFlush && currentEpisodes - lastSavedEpisodes >= checkpointEvery) {
      await request({ type: 'save', path: latestPath, include_replay: true });
      lastSavedEpisodes = currentEpisodes;
    }
  }

  await flushTraining();
  if (stopRequested && currentEpisodes < targetEpisodes) {
    if (currentEpisodes !== lastSavedEpisodes || !fs.existsSync(latestPath)) {
      await request({ type: 'save', path: latestPath, include_replay: true });
      lastSavedEpisodes = currentEpisodes;
    }
    console.log('Stopped safely at rally ' + currentEpisodes +
      '. Run npm run train:cpu again to resume from checkpoints-v3/latest.npz.');
    await saveBot();
    return;
  }
  await saveBot();
  await request({ type: 'save', path: latestPath, include_replay: true });
}

main().catch((error) => {
  process.stderr.write('Offline training failed: ' + error.stack + '\n');
  process.exitCode = 1;
}).finally(() => {
  worker.stdin.end();
});
