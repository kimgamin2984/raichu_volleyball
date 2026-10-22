'use strict';

const INPUT_DIM = 11;
const ONLINE_STORAGE_KEY = 'pv-online-bot-v1';
const ONLINE_EPSILON = 0.015;
const ONLINE_GAMMA = 0.995;
const ONLINE_LEARNING_RATE = 0.00002;
const ONLINE_BATCH_SIZE = 16;
const ONLINE_MIN_REPLAY = 64;
const ONLINE_REPLAY_CAPACITY = 12000;
const ONLINE_UPDATES_PER_RALLY_LIMIT = 8;

let model = null;
let baseModel = null;
let baseVersion = null;
let runtimeWeights = null;
let baseWeights = null;
let loadingPromise = null;
let inferenceScratch = null;
let onlineRallies = 0;
let onlineUpdates = 0;
let replay = new Array(ONLINE_REPLAY_CAPACITY);
let replayCounter = 0;
let replayPointer = 0;
let currentRally = [];
let saveCounter = 0;

function normalized(value, scale) {
  return Math.max(-1, Math.min(1, value / scale));
}

function isActionAllowed(state, action) {
  const playerX = (1 - state[0]) * 432;
  const direction = action % 9;
  const xDirection = -(Math.floor(direction / 3) - 1);
  if (playerX >= 400 && xDirection > 0) return false;
  if (playerX <= 248 && xDirection < 0) return false;
  return true;
}

/** State is mirrored so the learned policy always plays from the right side. */
export function getAiState(physics) {
  const player = physics.player2;
  const opponent = physics.player1;
  const ball = physics.ball;
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

export function decodeAction(action) {
  const powerHit = action >= 9 ? 1 : 0;
  const direction = action % 9;
  return {
    xDirection: -(Math.floor(direction / 3) - 1),
    yDirection: (direction % 3) - 1,
    powerHit,
  };
}

function validateModel(value) {
  const hidden = value.hidden_dim;
  const weights = value.weights;
  if (value.input_dim !== INPUT_DIM || value.output_dim !== 18 ||
      !Number.isInteger(hidden) || !weights ||
      !Array.isArray(weights.W1) || weights.W1.length !== INPUT_DIM ||
      !Array.isArray(weights.W2) || weights.W2.length !== hidden ||
      !Array.isArray(weights.W3) || weights.W3.length !== hidden ||
      !Array.isArray(weights.b1) || weights.b1.length !== hidden ||
      !Array.isArray(weights.b2) || weights.b2.length !== hidden ||
      !Array.isArray(weights.b3) || weights.b3.length !== 18 ||
      weights.W1.some((row) => !Array.isArray(row) || row.length !== hidden) ||
      weights.W2.some((row) => !Array.isArray(row) || row.length !== hidden) ||
      weights.W3.some((row) => !Array.isArray(row) || row.length !== 18)) {
    throw new Error('AI checkpoint architecture or weights are invalid');
  }
  return value;
}

function migrateModel(value) {
  if (value.input_dim === INPUT_DIM - 1 && Array.isArray(value.weights?.W1) &&
      value.weights.W1.length === INPUT_DIM - 1) {
    value.weights.W1.push(new Array(value.hidden_dim).fill(0));
    value.input_dim = INPUT_DIM;
    value.added_landing_feature = true;
  }
  return validateModel(value);
}

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value));
}

function makeBaseVersion(value) {
  return [
    value.input_dim,
    value.training_version ?? 0,
    value.training_steps ?? 0,
    value.episodes ?? value.trained_rallies ?? 0,
  ].join(':');
}

function flattenWeights(value) {
  return {
    W1: Float32Array.from(value.weights.W1.flat()),
    b1: Float32Array.from(value.weights.b1),
    W2: Float32Array.from(value.weights.W2.flat()),
    b2: Float32Array.from(value.weights.b2),
    W3: Float32Array.from(value.weights.W3.flat()),
    b3: Float32Array.from(value.weights.b3),
  };
}

function copyWeights(weights) {
  return {
    W1: weights.W1.slice(), b1: weights.b1.slice(),
    W2: weights.W2.slice(), b2: weights.b2.slice(),
    W3: weights.W3.slice(), b3: weights.b3.slice(),
  };
}

function matrixFromFlat(values, rows, columns) {
  const matrix = new Array(rows);
  for (let row = 0; row < rows; row++) {
    matrix[row] = Array.from(values.subarray(row * columns, (row + 1) * columns));
  }
  return matrix;
}

function weightsForStorage() {
  const hidden = model.hidden_dim;
  return {
    W1: matrixFromFlat(runtimeWeights.W1, INPUT_DIM, hidden),
    b1: Array.from(runtimeWeights.b1),
    W2: matrixFromFlat(runtimeWeights.W2, hidden, hidden),
    b2: Array.from(runtimeWeights.b2),
    W3: matrixFromFlat(runtimeWeights.W3, hidden, 18),
    b3: Array.from(runtimeWeights.b3),
  };
}

function makeScratch(hidden) {
  return {
    z1: new Float32Array(hidden), a1: new Float32Array(hidden),
    z2: new Float32Array(hidden), a2: new Float32Array(hidden),
    output: new Float32Array(18),
  };
}

function forward(state, weights, scratch) {
  const hidden = model.hidden_dim;
  const firstWeights = weights.W1;
  for (let j = 0; j < hidden; j++) {
    let value = weights.b1[j];
    for (let i = 0; i < INPUT_DIM; i++) value += state[i] * firstWeights[i * hidden + j];
    scratch.z1[j] = value;
    scratch.a1[j] = Math.max(0, value);
  }
  for (let j = 0; j < hidden; j++) {
    let value = weights.b2[j];
    for (let i = 0; i < hidden; i++) value += scratch.a1[i] * weights.W2[i * hidden + j];
    scratch.z2[j] = value;
    scratch.a2[j] = Math.max(0, value);
  }
  for (let j = 0; j < 18; j++) {
    let value = weights.b3[j];
    for (let i = 0; i < hidden; i++) value += scratch.a2[i] * weights.W3[i * 18 + j];
    scratch.output[j] = value;
  }
  return scratch.output;
}

function serializeOnlineModel() {
  return {
    ...model,
    weights: weightsForStorage(),
    source_model_version: baseVersion,
    online_rallies: onlineRallies,
    online_updates: onlineUpdates,
  };
}

function saveOnlineModel() {
  if (!model || !runtimeWeights || typeof localStorage === 'undefined') return false;
  try {
    localStorage.setItem(ONLINE_STORAGE_KEY, JSON.stringify(serializeOnlineModel()));
    return true;
  } catch (error) {
    console.warn('Could not persist the locally trained AI model.', error);
    return false;
  }
}

function notifyLearningStatus() {
  if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
    window.dispatchEvent(new CustomEvent('pv-ai-learning-update', {
      detail: getAiLearningStatus(),
    }));
  }
}

function setActiveModel(value) {
  model = validateModel(value);
  runtimeWeights = flattenWeights(model);
  inferenceScratch = makeScratch(model.hidden_dim);
}

/** Load the bundled checkpoint and continue any locally saved gameplay learning. */
export function loadAiModel() {
  if (loadingPromise) return loadingPromise;
  loadingPromise = (async () => {
    try {
      const url = new URL('../resources/assets/ai/bot.json', window.location.href);
      const response = await fetch(url);
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const bundled = migrateModel(await response.json());
      baseModel = cloneJson(bundled);
      baseVersion = makeBaseVersion(bundled);
      baseWeights = flattenWeights(baseModel);
      setActiveModel(cloneJson(baseModel));

      try {
        const saved = localStorage.getItem(ONLINE_STORAGE_KEY);
        if (saved) {
          const localModel = JSON.parse(saved);
          if (localModel.source_model_version === baseVersion) {
            setActiveModel(migrateModel(localModel));
            onlineRallies = Math.max(0, Number(localModel.online_rallies) || 0);
            onlineUpdates = Math.max(0, Number(localModel.online_updates) || 0);
          } else {
            localStorage.removeItem(ONLINE_STORAGE_KEY);
          }
        }
      } catch (error) {
        console.warn('Ignoring invalid locally trained AI data.', error);
        try { localStorage.removeItem(ONLINE_STORAGE_KEY); } catch (_) { /* storage may be disabled */ }
        setActiveModel(cloneJson(baseModel));
        onlineRallies = 0;
        onlineUpdates = 0;
      }
      return { ready: true, ...getAiLearningStatus() };
    } catch (error) {
      console.warn('Could not load the single-bot checkpoint.', error);
      return { ready: false, ...getAiLearningStatus() };
    }
  })();
  return loadingPromise;
}

/** Deterministic feed-forward inference with light exploration for online learning. */
export function getAiAction(state, allowExploration = true) {
  if (!model || !runtimeWeights) return null;
  const validActions = [];
  for (let action = 0; action < 18; action++) {
    if (isActionAllowed(state, action)) validActions.push(action);
  }
  if (allowExploration && Math.random() < ONLINE_EPSILON) {
    return validActions[Math.floor(Math.random() * validActions.length)];
  }

  const values = forward(state, runtimeWeights, inferenceScratch);
  let best = -1;
  for (const action of validActions) {
    if (best < 0 || values[action] > values[best]) best = action;
  }
  return best;
}

/** Keep one transition for the current AI-controlled frame until its rally ends. */
export function recordOnlineTransition(state, action, nextState) {
  if (!model || !Number.isInteger(action)) return;
  currentRally.push({
    state: Float32Array.from(state),
    action,
    nextState: Float32Array.from(nextState),
  });
}

export function discardOnlineRally() {
  currentRally = [];
}

function addReplay(item) {
  replay[replayPointer] = item;
  replayPointer = (replayPointer + 1) % ONLINE_REPLAY_CAPACITY;
  replayCounter = Math.min(replayCounter + 1, ONLINE_REPLAY_CAPACITY);
}

function trainOnlineBatch() {
  if (replayCounter < ONLINE_MIN_REPLAY) return false;
  const hidden = model.hidden_dim;
  const gradients = {
    W1: new Float32Array(runtimeWeights.W1.length),
    b1: new Float32Array(runtimeWeights.b1.length),
    W2: new Float32Array(runtimeWeights.W2.length),
    b2: new Float32Array(runtimeWeights.b2.length),
    W3: new Float32Array(runtimeWeights.W3.length),
    b3: new Float32Array(runtimeWeights.b3.length),
  };
  const scratch = makeScratch(hidden);
  const dA1 = new Float32Array(hidden);
  const dA2 = new Float32Array(hidden);

  for (let sample = 0; sample < ONLINE_BATCH_SIZE; sample++) {
    const item = replay[Math.floor(Math.random() * replayCounter)];
    if (!item) continue;
    const output = forward(item.state, runtimeWeights, scratch);
    const error = output[item.action] - item.target;
    const delta = Math.max(-1, Math.min(1, error)); // Huber loss gradient
    gradients.b3[item.action] += delta;
    for (let i = 0; i < hidden; i++) {
      gradients.W3[i * 18 + item.action] += scratch.a2[i] * delta;
      dA2[i] = runtimeWeights.W3[i * 18 + item.action] * delta;
    }
    for (let j = 0; j < hidden; j++) {
      const delta2 = scratch.z2[j] > 0 ? dA2[j] : 0;
      gradients.b2[j] += delta2;
      dA1[j] = 0;
      for (let k = 0; k < hidden; k++) {
        gradients.W2[j * hidden + k] += scratch.a1[j] * (scratch.z2[k] > 0 ? dA2[k] : 0);
        dA1[j] += (scratch.z2[k] > 0 ? dA2[k] : 0) * runtimeWeights.W2[j * hidden + k];
      }
      dA1[j] = scratch.z1[j] > 0 ? dA1[j] : 0;
    }
    for (let i = 0; i < INPUT_DIM; i++) {
      for (let j = 0; j < hidden; j++) {
        gradients.W1[i * hidden + j] += item.state[i] * dA1[j];
      }
    }
    for (let j = 0; j < hidden; j++) gradients.b1[j] += dA1[j];
  }

  for (const name of ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']) {
    const parameters = runtimeWeights[name];
    const anchor = baseWeights[name];
    const gradient = gradients[name];
    for (let i = 0; i < parameters.length; i++) {
      // A small anchor keeps a short set of human games from erasing the base bot.
      const regularizedGradient = gradient[i] / ONLINE_BATCH_SIZE + 0.002 * (parameters[i] - anchor[i]);
      parameters[i] -= ONLINE_LEARNING_RATE * Math.max(-10, Math.min(10, regularizedGradient));
    }
  }
  onlineUpdates++;
  return true;
}

/** Apply the rally result to its recorded actions and persist the updated policy locally. */
export function finishOnlineRally(reward) {
  if (!model || currentRally.length === 0) {
    currentRally = [];
    return getAiLearningStatus();
  }

  let discountedReturn = reward;
  for (let i = currentRally.length - 1; i >= 0; i--) {
    const transition = currentRally[i];
    addReplay({ state: transition.state, action: transition.action, target: discountedReturn });
    discountedReturn *= ONLINE_GAMMA;
  }
  const updates = Math.min(
    ONLINE_UPDATES_PER_RALLY_LIMIT,
    Math.max(1, Math.floor(currentRally.length / 16)),
  );
  currentRally = [];
  onlineRallies++;
  for (let i = 0; i < updates; i++) trainOnlineBatch();

  saveCounter++;
  if (saveCounter >= 1) {
    saveOnlineModel();
    saveCounter = 0;
  }
  notifyLearningStatus();
  return getAiLearningStatus();
}

export function getAiLearningStatus() {
  return {
    onlineRallies,
    onlineUpdates,
    replaySize: replayCounter,
    learning: Boolean(model),
  };
}

/** Restore the bundled weights and discard this browser's online adjustments. */
export function resetOnlineLearning() {
  if (!baseModel) return false;
  setActiveModel(cloneJson(baseModel));
  onlineRallies = 0;
  onlineUpdates = 0;
  replay = new Array(ONLINE_REPLAY_CAPACITY);
  replayCounter = 0;
  replayPointer = 0;
  currentRally = [];
  saveCounter = 0;
  try { localStorage.removeItem(ONLINE_STORAGE_KEY); } catch (_) { /* storage may be disabled */ }
  notifyLearningStatus();
  return true;
}
