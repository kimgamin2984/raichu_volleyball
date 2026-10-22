# Pikachu Volleyball

_&check;_ _English_ | [_Korean(한국어)_](README.ko.md)

Pikachu Volleyball (対戦ぴかちゅ～　ﾋﾞｰﾁﾊﾞﾚｰ編) is an old Windows game that was developed by "(C) SACHI SOFT / SAWAYAKAN Programmers" and "(C) Satoshi Takenouchi" in 1997. The source code on this repository was obtained by reverse engineering the core parts of the machine code &mdash; including the physics engine and the AI &mdash; of the original game and reimplementing them in JavaScript.

You can play this game on the website: https://gorisanson.github.io/pikachu-volleyball/en/

<img src="src/resources/assets/images/screenshot.png" alt="Pikachu Volleyball game screenshot" width="648">

## How to run locally

1. Clone this repository and get into the directory.

```sh
git clone https://github.com/gorisanson/pikachu-volleyball.git
cd pikachu-volleyball
```

2. Install dependencies. (If errors occur, you can try with `node v16` and `npm v8`.)

```sh
npm install
```

3. Bundle the code.

```sh
npm run build
```

4. Run a local web server.

```sh
npx http-server dist
```

5. Connect to the local web server on a web browser. (In most cases, the URL for connecting to the server would be `http://localhost:8080`. For the exact URL, it is supposed to be found on the printed messages on your terminal.)

## Offline DQN training for the single bot

Offline training and gameplay use separate processes. The browser uses its bundled JSON weights, then makes small local-only updates during matches against a human. Those updates are saved in that browser and do not modify the offline checkpoint or shipped `bot.json`. GPU offline training simulates many independent rallies in parallel with a CuPy CUDA kernel. Gameplay continues to use the original JavaScript physics engine.

On Windows with a CUDA 13.x NVIDIA driver, set up the project-local GPU environment and train:

    py -3.12 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    $env:PIKA_DEVICE = 'cuda'
    npm run train

The GPU trainer uses the project-local `.venv` automatically. For the legacy CPU rollout path, install `requirements.txt` and run `npm run train:cpu`; set `PIKA_DEVICE=cpu` to force NumPy.

The GPU trainer uses the fixed [AlphaChu-v1](https://huggingface.co/alphachu-volleyball/alphachu-v1) ONNX policy as its training opponent instead of the built-in hard AI. The model runs locally from `checkpoints/alphachu-v1/model.onnx`: it is downloaded only when missing, verified with SHA-256, and transferred to GPU memory once at process startup. Rally inference does not call a remote service. The AlphaChu adapter follows the model author's `alphachu-volleyball/pika-zoo` contract: player 1's 13 raw-action mapping and clipped min-max observation ranges. These differ from the older `helpingstar/pika-zoo` mapping.

Training version 14 saves to `checkpoints-v6-alphachu/latest.npz`. A new run can initialize its policy from the previous `checkpoints-v5-alphachu/best-bot.json`, evaluated again against AlphaChu. The new replay stores five-step returns and explicit bootstrap discounts, so legacy replay is discarded rather than loaded into this schema. Reference experience has a capacity of 50,000 examples, deduplicated by the complete TD transition, including its return and next state. It supplies TD training examples; it is neither a behavior-cloning objective nor a frozen-policy guarantee.

The GPU defaults are five-step Double DQN, a batch size of 1,024, one sampled training item per collected transition, a target-network refresh every 250 updates, and a learning rate of `0.00002`. GPU bootstrap values and TD targets are bounded to `[-1, 1]` for the terminal `-1/+1` reward. Legacy checkpoints can receive a one-time positive rescaling of their output layer to calibrate oversized Q values; the rescaling is kept only when paired evaluations retain the previous performance. These changes do not establish a 70% win rate.

Exploration starts at `0.05` and decays to `0.01` over 300,000 newly tracked exploration rallies, separately from inherited model rallies and rollout RNG seeds. Training also uses a random legal-action prefix of at most 16 frames to diversify trajectories against the same fixed AlphaChu opponent. Evaluation runs every 10,000 training rallies after a completed GPU batch. It compares both standard starts and starts with the same seeded random prefixes, with exploration disabled after the prefix; the selection score is the mean of the two rally win rates. These are independent single-rally points, not full 15-point games. Summary logs report serve-side win rates, contacts, crossings, Q/TD diagnostics, and time spent in rollout, updates, evaluation, and saving. The original hard AI remains a separate benchmark every 50,000 rallies.

Promotion requires a higher mean selection score with no reduction in standard-start performance. If either the standard-start win rate or the mean score drops at least 5 percentage points below the champion, the trainer saves the failed candidate's full NPZ checkpoint and a JSON diagnostic report, including the first differing policy action where it can be identified, before restoring the best model and stopping. The best evaluated model is exported as `bot.json`. Checkpoints save every 25,000 rallies after a completed GPU batch. Press Ctrl+C to stop and retain a completed resumable checkpoint.

Pass options through npm, for example `npm run train -- --episodes 5000000 --env-count 2048`. `--episodes` is the total model rally target, including inherited rallies; the log's `run +...` counts rallies collected in the current process. The GPU training controls are:

| Option | Default |
| --- | --- |
| `--n-step` | `5` |
| `--batch-size` | `1024` |
| `--samples-per-transition` | `1.0` |
| `--target-update-interval` | `250` |
| `--learning-rate` | `0.00002` |
| `--epsilon-start` / `--epsilon-min` | `0.05` / `0.01` |
| `--epsilon-decay-rallies` | `300000` |
| `--random-prefix-frames` | `16` |
| `--eval-every` / `--checkpoint-every` | `10000` / `25000` |
| `--checkpoint-dir` | `checkpoints-v6-alphachu` |

The base model is `src/resources/assets/ai/bot.json` and is included in the game build. During human matches the bot now makes small online updates from rally outcomes and saves its personalized weights in that browser's `localStorage`; this does not modify the offline checkpoint or the distributable `bot.json`. Use the `Reset gameplay learning` button below the game to restore the bundled model. The policy also receives the physics engine's predicted ball landing point as an explicit input. Launch gameplay with `npm start` or create a production build with `npm run build`. If `bot.json` is missing, gameplay falls back to the original hard built-in AI.

On Windows, checkpoint replacement retries temporary file locks. If the destination remains locked, the completed snapshot is retained beside it under a name such as `latest.npz.pending-*`, and training continues. Restarting selects the most recent completed canonical or recovery file. `.writing-*` files are excluded from automatic recovery. If the gameplay `bot.json` stays locked, the log gives the recovery path and states that the browser export has not been replaced.

## Game structure

- Physics Engine: The physics engine, which calculates the position of the ball and the players (Pikachus), is contained in the file [`src/resources/js/physics.js`](src/resources/js/physics.js). (This file also containes the AI which determines the keyboard input of the computer when you are playing against your computer.) This source code file is gained by reverse engineering the function at the address 00403dd0 of the machine code of the original game.

- Rendering: [PixiJS](https://github.com/pixijs/pixi.js) library is used for rendering the game.

Refer comments on [`src/resources/js/main.js`](src/resources/js/main.js) for other details.

## Methods used for reverse engineering

The main tools used for reverse engineering are following.

- [Ghidra](https://ghidra-sre.org/)
- [Cheat Engine](https://www.cheatengine.org/)
- [OllyDbg](http://www.ollydbg.de/)
- [Resource Hacker](http://www.angusj.com/resourcehacker/)

[Ghidra](https://ghidra-sre.org/) is used for decompiling the machine code to C code. At first look, the decompiled C code looked incomprehensible. One of the reason was that the variable names (`iVar1`, `iVar2`, ...) and function names (`FUN_00402dc0`, `FUN_00403070`, ...) in the decompiled C code are meaningless. But, with the aid of [Cheat Engine](https://www.cheatengine.org/), I could find the location of some significant variables &mdash; x, y coordinate of the ball and the players. And reading from the location of the variables, the decompiled C code was comprehensible! [OllyDbg](http://www.ollydbg.de/) was used for altering a specific part of the machine code. For example, to make slower version of the game so that it would be easier to count the number of frames of "Ready?" message on the start of new round in the game. [Resource Hacker](http://www.angusj.com/resourcehacker/) was used for extract the assets (sprites and sounds) of the game.

## An intended deviation from the original game

If there is no keyboard input, AI vs AI match is started after a while. In the original game, the match lasts only for about 40 seconds. But in this JavaScript version, there's no time limit to the AI vs AI match so you can watch it as long as you want.
