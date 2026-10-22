"""Batched CUDA rollout kernel for the training subset of Pikachu Volleyball physics."""

import cupy as cp


CUDA_SOURCE = r'''
typedef unsigned int uint32_t;

static const float GROUND_WIDTH = 432.0f;
static const float HALF_GROUND = 216.0f;
static const float PLAYER_HALF = 32.0f;
static const float PLAYER_GROUND_Y = 244.0f;
static const float BALL_RADIUS = 20.0f;
static const float BALL_GROUND_Y = 252.0f;
static const int LOOP_LIMIT = 1000;

struct Player {
    float x, y, yv;
    int state, diving, lying, collision;
    int frame, delay, swing, standBy;
    int powerHitPrevious;
};

struct Ball {
    float x, y, xv, yv, expectedX, punchX;
    float previousX, previousY, previousPreviousX, previousPreviousY;
    int isPowerHit;
};

__device__ __forceinline__ int int_abs(int value) {
    return value < 0 ? -value : value;
}

__device__ __forceinline__ uint32_t random_u32(uint32_t *seed) {
    uint32_t x = *seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    if (x == 0) x = 0x6d2b79f5u;
    *seed = x;
    return x;
}

__device__ __forceinline__ int random_mod(uint32_t *seed, int n) {
    return (int)(random_u32(seed) % (uint32_t)n);
}

__device__ __forceinline__ float random_unit(uint32_t *seed) {
    return (float)(random_u32(seed) >> 8) * (1.0f / 16777216.0f);
}

__device__ __forceinline__ float normalized(float value, float scale) {
    return fmaxf(-1.0f, fminf(1.0f, value / scale));
}

// AlphaChu-v1's own pika-zoo wrapper contract (not the older helpingstar
// wrapper): player ranges x=[0,432], y=[0,244], yv=[-16,16], diving=[-1,1],
// lying=[-1,3], frame=[0,4], delay=[0,5]; ball y=[0,304], vy=[-30,30].
__device__ __constant__ float ALPHACHU_OBS_LOW[35] = {
    0,0,-16,-1,-1,0,0,0,0,0,0,0,0,
    0,0,-16,-1,-1,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,-20,-30,0
};
__device__ __constant__ float ALPHACHU_OBS_HIGH[35] = {
    432,244,16,1,3,4,5,1,1,1,1,1,1,
    432,244,16,1,3,4,5,1,1,1,1,1,1,
    432,304,432,304,432,304,20,30,1
};

// AlphaChu-v1 P1 action ids from alphachu-volleyball/pika-zoo's P1_MAP.
// Each row is x direction, y direction, and held power-hit key.
__device__ __constant__ int ALPHACHU_P1_ACTIONS[13][3] = {
    { 0,  0, 0}, // 0 NOOP
    { 0,  0, 1}, // 1 FIRE
    { 0, -1, 0}, // 2 UP
    { 1,  0, 0}, // 3 RIGHT / toward net
    {-1,  0, 0}, // 4 LEFT / away from net
    { 0,  1, 0}, // 5 DOWN
    { 1, -1, 0}, // 6 UP+RIGHT
    {-1, -1, 0}, // 7 UP+LEFT
    { 1,  1, 0}, // 8 DOWN+RIGHT
    {-1,  1, 0}, // 9 DOWN+LEFT
    { 0, -1, 1}, // 10 UP+FIRE
    { 1,  0, 1}, // 11 RIGHT+FIRE
    {-1,  0, 1}  // 12 LEFT+FIRE
};

__device__ float expected_landing(float x, float y, float xv, float yv) {
    for (int loop = 1; loop <= LOOP_LIMIT; ++loop) {
        float futureX = x + xv;
        if (futureX < BALL_RADIUS || futureX > GROUND_WIDTH) xv = -xv;
        if (y + yv < 0.0f) yv = 1.0f;
        if (fabsf(x - HALF_GROUND) < 25.0f && y > 176.0f) {
            if (y < 192.0f) {
                if (yv > 0.0f) yv = -yv;
            } else if (x < HALF_GROUND) {
                xv = -fabsf(xv);
            } else {
                xv = fabsf(xv);
            }
        }
        y += yv;
        if (y > BALL_GROUND_Y || loop >= LOOP_LIMIT) return x;
        x += xv;
        yv += 1.0f;
    }
    return x;
}

__device__ float expected_power_landing(int xDirection, int yDirection, const Ball *ball) {
    float x = ball->x;
    float y = ball->y;
    float xv = x < HALF_GROUND ? (int_abs(xDirection) + 1) * 10.0f
                                : -(int_abs(xDirection) + 1) * 10.0f;
    float yv = fabsf(ball->yv) * (float)yDirection * 2.0f;
    for (int loop = 1; loop <= LOOP_LIMIT; ++loop) {
        float futureX = x + xv;
        if (futureX < BALL_RADIUS || futureX > GROUND_WIDTH) xv = -xv;
        if (y + yv < 0.0f) yv = 1.0f;
        if (fabsf(x - HALF_GROUND) < 25.0f && y > 176.0f && yv > 0.0f) yv = -yv;
        y += yv;
        if (y > BALL_GROUND_Y || loop >= LOOP_LIMIT) return x;
        x += xv;
        yv += 1.0f;
    }
    return x;
}

__device__ __forceinline__ void calculate_expected(Ball *ball) {
    ball->expectedX = expected_landing(ball->x, ball->y, ball->xv, ball->yv);
}

__device__ int move_ball_world(Ball *ball) {
    ball->previousPreviousX = ball->previousX;
    ball->previousPreviousY = ball->previousY;
    ball->previousX = ball->x;
    ball->previousY = ball->y;

    float futureX = ball->x + ball->xv;
    if (futureX < BALL_RADIUS || futureX > GROUND_WIDTH) ball->xv = -ball->xv;
    if (ball->y + ball->yv < 0.0f) ball->yv = 1.0f;
    if (fabsf(ball->x - HALF_GROUND) < 25.0f && ball->y > 176.0f) {
        if (ball->y <= 192.0f) {
            if (ball->yv > 0.0f) ball->yv = -ball->yv;
        } else if (ball->x < HALF_GROUND) {
            ball->xv = -fabsf(ball->xv);
        } else {
            ball->xv = fabsf(ball->xv);
        }
    }
    float futureY = ball->y + ball->yv;
    if (futureY > BALL_GROUND_Y) {
        ball->yv = -ball->yv;
        ball->punchX = ball->x;
        ball->y = BALL_GROUND_Y;
        return 1;
    }
    ball->y = futureY;
    ball->x += ball->xv;
    ball->yv += 1.0f;
    return 0;
}

__device__ void decide_power_hit(Player *player, const Player *other, const Ball *ball,
                                 int *xDirection, int *yDirection, int *powerHit,
                                 uint32_t *seed) {
    if (random_mod(seed, 2) == 0) {
        for (int xDirectionTry = 1; xDirectionTry > -1; --xDirectionTry) {
            for (int yDirectionTry = -1; yDirectionTry < 2; ++yDirectionTry) {
                float landing = expected_power_landing(xDirectionTry, yDirectionTry, ball);
                if ((landing <= 0.0f || landing >= HALF_GROUND) &&
                    fabsf(landing - other->x) > 64.0f) {
                    *xDirection = xDirectionTry;
                    *yDirection = yDirectionTry;
                    *powerHit = 1;
                    return;
                }
            }
        }
    } else {
        for (int xDirectionTry = 1; xDirectionTry > -1; --xDirectionTry) {
            for (int yDirectionTry = 1; yDirectionTry > -2; --yDirectionTry) {
                float landing = expected_power_landing(xDirectionTry, yDirectionTry, ball);
                if ((landing <= 0.0f || landing >= HALF_GROUND) &&
                    fabsf(landing - other->x) > 64.0f) {
                    *xDirection = xDirectionTry;
                    *yDirection = yDirectionTry;
                    *powerHit = 1;
                    return;
                }
            }
        }
    }
    (void)player;
}

__device__ void computer_input(Player *player, const Player *other, const Ball *ball,
                               int boldness, int *xDirection, int *yDirection,
                               int *powerHit, uint32_t *seed) {
    *xDirection = 0;
    *yDirection = 0;
    *powerHit = 0;
    float virtualExpectedX = ball->expectedX;

    if (fabsf(ball->x - player->x) > 100.0f && fabsf(ball->xv) < boldness + 5.0f) {
        if ((ball->expectedX <= 0.0f || ball->expectedX >= HALF_GROUND) && player->standBy == 0)
            virtualExpectedX = 108.0f;
    }
    if (fabsf(virtualExpectedX - player->x) > boldness + 8.0f) {
        *xDirection = player->x < virtualExpectedX ? 1 : -1;
    } else if (random_mod(seed, 20) == 0) {
        player->standBy = random_mod(seed, 2);
    }

    if (player->state == 0) {
        if (fabsf(ball->xv) < boldness + 3.0f &&
            fabsf(ball->x - player->x) < PLAYER_HALF &&
            ball->y > -36.0f && ball->y < 10.0f * boldness + 84.0f && ball->yv > 0.0f)
            *yDirection = -1;

        if (ball->expectedX > 0.0f && ball->expectedX < HALF_GROUND &&
            fabsf(ball->x - player->x) > boldness * 5.0f + 64.0f &&
            ball->x > 0.0f && ball->x < HALF_GROUND && ball->y > 174.0f) {
            *powerHit = 1;
            *xDirection = player->x < ball->x ? 1 : -1;
        }
    } else if (player->state == 1 || player->state == 2) {
        if (fabsf(ball->x - player->x) > 8.0f)
            *xDirection = player->x < ball->x ? 1 : -1;
        if (fabsf(ball->x - player->x) < 48.0f && fabsf(ball->y - player->y) < 48.0f) {
            decide_power_hit(player, other, ball, xDirection, yDirection, powerHit, seed);
            if (*powerHit && fabsf(other->x - player->x) < 80.0f && *yDirection != -1)
                *yDirection = -1;
        }
    }
}

__device__ void move_player(Player *player, int isRight, int xDirection,
                            int yDirection, int powerHit) {
    if (player->state == 4) {
        player->lying -= 1;
        if (player->lying < -1) player->state = 0;
        return;
    }

    float velocityX = 0.0f;
    if (player->state < 5) {
        if (player->state < 3) velocityX = xDirection * 6.0f;
        else velocityX = player->diving * 8.0f;
    }
    float futureX = player->x + velocityX;
    if (!isRight) {
        if (futureX < PLAYER_HALF) player->x = PLAYER_HALF;
        else if (futureX > HALF_GROUND - PLAYER_HALF) player->x = HALF_GROUND - PLAYER_HALF;
        else player->x = futureX;
    } else {
        if (futureX < HALF_GROUND + PLAYER_HALF) player->x = HALF_GROUND + PLAYER_HALF;
        else if (futureX > GROUND_WIDTH - PLAYER_HALF) player->x = GROUND_WIDTH - PLAYER_HALF;
        else player->x = futureX;
    }

    if (player->state < 3 && yDirection == -1 && player->y == PLAYER_GROUND_Y) {
        player->yv = -16.0f;
        player->state = 1;
        player->frame = 0;
    }
    float futureY = player->y + player->yv;
    player->y = futureY;
    if (futureY < PLAYER_GROUND_Y) {
        player->yv += 1.0f;
    } else if (futureY > PLAYER_GROUND_Y) {
        player->yv = 0.0f;
        player->y = PLAYER_GROUND_Y;
        player->frame = 0;
        if (player->state == 3) {
            player->state = 4;
            player->frame = 0;
            player->lying = 3;
        } else {
            player->state = 0;
        }
    }

    if (powerHit) {
        if (player->state == 1) {
            player->delay = 5;
            player->frame = 0;
            player->state = 2;
        } else if (player->state == 0 && xDirection != 0) {
            player->state = 3;
            player->frame = 0;
            player->diving = xDirection;
            player->yv = -5.0f;
        }
    }

    if (player->state == 1) {
        player->frame = (player->frame + 1) % 3;
    } else if (player->state == 2) {
        if (player->delay < 1) {
            player->frame += 1;
            if (player->frame > 4) {
                player->frame = 0;
                player->state = 1;
            }
        } else {
            player->delay -= 1;
        }
    } else if (player->state == 0) {
        player->delay += 1;
        if (player->delay > 3) {
            player->delay = 0;
            int futureFrame = player->frame + player->swing;
            if (futureFrame < 0 || futureFrame > 4) player->swing = -player->swing;
            player->frame += player->swing;
        }
    }
}

__device__ void apply_collision(Ball *ball, Player *player, int xDirection,
                                int yDirection, uint32_t *seed) {
    int happened = fabsf(ball->x - player->x) <= PLAYER_HALF &&
                   fabsf(ball->y - player->y) <= PLAYER_HALF;
    if (happened) {
        if (!player->collision) {
            if (ball->x < player->x) ball->xv = -(int)(fabsf(ball->x - player->x) / 3.0f);
            else if (ball->x > player->x) ball->xv = (int)(fabsf(ball->x - player->x) / 3.0f);
            if (ball->xv == 0.0f) ball->xv = (float)(random_mod(seed, 3) - 1);
            float absYVelocity = fabsf(ball->yv);
            ball->yv = -absYVelocity;
            if (absYVelocity < 15.0f) ball->yv = -15.0f;
            if (player->state == 2) {
                if (ball->x < HALF_GROUND) ball->xv = (int_abs(xDirection) + 1) * 10.0f;
                else ball->xv = -(int_abs(xDirection) + 1) * 10.0f;
                ball->punchX = ball->x;
                ball->yv = fabsf(ball->yv) * yDirection * 2.0f;
                ball->isPowerHit = 1;
            } else {
                ball->isPowerHit = 0;
            }
            calculate_expected(ball);
            player->collision = 1;
        }
    } else {
        player->collision = 0;
    }
}

__device__ void observe(const Player *p1, const Player *p2, const Ball *ball, float *out) {
    out[0] = (GROUND_WIDTH - p2->x) / GROUND_WIDTH;
    out[1] = p2->y / 304.0f;
    out[2] = (GROUND_WIDTH - ball->x) / GROUND_WIDTH;
    out[3] = ball->y / 304.0f;
    out[4] = normalized(-ball->xv, 20.0f);
    out[5] = normalized(ball->yv, 20.0f);
    out[6] = (GROUND_WIDTH - p1->x) / GROUND_WIDTH;
    out[7] = p1->y / 304.0f;
    out[8] = normalized(p2->yv, 12.0f);
    out[9] = p2->state / 6.0f;
    out[10] = (GROUND_WIDTH - ball->expectedX) / GROUND_WIDTH;
}

__device__ int action_allowed(const float *state, int action) {
    float playerX = (1.0f - state[0]) * GROUND_WIDTH;
    int direction = action % 9;
    int xDirection = -((direction / 3) - 1);
    if (playerX >= GROUND_WIDTH - PLAYER_HALF && xDirection > 0) return 0;
    if (playerX <= HALF_GROUND + PLAYER_HALF && xDirection < 0) return 0;
    return 1;
}

__device__ int policy_action(const float *state, float epsilon, uint32_t *seed,
                             const float *W1, const float *b1,
                             const float *W2, const float *b2,
                             const float *W3, const float *b3,
                             int *exploratory) {
    *exploratory = 0;
    if (random_unit(seed) < epsilon) {
        *exploratory = 1;
        int validCount = 0;
        for (int j = 0; j < 18; ++j) validCount += action_allowed(state, j);
        int selected = random_mod(seed, validCount);
        for (int j = 0; j < 18; ++j) {
            if (action_allowed(state, j)) {
                if (selected == 0) return j;
                selected -= 1;
            }
        }
        return 0;
    }
    float layer1[64];
    float layer2[64];
    float output[18];
    for (int j = 0; j < 64; ++j) {
        float v = b1[j];
        for (int i = 0; i < 11; ++i) v += state[i] * W1[i * 64 + j];
        layer1[j] = fmaxf(0.0f, v);
    }
    for (int j = 0; j < 64; ++j) {
        float v = b2[j];
        for (int i = 0; i < 64; ++i) v += layer1[i] * W2[i * 64 + j];
        layer2[j] = fmaxf(0.0f, v);
    }
    for (int j = 0; j < 18; ++j) {
        float v = b3[j];
        for (int i = 0; i < 64; ++i) v += layer2[i] * W3[i * 18 + j];
        output[j] = v;
    }
    int best = -1;
    for (int j = 0; j < 18; ++j) {
        if (action_allowed(state, j) && (best < 0 || output[j] > output[best])) best = j;
    }
    return best;
}

__device__ int alphachu_action(
    const Player *p1, const Player *p2, const Ball *ball,
    const float *W1, const float *b1, const float *W2, const float *b2,
    const float *W3, const float *b3) {
    // Build the raw 35-feature observation in AlphaChu-v1's training format.
    float obs[35];
    obs[0] = p1->x;
    obs[1] = p1->y;
    obs[2] = p1->yv;
    obs[3] = p1->diving;
    obs[4] = p1->lying;
    obs[5] = p1->frame;
    obs[6] = p1->delay;
    for (int i = 0; i < 5; ++i) obs[7 + i] = p1->state == i ? 1.0f : 0.0f;
    obs[12] = p1->powerHitPrevious ? 1.0f : 0.0f;

    obs[13] = p2->x;
    obs[14] = p2->y;
    obs[15] = p2->yv;
    obs[16] = p2->diving;
    obs[17] = p2->lying;
    obs[18] = p2->frame;
    obs[19] = p2->delay;
    for (int i = 0; i < 5; ++i) obs[20 + i] = p2->state == i ? 1.0f : 0.0f;
    obs[25] = p2->powerHitPrevious ? 1.0f : 0.0f;

    obs[26] = ball->x;
    obs[27] = ball->y;
    obs[28] = ball->previousX;
    obs[29] = ball->previousY;
    obs[30] = ball->previousPreviousX;
    obs[31] = ball->previousPreviousY;
    obs[32] = ball->xv;
    obs[33] = ball->yv;
    obs[34] = ball->isPowerHit ? 1.0f : 0.0f;

    // Match AlphaChu's NormalizeObservation: min-max scale and clip each field.
    for (int i = 0; i < 35; ++i) {
        float value = (obs[i] - ALPHACHU_OBS_LOW[i]) /
                      (ALPHACHU_OBS_HIGH[i] - ALPHACHU_OBS_LOW[i]);
        obs[i] = fmaxf(0.0f, fminf(1.0f, value));
    }

    float hidden1[64];
    float hidden2[64];
    float logits[13];
    for (int j = 0; j < 64; ++j) {
        float value = b1[j];
        for (int i = 0; i < 35; ++i) value += obs[i] * W1[j * 35 + i];
        hidden1[j] = tanhf(value);
    }
    for (int j = 0; j < 64; ++j) {
        float value = b2[j];
        for (int i = 0; i < 64; ++i) value += hidden1[i] * W2[j * 64 + i];
        hidden2[j] = tanhf(value);
    }
    for (int j = 0; j < 13; ++j) {
        float value = b3[j];
        for (int i = 0; i < 64; ++i) value += hidden2[i] * W3[j * 64 + i];
        logits[j] = value;
    }

    int best = 0;
    for (int i = 1; i < 13; ++i) {
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

__device__ void alphachu_input(int action, Player *player,
                               int *xDirection, int *yDirection, int *powerHit) {
    // AlphaChu-v1 action 8 is DOWN+RIGHT (no fire), not FRONT+FIRE.
    int safeAction = action < 0 ? 0 : (action > 12 ? 12 : action);
    *xDirection = ALPHACHU_P1_ACTIONS[safeAction][0];
    *yDirection = ALPHACHU_P1_ACTIONS[safeAction][1];
    int powerKeyDown = ALPHACHU_P1_ACTIONS[safeAction][2];

    // Pika-zoo FIRE is a held key; only its rising edge triggers powerHit.
    *powerHit = powerKeyDown && !player->powerHitPrevious;
    player->powerHitPrevious = powerKeyDown;
}

extern "C" __global__ void rollout(
    const float *W1, const float *b1, const float *W2, const float *b2,
    const float *W3, const float *b3,
    const float *alphaW1, const float *alphaB1,
    const float *alphaW2, const float *alphaB2,
    const float *alphaW3, const float *alphaB3,
    float *states, int *actions, float *rewards,
    float *nextStates, float *dones, int *lengths, int *exploratoryFlags,
    int envCount, int maxFrames, int startEpisode,
    int epsilonDecayRallies, float epsilonMin, int opponentDifficulty,
    int opponentKind,
    float epsilonOverride, int explorationStartEpisode, int recordTransitions,
    float *summaryRewards, int *learnerServes, int *learnerHitCounts,
    int *opponentHitCounts, int *netCrossingCounts,
    int randomPrefixFrames, int *prefixFrameCounts,
    const int *scenarioIndices, const int *forcedFrames,
    const int *forcedActions, int *forcedApplied) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= envCount) return;

    int scenario = scenarioIndices[env];
    uint32_t seed = (uint32_t)(startEpisode + scenario + 1) * 747796405u + 2891336453u;
    if (seed == 0) seed = 1u;
    // Training defaults to the strongest opponent; evaluation can select a level.
    int difficulty = opponentDifficulty;
    int boldness = difficulty * 2;
    float epsilon;
    if (epsilonOverride >= 0.0f) {
        epsilon = epsilonOverride;
    } else {
        float progress = fminf(1.0f, (float)(explorationStartEpisode + scenario) /
                                      (float)epsilonDecayRallies);
        epsilon = 1.0f - (1.0f - epsilonMin) * progress;
    }

    Player p1 = {36.0f, PLAYER_GROUND_Y, 0.0f, 0, 0, -1, 0, 0, 0, 1, 0, 0};
    Player p2 = {GROUND_WIDTH - 36.0f, PLAYER_GROUND_Y, 0.0f, 0, 0, -1, 0, 0, 0, 1, 0, 0};
    Ball ball;
    ball.x = random_unit(&seed) < 0.5f ? 56.0f : GROUND_WIDTH - 56.0f;
    ball.y = 0.0f;
    ball.xv = 0.0f;
    ball.yv = 1.0f;
    ball.expectedX = 0.0f;
    ball.punchX = 0.0f;
    ball.previousX = 0.0f;
    ball.previousY = 0.0f;
    ball.previousPreviousX = 0.0f;
    ball.previousPreviousY = 0.0f;
    ball.isPowerHit = 0;
    int learnerServe = ball.x > HALF_GROUND;
    int learnerHits = 0;
    int opponentHits = 0;
    int netCrossings = 0;
    // The prefix uses only its seed and legal actions, so every candidate sees
    // the same opening trajectory. Keep one quarter of starts unchanged.
    int prefixFrames = 0;
    if (randomPrefixFrames > 0 && scenario % 4 != 0) {
        prefixFrames = 1 + random_mod(&seed, randomPrefixFrames);
    }
    prefixFrameCounts[env] = prefixFrames;
    forcedApplied[env] = 0;

    float state[11];
    float nextState[11];
    for (int frame = 0; frame < maxFrames; ++frame) {
        observe(&p1, &p2, &ball, state);
        float frameEpsilon = frame < prefixFrames ? 1.0f : epsilon;
        int exploratory = 0;
        int action = policy_action(state, frameEpsilon, &seed, W1, b1, W2, b2, W3, b3,
                                   &exploratory);
        // Draw the ordinary policy RNG first, then change only this action.
        // Repeated scenario indices therefore reproduce a chosen original
        // environment exactly until its explicitly forced decision.
        int forcedAction = forcedActions[env];
        if (frame == forcedFrames[env] && forcedAction >= 0 && forcedAction < 18 &&
                action_allowed(state, forcedAction)) {
            action = forcedAction;
            exploratory = 1;
            forcedApplied[env] = 1;
        }
        int direction = action % 9;
        int p2xDirection = -((direction / 3) - 1);
        int p2yDirection = (direction % 3) - 1;
        int p2powerHit = action >= 9;

        int p1xDirection = 0, p1yDirection = 0, p1powerHit = 0;
        if (opponentKind == 1) {
            int alphaAction = alphachu_action(
                &p1, &p2, &ball, alphaW1, alphaB1, alphaW2, alphaB2,
                alphaW3, alphaB3);
            alphachu_input(alphaAction, &p1, &p1xDirection,
                           &p1yDirection, &p1powerHit);
        }
        // Expose this frame's held-key state to AlphaChu on the next frame.
        p2.powerHitPrevious = p2powerHit;

        int ballWasOnRight = ball.x > HALF_GROUND;
        int touchedGround = move_ball_world(&ball);
        calculate_expected(&ball);
        if (opponentKind != 1) {
            computer_input(&p1, &p2, &ball, boldness,
                           &p1xDirection, &p1yDirection, &p1powerHit, &seed);
        }
        move_player(&p1, 0, p1xDirection, p1yDirection, p1powerHit);
        move_player(&p2, 1, p2xDirection, p2yDirection, p2powerHit);
        int opponentWasColliding = p1.collision;
        apply_collision(&ball, &p1, p1xDirection, p1yDirection, &seed);
        int learnerWasColliding = p2.collision;
        apply_collision(&ball, &p2, p2xDirection, p2yDirection, &seed);
        if (!recordTransitions) {
            opponentHits += !opponentWasColliding && p1.collision;
            learnerHits += !learnerWasColliding && p2.collision;
            netCrossings += ballWasOnRight != (ball.x > HALF_GROUND);
        }

        int done = touchedGround || frame == maxFrames - 1;
        float reward = touchedGround ? (ball.punchX < HALF_GROUND ? 1.0f : -1.0f) : 0.0f;
        if (recordTransitions) {
            observe(&p1, &p2, &ball, nextState);
            long long sample = (long long)env * maxFrames + frame;
            long long stateBase = sample * 11;
            for (int k = 0; k < 11; ++k) {
                states[stateBase + k] = state[k];
                nextStates[stateBase + k] = nextState[k];
            }
            actions[sample] = action;
            rewards[sample] = reward;
            dones[sample] = done ? 1.0f : 0.0f;
            exploratoryFlags[sample] = exploratory;
        }
        if (done) {
            lengths[env] = frame + 1;
            if (!recordTransitions) {
                summaryRewards[env] = reward;
                learnerServes[env] = learnerServe;
                learnerHitCounts[env] = learnerHits;
                opponentHitCounts[env] = opponentHits;
                netCrossingCounts[env] = netCrossings;
            }
            return;
        }
    }
}
'''

_ROLLOUT_KERNEL = cp.RawKernel(CUDA_SOURCE, 'rollout', options=('--std=c++11',))


NSTEP_SOURCE = r'''
extern "C" __global__ void pack_nstep(
    const float *states, const int *actions, const float *rewards,
    const float *nextStates, const float *dones, const int *lengths,
    const int *exploratoryFlags, const int *prefixFrameCounts,
    const long long *offsets, int envCount, int maxFrames, int nStep,
    float gamma, float *packedStates, int *packedActions,
    float *packedRewards, float *packedNextStates, float *packedDones,
    float *packedDiscounts, float *packedOutcomes, float *packedAfterPrefix) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)envCount * maxFrames) return;
    int env = index / maxFrames;
    int frame = index % maxFrames;
    int remaining = lengths[env] - frame;
    if (remaining <= 0) return;
    int steps = nStep < remaining ? nStep : remaining;
    // A sampled current action is still a valid Q-learning example. Rewards
    // from future exploratory behavior must not masquerade as a greedy return:
    // bootstrap at its preceding next state instead of crossing that action.
    for (int offset = 1; offset < steps; ++offset) {
        if (exploratoryFlags[index + offset]) {
            steps = offset;
            break;
        }
    }
    long long last = index + steps - 1;
    long long packed = offsets[env] + frame;

    float totalReward = 0.0f;
    float discount = 1.0f;
    for (int offset = 0; offset < steps; ++offset) {
        totalReward += discount * rewards[index + offset];
        discount *= gamma;
    }
    // Ground contact is the only true game termination. A max-frame cutoff
    // has zero final reward and keeps the bootstrap from its final next state.
    if (rewards[last] != 0.0f) discount = 0.0f;
    for (int feature = 0; feature < 11; ++feature) {
        packedStates[packed * 11 + feature] = states[index * 11 + feature];
        packedNextStates[packed * 11 + feature] = nextStates[last * 11 + feature];
    }
    packedActions[packed] = actions[index];
    packedRewards[packed] = totalReward;
    packedDones[packed] = dones[last];
    packedDiscounts[packed] = discount;
    packedOutcomes[packed] = rewards[(long long)env * maxFrames + lengths[env] - 1];
    packedAfterPrefix[packed] = frame >= prefixFrameCounts[env] ? 1.0f : 0.0f;
}
'''

_NSTEP_KERNEL = cp.RawKernel(NSTEP_SOURCE, 'pack_nstep', options=('--std=c++11',))


def rollout(dqn, start_episode, env_count=2048, max_frames=1500,
            epsilon_decay_rallies=300000, epsilon_min=0.005,
            opponent_difficulty=2, epsilon_override=None,
            opponent_kind='alphachu', alphachu_weights=None,
            exploration_start=None, random_prefix_frames=0):
    """Simulate independent rallies on the GPU and return packed GPU arrays."""
    arrays, _ = _run_rollout(
        dqn, start_episode, env_count, max_frames, epsilon_decay_rallies,
        epsilon_min, opponent_difficulty, epsilon_override, opponent_kind,
        alphachu_weights, exploration_start, record_transitions=True,
        random_prefix_frames=random_prefix_frames,
    )
    states, actions, rewards, next_states, dones, lengths, _ = arrays
    frame_indices = cp.arange(max_frames, dtype=cp.int32)[None, :]
    valid = frame_indices < lengths[:, None]
    return (
        states[valid], actions[valid], rewards[valid],
        next_states[valid], dones[valid], int(cp.sum(lengths).get()),
    )


def rollout_summary(dqn, start_episode, env_count=2048, max_frames=1500,
                    epsilon_decay_rallies=300000, epsilon_min=0.005,
                    opponent_difficulty=2, epsilon_override=None,
                    opponent_kind='alphachu', alphachu_weights=None,
                    exploration_start=None, random_prefix_frames=0,
                    scenario_indices=None, forced_frames=None, forced_actions=None):
    """Return per-rally GPU metrics without allocating transition histories.

    ``learner_serves`` identifies starts on the right court. Hit counts track
    new ball/player contacts; crossings track changes of the ball's court side.
    """
    _, summary = _run_rollout(
        dqn, start_episode, env_count, max_frames, epsilon_decay_rallies,
        epsilon_min, opponent_difficulty, epsilon_override, opponent_kind,
        alphachu_weights, exploration_start, record_transitions=False,
        random_prefix_frames=random_prefix_frames,
        scenario_indices=scenario_indices, forced_frames=forced_frames,
        forced_actions=forced_actions,
    )
    return summary


def rollout_trace(dqn, start_episode, env_count=2048, max_frames=1500,
                  epsilon_decay_rallies=300000, epsilon_min=0.005,
                  opponent_difficulty=2, epsilon_override=None,
                  opponent_kind='alphachu', alphachu_weights=None,
                  exploration_start=None, random_prefix_frames=0,
                  scenario_indices=None, forced_frames=None, forced_actions=None):
    """Return unpacked GPU histories for comparing opening decisions.

    Frame arrays have dimensions ``(env_count, max_frames, ...)``. Only frames
    below that environment's ``lengths`` entry are initialized and valid.
    """
    arrays, summary = _run_rollout(
        dqn, start_episode, env_count, max_frames, epsilon_decay_rallies,
        epsilon_min, opponent_difficulty, epsilon_override, opponent_kind,
        alphachu_weights, exploration_start, record_transitions=True,
        random_prefix_frames=random_prefix_frames,
        scenario_indices=scenario_indices, forced_frames=forced_frames,
        forced_actions=forced_actions,
    )
    return dict(zip(('states', 'actions', 'rewards', 'next_states', 'dones',
                     'lengths', 'exploratory'), arrays),
                prefix_frames=summary['prefix_frames'],
                forced_applied=summary['forced_applied'])


def rollout_training(dqn, start_episode, env_count=2048, max_frames=1500,
                     epsilon_decay_rallies=300000, epsilon_min=0.005,
                     opponent_difficulty=2, epsilon_override=None,
                     opponent_kind='alphachu', alphachu_weights=None,
                     exploration_start=None, n_step=5, gamma=0.99,
                     random_prefix_frames=16):
    """Return packed GPU N-step transitions and their per-rally outcomes.

    ``discount`` includes the N-step power of gamma and becomes zero only for
    true game termination. ``done`` also marks time-limit cutoffs, whose target
    still bootstraps using ``discount``. Horizons stop before any future
    exploratory action. ``after_prefix`` identifies reference-policy frames
    when collection uses epsilon zero. No transition spans separate rallies.
    """
    if int(n_step) != n_step or n_step < 1:
        raise ValueError('n_step must be a positive integer')
    if not 0.0 <= gamma <= 1.0:
        raise ValueError('gamma must be in the range [0, 1]')
    arrays, summary = _run_rollout(
        dqn, start_episode, env_count, max_frames, epsilon_decay_rallies,
        epsilon_min, opponent_difficulty, epsilon_override, opponent_kind,
        alphachu_weights, exploration_start, record_transitions=True,
        random_prefix_frames=random_prefix_frames,
    )
    states, actions, rewards, next_states, dones, lengths, exploratory = arrays
    offsets = cp.cumsum(lengths, dtype=cp.int64) - lengths
    transition_count = int(cp.sum(lengths).get())
    packed = {
        'state': cp.empty((transition_count, 11), dtype=cp.float32),
        'action': cp.empty(transition_count, dtype=cp.int32),
        'reward': cp.empty(transition_count, dtype=cp.float32),
        'next_state': cp.empty((transition_count, 11), dtype=cp.float32),
        'done': cp.empty(transition_count, dtype=cp.float32),
        'discount': cp.empty(transition_count, dtype=cp.float32),
        'outcome': cp.empty(transition_count, dtype=cp.float32),
        'after_prefix': cp.empty(transition_count, dtype=cp.float32),
        'transition_count': transition_count,
    }
    _NSTEP_KERNEL(
        ((env_count * max_frames + 255) // 256,), (256,),
        (states, actions, rewards, next_states, dones, lengths,
         exploratory, summary['prefix_frames'], offsets,
         cp.int32(env_count), cp.int32(max_frames), cp.int32(n_step),
         cp.float32(gamma), packed['state'], packed['action'], packed['reward'],
         packed['next_state'], packed['done'], packed['discount'], packed['outcome'],
         packed['after_prefix']),
    )
    return packed


def _run_rollout(dqn, start_episode, env_count, max_frames,
                 epsilon_decay_rallies, epsilon_min, opponent_difficulty,
                 epsilon_override, opponent_kind, alphachu_weights,
                 exploration_start, record_transitions, random_prefix_frames=0,
                 scenario_indices=None, forced_frames=None, forced_actions=None):
    if opponent_difficulty not in (0, 1, 2):
        raise ValueError('opponent_difficulty must be 0, 1, or 2')
    if opponent_kind not in ('alphachu', 'builtin'):
        raise ValueError("opponent_kind must be 'alphachu' or 'builtin'")
    if alphachu_weights is None or len(alphachu_weights) != 6:
        raise ValueError('Six AlphaChu ONNX weight tensors are required')
    if int(random_prefix_frames) != random_prefix_frames or random_prefix_frames < 0:
        raise ValueError('random_prefix_frames must be a nonnegative integer')
    scenario_indices = (cp.arange(env_count, dtype=cp.int32) if scenario_indices is None
                        else cp.ascontiguousarray(scenario_indices, dtype=cp.int32))
    forced_frames = (cp.full(env_count, -1, dtype=cp.int32) if forced_frames is None
                     else cp.ascontiguousarray(forced_frames, dtype=cp.int32))
    forced_actions = (cp.full(env_count, -1, dtype=cp.int32) if forced_actions is None
                      else cp.ascontiguousarray(forced_actions, dtype=cp.int32))
    for name, value in (('scenario_indices', scenario_indices),
                        ('forced_frames', forced_frames), ('forced_actions', forced_actions)):
        if value.shape != (env_count,):
            raise ValueError(f'{name} must contain one entry per environment')
    dummy_float = cp.empty(1, dtype=cp.float32)
    dummy_int = cp.empty(1, dtype=cp.int32)
    if record_transitions:
        states = cp.empty((env_count, max_frames, 11), dtype=cp.float32)
        actions = cp.empty((env_count, max_frames), dtype=cp.int32)
        rewards = cp.empty((env_count, max_frames), dtype=cp.float32)
        next_states = cp.empty((env_count, max_frames, 11), dtype=cp.float32)
        dones = cp.empty((env_count, max_frames), dtype=cp.float32)
        exploratory = cp.empty((env_count, max_frames), dtype=cp.int32)
    else:
        states = rewards = next_states = dones = dummy_float
        actions = exploratory = dummy_int
    lengths = cp.zeros(env_count, dtype=cp.int32)
    summary = {
        'rewards': cp.empty(env_count, dtype=cp.float32) if not record_transitions else dummy_float,
        'lengths': lengths,
        'prefix_frames': cp.empty(env_count, dtype=cp.int32),
        'forced_applied': cp.empty(env_count, dtype=cp.int32),
    }
    for name in ('learner_serves', 'learner_hits', 'opponent_hits', 'net_crossings'):
        summary[name] = cp.empty(env_count, dtype=cp.int32) if not record_transitions else dummy_int
    weights = (dqn.W1, dqn.b1, dqn.W2, dqn.b2, dqn.W3, dqn.b3)
    alpha_weights = tuple(cp.ascontiguousarray(value) for value in alphachu_weights)
    _ROLLOUT_KERNEL(
        ((env_count + 127) // 128,), (128,),
        (*weights, *alpha_weights, states, actions, rewards, next_states, dones, lengths,
         exploratory,
         cp.int32(env_count), cp.int32(max_frames), cp.int32(start_episode),
         cp.int32(epsilon_decay_rallies), cp.float32(epsilon_min),
         cp.int32(opponent_difficulty),
         cp.int32(1 if opponent_kind == 'alphachu' else 0),
         cp.float32(-1.0 if epsilon_override is None else epsilon_override),
         cp.int32(start_episode if exploration_start is None else exploration_start),
         cp.int32(1 if record_transitions else 0),
         summary['rewards'], summary['learner_serves'], summary['learner_hits'],
         summary['opponent_hits'], summary['net_crossings'],
         cp.int32(random_prefix_frames), summary['prefix_frames'],
         scenario_indices, forced_frames, forced_actions, summary['forced_applied']),
    )
    return (states, actions, rewards, next_states, dones, lengths, exploratory), summary
