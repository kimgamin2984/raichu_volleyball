from dqn import DQN, ReplayBuffer
import sys
import json

batch_size = 64

while 1:
    input = sys.stdin.readline()
    
    if input:
        input = json.loads(input)

    dqn = DQN
    buffer = ReplayBuffer

    if input['type'] == 'act':
        predict = dqn.predict(input['state'])
        print({'action':predict})
        sys.stdout.flush()
        # break

    elif input['type'] == 'observe':
        buffer.push(input['state'],input['action'],input['reward'],input['next_reward'],input['done'])
        dqn.train_step(input['state'])
        buffer.sample(batch_size)