from dqn import DQN, ReplayBuffer
import sys
import json

batch_size = 64

dqn = DQN()
buffer = ReplayBuffer()

while 1:
    input = sys.stdin.readline()
    
    if input:
        input = json.loads(input)
    
    elif input == '':
        break

    if input['type'] == 'act':
        action = dqn.get_action(input['state'])
        print(json.dumps({'action': action}))
        sys.stdout.flush()

    elif input['type'] == 'observe':
        buffer.push(input['state'],input['action'],input['reward'],input['next_state'],input['done'])
        if batch_size <= buffer.counter:
            print(json.dumps({
                              'status': 'trained',
                              'loss': float(dqn.train_step(buffer.sample(batch_size)))
                              }))
        else:
            print(json.dumps({
                              'status': 'buffered',
                              'loss': None
                              }))    
        sys.stdout.flush()
    
    else:
        raise TypeError('Unknown message')