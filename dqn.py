import numpy as np
# import cupy as np
class DQN:
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.9
        self.epsilon  = 1.0

        self.input_dim = 10
        self.hidden_dim = 64
        self.output_dim = 32

        self.rng = np.random.default_rng()

        self.W1 = self.rng.standard_normal(size=(self.input_dim,self.hidden_dim)) * np.sqrt(2/self.input_dim)
        self.W2 = self.rng.standard_normal(size=(self.hidden_dim,self.hidden_dim)) * np.sqrt(2/self.hidden_dim)
        self.W3 = self.rng.standard_normal(size=(self.hidden_dim,self.output_dim)) * np.sqrt(2/self.hidden_dim)

        self.b1 = np.zeros(self.hidden_dim)
        self.b2 = np.zeros(self.hidden_dim)
        self.b3 = np.zeros(self.output_dim)

    def _forward(self,X):
        # input = np.array(X,dtype=np.float16)
        input = np.array(X,dtype=np.float32)

        Z1 = np.dot(input,self.W1)+self.b1
        A1 = np.maximum(Z1,0)
        Z2 = np.dot(A1,self.W2)+self.b2
        A2 = np.maximum(Z2,0)
        Z3 = np.dot(A2,self.W3)+self.b3
        return {'Z1':Z1,'Z2':Z2,'Z3':Z3,'A1':A1,'A2':A2}

    def predict(self,X):
        return self._forward(X)['Z3']

    def train_step(self,X,Y):
        # X = np.array(X,dtype=np.float16)
        X = np.array(X,dtype=np.float32)
        # Y = np.array(Y,dtype=np.float16)
        Y = np.array(Y,dtype=np.float32)

        forwarded = self._forward(X)
        
        A1 = forwarded['A1']
        A2 = forwarded['A2']
        Z1 = forwarded['Z1']
        Z2 = forwarded['Z2']
        Z3 = forwarded['Z3']

        dZ3 = (Z3-Y)/X.shape[0]
        dW3 = np.dot(A2.T,dZ3)
        dZ2 = np.dot(dZ3,self.W3.T)*(Z2>0)
        dW2 = np.dot(A1.T,dZ2)
        dZ1 = np.dot(dZ2,self.W2.T)*(Z1>0)
        dW1 = np.dot(X.T,dZ1)
        db1 = np.sum(dZ1,axis=0)
        db2 = np.sum(dZ2,axis=0)
        db3 = np.sum(dZ3,axis=0)

        self.W1 -= self.lr*dW1
        self.W2 -= self.lr*dW2
        self.W3 -= self.lr*dW3
        self.b1 -= self.lr*db1
        self.b2 -= self.lr*db2
        self.b3 -= self.lr*db3

        # Wn = [self.W1,self.W2,self.W3]
        # bn = [self.b1,self.b2,self.b3]
        # dWn = [dW1,dW2,dW3]
        # dbn = [db1,db2,db3]

        # for i in range(3):
        #     Wname = f'W{i+1}'
        #     bname = f'b{i+1}'
        #     setattr(self,Wname,Wn[i] - self.lr*dWn[i])
        #     setattr(self,bname,bn[i] - self.lr*dbn[i])

        return np.mean((Z3-Y)**2)/2

    def get_action(self,X):
        if self.rng.random() <= self.epsilon:
            return int(self.rng.integers(0,self.output_dim))
        else:
            return int(np.argmax(self.predict(X)))

class ReplayBuffer:
    def __init__(self):
        self.capacity = 100000
        self.state_dim = 10

        self.counter = 0
        self.pointer = 0
        
        self.rng = np.random.default_rng()

        self.state_buffer = np.zeros((self.capacity,self.state_dim))
        self.next_state_buffer = np.zeros((self.capacity,self.state_dim))
        self.action_buffer = np.zeros(self.capacity)
        self.reward_buffer  = np.zeros(self.capacity)
        self.done_buffer = np.zeros(self.capacity)

    def push(self,state,action,reward,next_state,done):
        self.state_buffer[self.pointer] = state
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.next_state_buffer[self.pointer] = next_state
        self.done_buffer[self.pointer] = done

        self.counter += 1
        self.pointer = self.counter % self.capacity

    def sample(self,batch_size):
        idx = self.rng.choice(np.minimum(self.counter,self.capacity),size=batch_size,replace=False,axis=0)
        dic = {}
        dic['state'] = self.state_buffer[idx]
        dic['action'] = self.action_buffer[idx]
        dic['reward'] = self.reward_buffer[idx]
        dic['next_state'] = self.next_state_buffer[idx]
        dic['done'] = self.done_buffer[idx]

        return dic