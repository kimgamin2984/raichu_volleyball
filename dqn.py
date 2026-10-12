import numpy as np
# import cupy as np
class DQN():
    def __init__(self):
        self.lr = 0
        self.gamma = 0
        self.epsilon  = 0

        self.input_dim = 10
        self.hidden_dim = 64
        self.output_dim = 32

        self.W1 = np.random.randn(self.input_dim,self.hidden_dim) * np.sqrt(2/self.input_dim)
        self.W2 = np.random.randn(self.hidden_dim,self.hidden_dim) * np.sqrt(2/self.hidden_dim)
        self.W3 = np.random.randn(self.hidden_dim,self.output_dim) * np.sqrt(2/self.hidden_dim)

        self.b1 = np.zeros(self.hidden_dim)
        self.b2 = np.zeros(self.hidden_dim)
        self.b3 = np.zeros(self.output_dim)

    def _forward(self,X):
        # self.input = np.array(X,dtype=np.float16)
        input = np.array(X,dtype=np.float32)

        Z1 = np.dot(input,self.W1)+self.b1
        A1 = np.max(Z1,0)
        Z2 = np.dot(A1,self.W2)+self.b2
        A2 = np.max(Z2,0)
        Q = np.dot(A2,self.W3)+self.b3
        return (Q,Z1,Z2,A1,A2)

    def predict(self,state):
        return self._forward(state)[0]

    def train_step(self,state,target):
        self._forward(state)