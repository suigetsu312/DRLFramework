import numpy as np

class CircularQueue:
    def __init__(self, size, shape):
        self.size = size
        self.queue = np.zeros((size, *shape), dtype=np.float32)
        self.top = 0
        self.count = 0

    def enqueue(self, data):
        self.queue[self.top,:] = data
        self.top = (self.top + 1) % self.size
        self.count += 1
        self.count = min(self.count, self.size)
    
    def sample(self, batch_size):
        if self.count < batch_size:
            return None
        indices = np.random.choice(self.count, batch_size, replace=False)
        return self.queue[indices]
    
class ReplayBuffer:
    def __init__(self,
                 state_shape,
                 action_shape,
                 reward_shape,
                 max_size=int(1e4)):
        self.actions = CircularQueue(max_size, action_shape)
        self.states = CircularQueue(max_size, state_shape)
        self.rewards = CircularQueue(max_size, reward_shape)    
        self.next_states = CircularQueue(max_size, state_shape)

    def add(self, state, action, reward, next_state):
        self.states.enqueue(state)
        self.actions.enqueue(action)
        self.rewards.enqueue(reward)
        self.next_states.enqueue(next_state)

    def sample(self, batch_size):
        self.states.sample(batch_size)
        self.actions.sample(batch_size)
        self.rewards.sample(batch_size)
        self.next_states.sample(batch_size)
        return (self.states.sample(batch_size),
                self.actions.sample(batch_size),
                self.rewards.sample(batch_size),
                self.next_states.sample(batch_size))
    
    def __len__(self):
        return self.states.count
    
if __name__ == "__main__":
    buffer = ReplayBuffer((100,100,3), (2,), (1,))
    for i in range(100):
        buffer.add(np.zeros((100,100,3)), np.random.rand(2), np.random.rand(1), np.zeros((100,100,3)))
    print(buffer.sample(10))  
