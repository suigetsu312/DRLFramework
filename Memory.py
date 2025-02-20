import numpy as np

class CircularQueue:
    def __init__(self, size, shape):
        self.size = size
        shape = (shape,) if isinstance(shape, int) else shape
        self.queue = np.zeros((size, *shape), dtype=np.float32)
        self.top = 0
        self.count = 0

    def enqueue(self, data):
        self.queue[self.top,...] = data
        self.top = (self.top + 1) % self.size
        self.count += 1
        self.count = min(self.count, self.size)
        
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
        self.dones = CircularQueue(max_size, 1)

    def add(self, state, action, reward, next_state, done):
        self.states.enqueue(state)
        self.actions.enqueue(action)
        self.rewards.enqueue(reward)
        self.next_states.enqueue(next_state)
        self.dones.enqueue(done)

    def sample(self, batch_size):
        if self.__len__() < batch_size:
            return None

        indices = np.random.choice(self.__len__(), batch_size, replace=False)

        return (self.states.queue[indices],
                self.actions.queue[indices],
                self.rewards.queue[indices],
                self.next_states.queue[indices],
                self.dones.queue[indices])
    
    def __len__(self):
        return self.states.count
    
if __name__ == "__main__":
    buffer = ReplayBuffer((100,100,3), (2,), (1,))
    for i in range(100):
        buffer.add(np.zeros((100,100,3)), np.random.rand(2), np.random.rand(1), np.zeros((100,100,3)))
    print(buffer.sample(10))  
