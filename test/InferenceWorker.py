

class DRLInference :

    def __init__(self, env, agent, log_freq = 1000):
        self.env = env
        self.agent = agent

    def run (self): 
        state = self.env.reset()[0]
        done = False
        step = 0
        accumulated = 0
        while not done:
            action = self.agent.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            state = next_state
            accumulated += reward
            step += 1
            self.env.render()
        return accumulated, step
        