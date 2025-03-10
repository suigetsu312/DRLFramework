class DRLTrainer:

    def __init__(self,
                 env,
                 agent,
                 log_freq = 1000):
        self.env = env
        self.agent = agent

    def fit(self):

        state = self.env.reset()[0]
        epoch = 0
        accumulated = 0
        curEpoch_step = 0
        for step in range(self.agent.timeStep):
            self.agent.update_epsilon(step)
            action = self.agent.choose_action(state)
            curEpoch_step += 1
            next_state, reward, done, _, _= self.env.step(action)

            if curEpoch_step >= self.agent.termination_step:
                done = True
            accumulated += reward
            self.agent.train_step(step, state, action, reward, next_state, done)

            if done:
                epoch +=1
                print(f"epoch {epoch} , {step}, {self.env.countOfReachingTarget}: accumulated reward : {accumulated}")
                accumulated = 0
                curEpoch_step = 0
                state = self.env.reset()[0]
            else:
                state = next_state

    def save(self, path):
        self.agent.save_weight(path, self.agent.timestep)