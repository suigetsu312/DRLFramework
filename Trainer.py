class DRLTraner:

    def __init__(self,
                 env,
                 agent):
        self.env = env
        self.agent = agent

    def fit(self):

        for step in range(self.agent.timestemp):

            action = self.agent.choose_action(state)
            next_state, reward, done, _, _= self.env.step(action)

            self.agent.train_step()
            self.agent.update_epsilon(step)

            if done:
                state = self.env.reset()
            else:
                state = next_state
