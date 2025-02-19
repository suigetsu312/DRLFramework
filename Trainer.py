class DRLTrainer:

    def __init__(self,
                 env,
                 agent):
        self.env = env
        self.agent = agent

    def fit(self):

        state = self.env.reset()

        for step in range(self.agent.timestep):

            action = self.agent.choose_action(state)
            next_state, reward, done, _, _= self.env.step(action)

            self.agent.train_step(step, state, action, reward, next_state, done)
            self.agent.update_epsilon(step)

            if done:
                state = self.env.reset()
            else:
                state = next_state
