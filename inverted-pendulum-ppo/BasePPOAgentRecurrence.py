from BasePPOAgent import BasePPOAgent

class BasePPOAgentRecurrence(BasePPOAgent):
    def __init__(self, actor, critic):
        super(BasePPOAgentRecurrence, self).__init__(actor, critic)

    def reset_env(self):
        state = self.env.reset()
        return state[:-1]

    def apply_action(self, action):
        obs, reward, done, info = self.env.step([action])
        return obs[:-1], reward, done, info