import numpy as np
import gym
import gym.spaces


class DoorOpenNoGripperObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.observation_space.low
        high = env.observation_space.high
        self.observation_space = gym.spaces.Box(
            np.concatenate((low[:3], low[4:])),
            np.concatenate((high[:3], high[4:])),
        )
        
    def observation(self, obs):
        return np.concatenate((obs[:3], obs[4:]))
    
    
class DoorOpenNoGripperControl(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = env.action_space.low
        high = env.action_space.high
        self.action_space = gym.spaces.Box(
            low[:3], high[:3])
        
    def action(self, action):
        return action[:3]


class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, max_length=6000):
        self.cnt = 0
        self.max_length = max_length
        super().__init__(env)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.cnt = 0
        return observation

    def step(self, action):
        self.cnt += 1

        # print(f'Counter is: {self.cnt}')

        observation, reward, done, info = self.env.step(action)
        return observation, reward, self.done(done), info

    def done(self, done):
        if done:
            return True
        if self.cnt >= self.max_length:
            return True
        return False
