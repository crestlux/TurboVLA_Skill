import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
import numpy as np

from collections import deque, defaultdict
import time
def time_decorator(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        out = func(self, *args, **kwargs)
        time_elapsed = time.time() - start_time
        
        if not hasattr(self, "_timings"):
            self._timings = defaultdict(list)
        self._timings[func.__name__].append(time_elapsed)
        
        return out

    return wrapper


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())
        
        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        extra_history_spaces = {}
        state_spaces = self.env.observation_space["state"].spaces
        if "left/wrench_history" in state_spaces:
            extra_history_spaces["left_force_history"] = self.env.observation_space["state"]["left/wrench_history"]
        if "right/wrench_history" in state_spaces:
            extra_history_spaces["right_force_history"] = self.env.observation_space["state"]["right/wrench_history"]
        if "left/state_history" in state_spaces:
            extra_history_spaces["left_state_history"] = self.env.observation_space["state"]["left/state_history"]
        if "right/state_history" in state_spaces:
            extra_history_spaces["right_state_history"] = self.env.observation_space["state"]["right/state_history"]
        

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
                **{f"{key}_depth_image": space for key, space in
                   self.env.observation_space.spaces.get("depth_images", {}).items()},
                **extra_history_spaces,
            }
        )

    @time_decorator
    def observation(self, obs):
        new_obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
            **{f"{key}_depth_image": image for key, image in obs.get("depth_images", {}).items()},
        }
        if "left/wrench_history" in obs["state"]:
            new_obs["left_force_history"] = np.asarray(
                obs["state"]["left/wrench_history"], dtype=np.float32
            )
        if "right/wrench_history" in obs["state"]:
            new_obs["right_force_history"] = np.asarray(
                obs["state"]["right/wrench_history"], dtype=np.float32
            )
        if "left/state_history" in obs["state"]:
            new_obs["left_state_history"] = np.asarray(
                obs["state"]["left/state_history"], dtype=np.float32
            )
        if "right/state_history" in obs["state"]:
            new_obs["right_state_history"] = np.asarray(
                obs["state"]["right/state_history"], dtype=np.float32
            )
        return new_obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
            **{f"{key}_depth_image": image for key, image in obs.get("depth_images", {}).items()},
        }
        return obs
