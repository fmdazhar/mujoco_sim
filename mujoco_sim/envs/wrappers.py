import time
import gymnasium
import numpy as np
from gymnasium.spaces import Box, flatten_space, flatten
from mujoco_sim.devices.input_utils import input2action
from mujoco_sim.devices.keyboard import Keyboard
from mujoco_sim.devices.mujoco_keyboard import MujocoKeyboard
from mujoco_sim.devices.spacemouse import SpaceMouse


class StackObsWrapper(gymnasium.ObservationWrapper):
    """
    Observation wrapper that stacks the last k observations into a single observation.
    
    The observation space is modified to be a Box with shape (k, *original_shape).
    """
    def __init__(self, env: gymnasium.Env, k: int):
        super(StackObsWrapper, self).__init__(env)
        self.k = k
        self.frames = np.zeros((k,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        low = np.repeat(env.observation_space.low[np.newaxis, ...], k, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], k, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        """
        Reset the environment and initialize the frame stack with the first observation.
        """
        observation, info = self.env.reset(**kwargs)
        self.frames[:] = observation
        return self.frames, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Update the frame stack with the latest observation.
        """
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = observation
        return self.frames


class CustomObsWrapper(gymnasium.ObservationWrapper):
    """
    Observation wrapper that filters the observation dictionary, keeping only selected keys.
    """
    def __init__(self, env: gymnasium.Env, keys_to_keep=None):
        super().__init__(env)
        if keys_to_keep is None:
            original_state_space = self.observation_space["state"]
            self.keys_to_keep = set(original_state_space.spaces.keys())
        else:
            self.keys_to_keep = set(keys_to_keep)

        original_state_space = self.observation_space["state"]
        modified_state_space = gymnasium.spaces.Dict(
            {
                key: space
                for key, space in original_state_space.spaces.items()
                if key in self.keys_to_keep
            }
        )
        self.observation_space = gymnasium.spaces.Dict({"state": modified_state_space})

    def observation(self, observation: dict) -> dict:
        """
        Filter the observation to include only the selected keys.
        """
        observation["state"] = {key: observation["state"][key] for key in self.keys_to_keep}
        return observation


class ObsWrapper(gymnasium.ObservationWrapper):
    """
    Observation wrapper that flattens the 'state' dictionary into a single vector.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),
            }
        )

    def observation(self, obs: dict) -> dict:
        """
        Flatten the 'state' observation.
        """
        obs = {
            "state": flatten(self.env.observation_space["state"], obs["state"]),
        }
        return obs


class GripperCloseEnv(gymnasium.ActionWrapper):
    """
    Action wrapper to enforce a closed gripper during the task.
    
    The action space is reduced to the first 6 dimensions. The gripper value (7th dimension)
    is always set to 1 (closed).
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Map a reduced action to the full action space, ensuring the gripper is closed.
        """
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 6:
            new_action[:6] = action.copy()
        elif action.shape[0] == 7:
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        new_action[6] = 1
        return new_action

    def step(self, action: np.ndarray):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info


class XYZGripperCloseEnv(gymnasium.ActionWrapper):
    """
    Action wrapper to reduce the action space to x, y, z translations,
    while enforcing a closed gripper.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:3], ub.high[:3])

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Map a reduced action (x, y, z) to the full action space. Rotational actions are zeroed.
        """
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 3:
            new_action[:3] = action.copy()
        elif action.shape[0] == 7:
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        new_action[3:6] = 0
        new_action[6] = 1
        return new_action

    def step(self, action: np.ndarray):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info


class XYZQzGripperCloseEnv(gymnasium.ActionWrapper):
    """
    Action wrapper to reduce the action space to x, y, z translations and z-axis rotation,
    while enforcing a closed gripper.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        low = np.concatenate([ub.low[:3], ub.low[5:6]], axis=0)
        high = np.concatenate([ub.high[:3], ub.high[5:6]], axis=0)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Map a reduced action (x, y, z, z-rotation) to the full action space.
        """
        new_action = np.zeros((7,), dtype=np.float32)
        if action.shape[0] == 4:
            new_action[:3] = action[:3]
            new_action[5] = action[3]
        elif action.shape[0] == 7:
            new_action = action.copy()
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")
        new_action[3:5] = 0
        new_action[6] = 1
        return new_action

    def step(self, action: np.ndarray):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info


class SpacemouseIntervention(gymnasium.ActionWrapper):
    """
    Action wrapper for expert intervention using a SpaceMouse or Keyboard.

    If a non-zero expert action is detected, it replaces the policy action.
    Additionally, actions and observations are recorded in a dataset.
    """
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        try:
            self.expert = SpaceMouse()
            self.expert.start_control()
            print("SpaceMouse connected successfully.")
        except OSError:
            print("SpaceMouse not found, falling back to Keyboard.")
            # self.expert = Keyboard()
            self.expert = MujocoKeyboard()

        self.expert.start_control()
        self.last_intervene = 0


    def action(self, action: np.ndarray) -> tuple:
        """
        Determine whether to replace the policy action with an expert action.
        
        Returns:
            A tuple (new_action, replaced) where replaced indicates if expert action was used.
        """
        expert_action = input2action(self.expert)
        if expert_action is None:
            return None, False
        if np.linalg.norm(expert_action[:6]) > 0.00001:
            self.last_intervene = time.time()
        if time.time() - self.last_intervene < 0.5:
            return expert_action, True
        return action, False

    def step(self, action: np.ndarray):
        new_action, replaced = self.action(action)
        if new_action is None:
            obs, info = self.env.reset()
            return obs, 0.0, False, False, info
        else:
            obs, rew, done, truncated, info = self.env.step(new_action)
            if replaced:
                info["intervene_action"] = new_action
            return obs, rew, done, truncated, info
        
    def reset(self, **kwargs):
        """Reset the environment and attach the keyboard callback (if needed)."""
        obs, info = super().reset(**kwargs)

        # Now that the environment is reset, the viewer should exist.
        if isinstance(self.expert, MujocoKeyboard):
            render_mode = getattr(self.env.unwrapped, "render_mode", None)
            if render_mode == "human":
                viewer = getattr(self.env.unwrapped._viewer, "viewer", None)
                # Make sure the viewer actually has 'set_external_key_callback'
                if viewer is not None and hasattr(viewer, "set_external_key_callback"):
                    viewer.set_external_key_callback(self.expert.external_key_callback)
                viewer = self.env.unwrapped._viewer.viewer
                viewer.set_external_key_callback(self.expert.external_key_callback)

        return obs, info
