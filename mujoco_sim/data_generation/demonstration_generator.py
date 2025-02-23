import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, TypedDict

import gymnasium as gym
import numpy as np
import torch
from mujoco_sim.config.train_config import Config


class StepData(TypedDict):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Dict[str, Any]
    value_function_estimate: torch.Tensor
    gae: torch.Tensor
    next_observation: torch.Tensor


class EpisodeData(TypedDict):
    steps: Dict[str, StepData]


class DemonstrationData(TypedDict):
    """
    Data convention for demonstrations.

    demonstrations = {
        "episode_0": {
            "step_0": {
                "observation": torch.Tensor,        # shape: (observation_space_dim,), torch dtype matches observation_space.dtype
                "action":  torch.Tensor,             # shape: (action_space_dim,), torch dtype matches action_space.dtype
                "reward": torch.Tensor,              # shape: (), torch.float32
                "terminated": torch.Tensor,          # shape: (), torch.bool
                "truncated": torch.Tensor,           # shape: (), torch.bool
                "info": dict,                        # generally empty
                "discounted_return": torch.Tensor,   # shape: (), torch.float32
                "next_observation": torch.Tensor,    # shape: (observation_space_dim,), torch.float32
            },
            "step_1": { ... },
        },
        "episode_1": { ... },
        ...
    }
    """
    episodes: Dict[str, EpisodeData]


class DemonstrationGenerator:
    """
    Class to generate demonstration data from an environment.
    """
    def __init__(self, env: gym.Env, config: Config):
        self.config = config
        self.env = env
        self.discounted_ret_errors = []

    def generate_demonstrations(self, num_episodes: int) -> DemonstrationData:
        """Generate demonstrations over a given number of episodes."""
        num_episodes_actor = num_episodes
        demonstrations: Dict[str, EpisodeData] = {}
        returns = []
        success_rate = 0
        episode_number = 1
        n = -2  # used to index demonstration paths in the configuration

        while episode_number <= num_episodes:
            print(f"Generating episode {episode_number}")
            episode_memory, success, ep_rew = self.generate_one_episode()
            success_rate += success
            returns.append(ep_rew)
            demonstrations[f"episode_{episode_number}"] = episode_memory
            if episode_number % 4 == 0:
                self.save_demonstrations(
                    demonstrations,
                    path=list(self.config.demonstration_data_paths.values())[n],
                )
                demonstrations = {}
                n -= 1
            if not success:
                num_episodes += 1
            print(f"Successful Episodes: {episode_number - (num_episodes - num_episodes_actor)}")
            print()
            episode_number += 1

        return demonstrations

    def generate_one_episode(self) -> tuple[EpisodeData, bool, float]:
        """Generate data for one episode."""
        terminated = False
        truncated = False
        episode_memory: Dict[str, Any] = {}
        step_number = 0
        observation, info = self.env.reset()
        success = 0
        rew = 0

        while not (terminated or truncated):
            action = self.get_action(observation=observation, wait_for_user=(step_number <= 2))
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            success += info.get("is_success", 0)
            if "intervene_action" in info:
                if self.config.env_id == "ur5ePegInHoleGymEnv_medium-v0":
                    action = info["intervene_action"][: self.config.action_dim]
                if self.config.env_id in ["ur5ePegInHoleGymEnv_hard-v0", "ur5ePegInHoleGymEnv_hard-v1"]:
                    action = np.zeros((4,), dtype=np.float32)
                    action[:3] = info["intervene_action"][:3]
                    if len(info["intervene_action"]) > 4:
                        action[3] = info["intervene_action"][5]
                    else:
                        action[3] = info["intervene_action"][-1]
                if self.config.env_id in ["ur5ePegInHoleGymEnv_very_hard-v1"]:
                    action = info["intervene_action"]
            action = action[: self.config.action_dim]
            step_data: StepData = {
                "observation": torch.tensor(self.preprocess_observation(observation), dtype=torch.float32),
                "action": torch.tensor(action, dtype=torch.float32),
                "reward": torch.tensor(reward, dtype=torch.float32),
                "terminated": torch.tensor(terminated, dtype=torch.bool),
                "truncated": torch.tensor(truncated, dtype=torch.bool),
                "info": info,
                "discounted_return": None,
                "next_observation": torch.tensor(self.preprocess_observation(next_observation), dtype=torch.float32),
                "value_function_estimate": torch.tensor(0.0, dtype=torch.float32),
                "gae": torch.tensor(0.0, dtype=torch.float32),
            }
            rew += reward
            episode_memory[f"step_{step_number}"] = step_data
            observation = next_observation
            step_number += 1

        episode_memory = self.compute_discounted_return(episode_memory)
        return episode_memory, success > 0, rew

    def save_demonstrations(self, demonstrations: DemonstrationData, path) -> None:
        """Save demonstrations to a file."""
        if isinstance(path, dict):
            # Convert the dictionary items to a list and save portions
            demonstrations_list = list(demonstrations.items())
            for k, v in path.items():
                torch.save(dict(demonstrations_list[: int(k)]), v)
                size_mb = os.path.getsize(v) / (1024 * 1024)
                print(f"Saved {k} episodes as demonstrations to {v} with size {size_mb:.2f} Mbyte.")
        elif isinstance(path, str):
            torch.save(demonstrations, path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"Saved {len(demonstrations)} episodes as demonstrations to {path} with size {size_mb:.2f} Mbyte.")

    def compute_discounted_return(self, episode_memory: EpisodeData) -> EpisodeData:
        """
        Compute discounted return for an episode.
        
        G_t = R_t + gamma * R_{t+1} + gamma^2 * R_{t+2} + ...
        """
        gamma = self.config.gamma
        rews = np.array([float(step["reward"]) for step in episode_memory.values()])
        min_rew = np.min(rews)
        max_rew = np.max(rews)
        episode_end_rew = {
            "r_min": min_rew,
            "r_max": max_rew,
            "r_mean": (min_rew + max_rew) / 2,
            "r_T": rews[-1],
        }
        bool_truncated = list(episode_memory.values())[-1]["truncated"].bool()

        discounted_return = 0.0
        if bool_truncated:
            discounted_return = episode_end_rew[self.config.rew_bootstrapping] / (1 - gamma)

        # Loop through steps in reverse order to compute discounted return
        for step_number in reversed(range(len(episode_memory))):
            step_key = f"step_{step_number}"
            step_data = episode_memory[step_key]
            reward = step_data["reward"].item()
            discounted_return = reward + gamma * discounted_return
            step_data["discounted_return"] = torch.tensor([discounted_return], dtype=torch.float32)
        
        return episode_memory

    def preprocess_observation(self, observation: dict) -> np.ndarray:
        """Preprocess observation by removing or concatenating keys as necessary."""
        obs = observation.copy()
        if self.env.spec.id[:9] == "FetchPush" and len(observation["observation"]) == 26:
            obs["observation"] = observation["observation"][:-1]
        if isinstance(obs, dict):
            obs = np.concatenate(list(obs.values()), axis=0)
        return obs

    def get_action(self, observation: np.ndarray, wait_for_user: bool) -> np.ndarray:
        """
        Get an action for the current observation.
        
        By default, returns a default action from the key mapping. Subclasses may override.
        """
        # In the base demonstration generator, we simply use a default action.
        return self.key_action_mapping["default"]  # type: ignore


class ManualDemonstrationGenerator(DemonstrationGenerator):
    """
    Demonstration generator that allows manual intervention.
    """
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        key_action_mapping: dict[str, np.ndarray],
        wait_for_input: bool,
    ):
        super().__init__(env=env, config=config)
        self.key_action_mapping = key_action_mapping
        self.wait_for_input = wait_for_input
        self.manual_action_bool = False

    def get_action(self, observation: np.ndarray, wait_for_user: bool) -> np.ndarray:
        """
        Get action from manual input. Currently returns a default action
        and sets manual_action_bool to False.
        """
        action = self.key_action_mapping["default"]
        self.manual_action_bool = False
        return action


class HeuristicDemonstrationGenerator(ManualDemonstrationGenerator):
    """
    Demonstration generator that uses a heuristic spiral controller.
    """
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        key_action_mapping: dict[str, np.ndarray],
        wait_for_input: bool,
        spiral_constant: float,
        Controller,
    ):
        super().__init__(env=env, config=config, key_action_mapping=key_action_mapping, wait_for_input=wait_for_input)
        self.center = np.zeros(6)
        self.phi = np.pi * 10
        self.a = spiral_constant
        self.Controller = Controller
        self.calls = 0

    def spiral_controller(self, observation: np.ndarray) -> np.ndarray:
        """Compute action using a spiral controller based on observation."""
        center_x = self.center[0]
        center_y = self.center[1]
        r = self.a * self.phi
        soll_x = center_x + r * np.cos(self.phi)
        soll_y = center_y + r * np.sin(self.phi)
        action = np.zeros(self.config.action_dim)
        out = self.Controller.update(setpoint=[soll_x, soll_y], measured_value=[observation[0], observation[1]])
        action[0] = out[0]
        action[1] = out[1]
        if observation[8] > 7.4e-02:
            action[0] = 0
            action[1] = 0
            action[2] = -0.5
        else:
            action[2] = (-1 + 2 * (observation[-4] < -15)) * abs(-15 - observation[-4]) / 20
        if observation[8] < 7.34e-02:
            action[0] = -0.5 + (observation[-6] < 0)
            action[1] = -0.5 + (observation[-5] < 0)
            action[2] = -1
        return action

    def get_action(self, observation: np.ndarray, wait_for_user: bool) -> np.ndarray:
        """Return action from spiral controller with heuristic modifications."""
        if wait_for_user:
            self.calls = 0
        if self.calls == 0:
            self.center = observation[6:12]
            self.phi = np.pi * 10
        self.calls += 1
        manual_action = super().get_action(observation=observation, wait_for_user=wait_for_user)
        if not self.manual_action_bool:
            action = self.spiral_controller(observation=observation)
            self.phi += min(0.0006 / (self.a * self.phi + 1e-7), np.pi / 20)
        else:
            action = np.zeros(self.config.action_dim)
            self.center = observation[6:12]
            self.phi = np.pi * 10
        return action


class SupportedManualGenerator(ManualDemonstrationGenerator):
    """
    Demonstration generator that supports manual intervention with additional rotation.
    """
    def __init__(
        self,
        env: gym.Env,
        config: Config,
        key_action_mapping: dict[str, np.ndarray],
        wait_for_input: bool,
        max_rot: float,
    ):
        super().__init__(env=env, config=config, key_action_mapping=key_action_mapping, wait_for_input=wait_for_input)
        self.max_rot = max_rot * np.pi / 180
        self.calls = 0
        self.init_rot = 0
        self.dir = 1
        self.max_target_rot = 20
        self.target_rot = 0
        self.manual = True

    def z_rot_controller(self, observation: np.ndarray) -> np.ndarray:
        """Compute action modifications based on z-axis rotation."""
        action = np.zeros(self.config.action_dim)
        if observation[8] > 7.8e-02:
            action[0] = 0
            action[1] = 0
            action[2] = -0.5
        elif observation[8] < 7.2e-02:
            action[3] = 5 * (observation[11] - observation[5])
        elif observation[8] > 7.2e-02:
            action[0] = 2 * self.dir
            action[1] = 2 * self.dir
            action[2] = (-1 + 2 * (observation[-4] < -15)) * abs(-15 - observation[-4]) / 40 + 0.5 * (abs(observation[-1]) > 1)
            action[3] = 10 * self.dir
            self.target_rot += action[3]
        self.dir = -self.dir
        return action

    def get_action(self, observation: np.ndarray, wait_for_user: bool) -> np.ndarray:
        """Return action combining manual and heuristic z-rotation controller outputs."""
        if wait_for_user:
            self.calls = 0
        if self.calls == 0:
            self.init_rot = observation[11]
        self.calls += 1
        manual_action = super().get_action(observation=observation, wait_for_user=wait_for_user)
        heuristic_action = self.z_rot_controller(observation=observation)
        action = np.zeros(self.config.action_dim)
        for r in range(self.config.action_dim):
            action[r] = manual_action[r] + heuristic_action[r] * (abs(manual_action[r]) < 1e-3)
        return action


class PIController:
    """
    A simple PI controller.
    """
    def __init__(self, kp: float, ki: float):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.integral = 0.0  # Integral term
        self.previous_error = 0.0  # Previous error

    def update(self, setpoint: Any, measured_value: Any, dt: float = 1) -> np.ndarray:
        """
        Update the PI controller and return the control output.
        
        Parameters:
            setpoint: Desired value.
            measured_value: Measured value.
            dt: Time delta.
        
        Returns:
            Control output as a numpy array.
        """
        error = np.array(setpoint) - np.array(measured_value)
        P = self.kp * error
        self.integral += error * dt
        I = self.ki * self.integral
        output = P + I
        self.previous_error = error
        return output


if __name__ == "__main__":
    # Test loading in a policy and sampling episodes if needed
    pass
