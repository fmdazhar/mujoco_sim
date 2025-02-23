import os
import time
import logging
from pathlib import Path
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from wonderwords import RandomWord


class Config:
    def __init__(self) -> None:
        """
        Configuration for demonstration generation and environment creation.
        Only parameters used by demonstration generation are kept.
        """
        # Set up basic paths.
        this_file_dir = Path(__file__).resolve().parent  # .../mujoco_sim/config
        self.base_path = this_file_dir.parent            # .../mujoco_sim

        # General settings.
        self.gamma = 0.99  # Discount factor for future rewards.
        self.render_mode = "human"
        self.manual_demonstration_generation = True
        self.heuristic_demonstration_generation = False
        self.wait_for_input = True
        self.env_id = "ur5ePegInHoleGymEnv_medium-v1"  # Example environment ID.
        self.max_env_steps = 1000

        # Demonstration generation settings.
        self.num_demonstration_episodes = [1, 4, 16, 32]
        self.demonstration_data_paths = {}
        for num_demos in self.num_demonstration_episodes:
            demonstration_data_path = os.path.join(
                self.base_path,
                "demonstration_data",
                self.env_id,
                f"demo_{num_demos}_episodes_vx.pt",
            )
            os.makedirs(os.path.dirname(demonstration_data_path), exist_ok=True)
            self.demonstration_data_paths[str(num_demos)] = demonstration_data_path

        # Bootstrapping key for discounted return computation.
        self.rew_bootstrapping = "r_min"  # Options: "r_min", "r_max", "r_mean", "r_T"

        # Environment is created lazily.
        self._env = None

        # Set up basic logging.
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.getLogger("PIL").setLevel(logging.WARNING)

    def create_env(self):
        """Lazily create and return the gym environment."""
        if self._env is None:
            self._env = gym.make(self.env_id, render_mode=self.render_mode, max_episode_steps=500)
            self.load_env_config()
            print("Environment created")
        return self._env

    @property
    def env(self):
        return self._env

    def load_env_config(self):
        """
        Loads environment configuration and sets key variables:
          - key_action_mapping: mapping for manual actions.
          - repo_id, filename: identifiers (placeholders here).
          - obs_dim, action_dim: dimensions of observation and action spaces.
        """
        obs_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]

        config_values = {
            "key_action_mapping": {"default": np.zeros((7,), dtype=np.float32)},
            "repo_id": "No agent available",
            "filename": "No agent available",
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
        (
            self.key_action_mapping,
            self.repo_id,
            self.filename,
            self.obs_dim,
            self.action_dim,
        ) = (
            config_values["key_action_mapping"],
            config_values["repo_id"],
            config_values["filename"],
            config_values["obs_dim"],
            config_values["action_dim"],
        )

    def get_random_adjective_noun(self) -> str:
        """
        Generate a random adjective-noun pair for naming experiments.
        """
        r = RandomWord()
        adjective = r.word(include_parts_of_speech=["adjective"])
        noun = r.word(include_parts_of_speech=["noun"])
        return f"{adjective}_{noun}"


if __name__ == "__main__":
    config = Config()
    print("Config created")
