import gymnasium as gym
from mujoco_sim.data_generation.demonstration_generator import (
    ManualDemonstrationGenerator,
    HeuristicDemonstrationGenerator,
    PIController,
    SupportedManualGenerator,
)
from mujoco_sim.config.train_config import Config
import torch

if __name__ == "__main__":
    # Initialize configuration and environment.
    config = Config()
    env = config.create_env()

    # Choose demonstration generator based on configuration.
    if config.manual_demonstration_generation:
        if config.heuristic_demonstration_generation:
            if config.env_id == "ur5ePegInHoleGymEnv_medium-v0":
                controller = PIController(1600, 0)
                demonstration_generator = HeuristicDemonstrationGenerator(
                    env=env,
                    config=config,
                    key_action_mapping=config.key_action_mapping,
                    wait_for_input=True,
                    spiral_constant=0.0001,
                    Controller=controller,
                )
            elif config.env_id in [
                "ur5ePegInHoleGymEnv_hard-v0",
                "ur5ePegInHoleGymEnv_hard-v1",
                "ur5ePegInHoleGymEnv_very_hard-v1",
            ]:
                demonstration_generator = SupportedManualGenerator(
                    env=env,
                    config=config,
                    key_action_mapping=config.key_action_mapping,
                    wait_for_input=True,
                    max_rot=5,
                )
        else:
            demonstration_generator = ManualDemonstrationGenerator(
                env=env,
                config=config,
                key_action_mapping=config.key_action_mapping,
                wait_for_input=True,
            )
    else:
        repo_id = config.repo_id
        filename = config.filename
        if config.heuristic_demonstration_generation:
            controller = PIController(1600, 0)
            demonstration_generator = HeuristicDemonstrationGenerator(
                env=env,
                config=config,
                key_action_mapping=config.key_action_mapping,
                wait_for_input=False,
                spiral_constant=0.0001,
                Controller=controller,
            )
        else:
            raise ValueError(
                "No demonstration generation method defined. Either enable "
                "heuristic_demonstration_generation or add another method."
            )

    # Generate and save demonstration data.
    demonstrations = demonstration_generator.generate_demonstrations(
        num_episodes=max(config.num_demonstration_episodes)
    )
    demonstration_generator.save_demonstrations(
        demonstrations, path=config.demonstration_data_paths
    )
