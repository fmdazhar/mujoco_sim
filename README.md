# Intro:

This package provide a simple UR5e arm and Robotiq Hand-e Gripper simulator written in Mujoco.
It includes a state-based Peg-in-Hole task environment, custom Gymnasium wrappers, and demonstration generators to facilitate manual and heuristic data collection for reinforcement learning tasks.

# Installation:

- cd into `mujoco_sim`.
- run `pip install -e .` to install this package.
- run `pip install -r requirements.txt` to install sim dependencies.

# Overview

- **Environment Wrappers:**  
  Customize observations and actions for Mujoco environments (e.g. UR5e Peg-In-Hole). Wrappers include:
  - Stacking and filtering of observations.
  - Flattening dictionary observations.
  - Enforcing gripper closure and reducing action dimensionality.
  - Enabling expert interventions via devices like SpaceMouse or Keyboard.

- **Demonstration Generation:**  
    Tools to generate demonstration data (episodes with observations, actions, rewards, discounted returns, etc.) using:
    -   Manual Demonstration: Direct user intervention.
    -  Heuristic Demonstration: Automated data collection using controllers (e.g., PI controller and spiral strategy).
    - Supported Manual Generation: Combines manual input with heuristic modifications.

# Configuration Details

## train_config.py
- **General:** `gamma`, `render_mode`, `manual_demonstration_generation`, `heuristic_demonstration_generation`, `wait_for_input`, `env_id`, `max_env_steps`
- **Demo Generation:** `num_demonstration_episodes`, `demonstration_data_paths`, `rew_bootstrapping`
- **Env Variables:** Auto-sets `key_action_mapping`, `repo_id`, `filename`, `obs_dim`, `action_dim`

## config.py (PegEnvConfig)
- **ENV_CONFIG:** `action_scale`, `control_dt`, `physics_dt`, `time_limit`, `seed`, `version`, plus flags for `enable_force_feedback`, `enable_force_visualization`, `enable_slider_controller`
- **DEFAULT_CAM_CONFIG:** Camera settings (`type`, `fixedcamid`, `lookat`, `distance`, `azimuth`, `elevation`)
- **UR5E_CONFIG:** Home/reset positions, Cartesian bounds, port settings, randomization flags, reset tolerance
- **CONTROLLER_CONFIG:** Gains, damping ratios, clipping limits, control method, etc.
- **RENDERING_CONFIG:** `width`, `height`
- **REWARD_CONFIG:** Dense/sparse reward shaping and task completion tolerance


# Usage
**Running the Environment**
- To launch the simulation and visualize the task, run:

```bash
python mujoco_sim/test/test_gym_env_render.py
```

**Generate Demonstrations:**
- To generate demonstration episodes, run:

```bash
python data_generation/generate_demonstrations.py
```
- The demonstration generator will create episodes and save the demonstration data to the paths specified in the configuration file.

# Notes:

- Error due to `egl` when running on a CPU machine:

```bash
export MUJOCO_GL=egl
conda install -c conda-forge libstdcxx-ng
```

