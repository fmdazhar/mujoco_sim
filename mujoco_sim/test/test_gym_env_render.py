import time
import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
import mujoco_sim
import imageio

# Flags
SAVE_VIDEO = True      # Set to False to skip video saving

env = gymnasium.make("ur5ePegInHoleGymEnv_easy-v0", render_mode="human")

action_spec = env.action_space
print(f"Action space: {action_spec}")

observation_spec = env.observation_space
print(f"Observation space: {observation_spec}")

def sample():
    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    return a.astype(action_spec.dtype)

obs, info = env.reset()
frames = [] if SAVE_VIDEO else None  # Initialize only if collecting frames

for i in range(500):
    a = sample()
    obs, rew, done, truncated, info = env.step(a)
    
    if SAVE_VIDEO:
        images = env.render()
        combined_frame = np.concatenate(images, axis=0)  # Stack images vertically
        frames.append(combined_frame)

    if done or truncated:
        obs, info = env.reset()

# Save video only if enabled
if SAVE_VIDEO:
    imageio.mimsave("render_test.mp4", frames, fps=20)
    print("Video saved as render_test.mp4")

# Properly close the environment
env.close()
