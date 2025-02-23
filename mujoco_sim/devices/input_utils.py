"""
Utility functions for grabbing user inputs
"""

import numpy as np
from mujoco_sim.devices import *

from mujoco_sim.config import PegEnvConfig


def input2action(device):
    """
    Converts an input from an active device into a valid action sequence that can be fed into an env.step() call

    If a reset is triggered from the device, immediately returns None. Else, returns the appropriate action

    Args:
        device (Device): A device from which user inputs can be converted into actions. Can be either a Spacemouse or
            Keyboard device class

        robot (Robot): Which robot we're controlling

        active_arm (str): Only applicable for multi-armed setups (e.g.: multi-arm environments or bimanual robots).
            Allows inputs to be converted correctly if the control type (e.g.: IK) is dependent on arm choice.
            Choices are {right, left}

        env_configuration (str or None): Only applicable for multi-armed environments. Allows inputs to be converted
            correctly if the control type (e.g.: IK) is dependent on the environment setup. Options are:
            {bimanual, single-arm-parallel, single-arm-opposed}

    Returns:
        2-tuple:

            - (None or np.array): Action interpreted from @device including any gripper action(s). None if we get a
                reset signal from the device
            - (None or int): 1 if desired close, -1 if desired open gripper state. None if get a reset signal from the
                device

    """
    state = device.get_controller_state()
    # Note: Devices output rotation with x and z flipped to account for robots starting with gripper facing down
    #       Also note that the outputted rotation is an absolute rotation, while outputted dpos is delta pos
    #       Raw delta rotations from neutral user input is captured in raw_drotation (roll, pitch, yaw)
    dpos, rotation, raw_drotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["raw_drotation"],
        state["grasp"],
        state["reset"],
    )

    # If we're resetting, immediately return None
    if reset:
        return None

    # First process the raw drotation
    drotation = raw_drotation[[1, 0, 2]]

    # Flip z
    drotation[2] = -drotation[2]

    # Scale rotation for teleoperation (tuned for OSC) -- gains tuned for each device
    drotation = drotation * 1.5 if isinstance(device, (MujocoKeyboard, Keyboard)) else drotation * 50
    dpos = dpos * 75 if isinstance(device, (MujocoKeyboard, Keyboard)) else dpos * 125
    # map 0 to -1 (open) and map 1 to 1 (closed)
    grasp = 1 if grasp else -1

    # adjust action scale to match with config
    config = PegEnvConfig()
    _action_scale = config.ENV_CONFIG["action_scale"]
    dpos = dpos * 1 / _action_scale[0]
    drotation = drotation * 1 / _action_scale[1]
    grasp = grasp * 1 / _action_scale[2]

    # Create action based on action space of individual robot
    action = np.concatenate([dpos, drotation, [grasp]])

    # Return the action and grasp
    return action
