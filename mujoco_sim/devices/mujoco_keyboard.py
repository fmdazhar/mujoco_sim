# external_keyboard.py

import numpy as np
import time
from mujoco_sim.utils.transform_utils import rotation_matrix
import mujoco
import glfw

class MujocoKeyboard:
    """
    External Keyboard Handler using MuJoCo's key callbacks.
    This class does not use pynput and instead relies on MuJoCo's GLFW window.

    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1/1000, rot_sensitivity=1/100):
        self._display_controls()

        self._reset_internal_state()

        self._reset_state = 0
        self._enabled = False
        self._pos_step = 1/150

        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    @staticmethod
    def _display_controls():
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (10 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Keys", "Command")
        print_command("q", "reset simulation")
        print_command("right shift", "toggle gripper (open/close)")
        print_command("up/down", "move arm along x-axis")
        print_command("left/right", "move arm along y-axis")
        print_command("l/p", "move arm along z-axis")
        print_command("n/m", "rotate arm about x-axis")
        print_command("j/k", "rotate arm about y-axis")
        print_command("i/o", "rotate arm about z-axis")
        print("")


    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], 
                                  [0.0, 1.0, 0.0], 
                                  [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # immediate roll, pitch, yaw delta values from keyboard hits
        self.last_drotation = np.zeros(3)
        self.pos = np.zeros(3)  # (x, y, z)
        self.last_pos = np.zeros(3)
        self.grasp = False

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """
        Grabs the current state of the keyboard.
        Returns:
            dict: A dictionary containing dpos, rotation, raw_drotation, grasp, and reset
        """

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = (
            self.raw_drotation - self.last_drotation
        )  # create local variable to return, then reset internal drotation
        self.last_drotation = np.array(self.raw_drotation)

        # Capture the reset state and then reset it to 0
        reset_state = self._reset_state
        self._reset_state = 0  # Reset the reset state after reading it

        return dict(
            dpos=dpos,
            rotation=self.rotation,
            raw_drotation=raw_drotation,
            grasp=int(self.grasp),
            reset=reset_state,
        )

    def external_key_callback(self, window, key, scancode, action, mods):
        """
        External key callback to handle key presses and releases.

        Args:
            window: The GLFW window
            key (int): Key code
            scancode (int): Scancode
            action (int): Action (press, release, etc.)
            mods (int): Modifier keys
        """
        # if action != glfw.PRESS and action != glfw.RELEASE:
        #     return

        if action == glfw.REPEAT:
            self._handle_key_repeat(key)
        elif action == glfw.RELEASE:
            self._handle_key_release(key)
        elif action == glfw.PRESS:
            self._handle_key_press(key)

    def _handle_key_repeat(self, key):
        """
        Handle key press events.

        Args:
            key (int): Key code
        """
        try:
            if key == glfw.KEY_UP:
                self.pos[0] -= self._pos_step * self.pos_sensitivity  # dec x
            elif key == glfw.KEY_DOWN:
                self.pos[0] += self._pos_step * self.pos_sensitivity  # inc x
            elif key == glfw.KEY_LEFT:
                self.pos[1] -= self._pos_step * self.pos_sensitivity  # dec y
            elif key == glfw.KEY_RIGHT:
                self.pos[1] += self._pos_step * self.pos_sensitivity  # inc y
            elif key == glfw.KEY_L:
                self.pos[2] -= self._pos_step * self.pos_sensitivity  # dec z
            elif key == glfw.KEY_P:
                self.pos[2] += self._pos_step * self.pos_sensitivity  # inc z

            # controls for moving orientation
            elif key == glfw.KEY_N:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] -= 0.1 * self.rot_sensitivity
            elif key == glfw.KEY_M:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates x
                self.raw_drotation[1] += 0.1 * self.rot_sensitivity
            elif key == glfw.KEY_J:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] += 0.1 * self.rot_sensitivity
            elif key == glfw.KEY_K:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates y
                self.raw_drotation[0] -= 0.1 * self.rot_sensitivity
            elif key == glfw.KEY_I:
                drot = rotation_matrix(angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] += 0.1 * self.rot_sensitivity
            elif key == glfw.KEY_O:
                drot = rotation_matrix(angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0])[:3, :3]
                self.rotation = self.rotation.dot(drot)  # rotates z
                self.raw_drotation[2] -= 0.1 * self.rot_sensitivity

        except Exception as e:
            print(f"Error handling key press: {e}")

    def _handle_key_release(self, key):
        """
        Handle key release events. Currently no actions on release.

        Args:
            key (int): Key code
        """
        pass  # Implement if needed

    def _handle_key_press(self, key):
        """
        Handle key press events. Currently no actions on release.

        Args:
            key (int): Key code
        """
        try:
            if key == glfw.KEY_RIGHT_SHIFT:
                self.grasp = not self.grasp  # toggle gripper

            elif key == glfw.KEY_Q:
                self._reset_state = 1
                self._reset_internal_state()
        
        except Exception as e:
            print(f"Error handling key press: {e}")

