# """
# Driver class for Keyboard controller.
# """

import numpy as np
import pygame
from .device import Device
from mujoco_sim.utils.transform_utils import rotation_matrix


class Keyboard(Device):
    """
    A minimalistic driver class for a Keyboard using pygame, supporting continuous movement.
    Args:
        pos_sensitivity (float): Magnitude of input position command scaling
        rot_sensitivity (float): Magnitude of scale input rotation commands scaling
    """

    def __init__(self, pos_sensitivity=1/1000, rot_sensitivity=1/100):
        # Initialize pygame and create a small hidden window to capture keyboard events
        pygame.init()
        pygame.display.set_mode((200, 200))  # Small visible window
        pygame.display.set_caption("Keyboard Control")  # Set a name to avoid confusion
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
        print_command("spacebar", "toggle gripper (open/close)")
        print_command("w-a-s-d", "move arm horizontally in x-y plane")
        print_command("r-f", "move arm vertically")
        print_command("y-x", "rotate arm about x-axis")
        print_command("t-g", "rotate arm about y-axis")
        print_command("c-v", "rotate arm about z-axis")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
        self.raw_drotation = np.zeros(3)  # roll, pitch, yaw delta values
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

        # Process any pending events and handle continuous movement
        self.update()

        dpos = self.pos - self.last_pos
        self.last_pos = np.array(self.pos)
        raw_drotation = self.raw_drotation - self.last_drotation
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

    def update(self):
        """
        Poll pygame events and update internal state accordingly.
        This should be called every frame.
        """
        # First handle discrete actions from events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # Discrete actions
                if event.key == pygame.K_SPACE:
                    self.grasp = not self.grasp  # toggle gripper
                elif event.key == pygame.K_q:
                    self._reset_state = 1
                    self._reset_internal_state()

        # Now handle continuous movement for keys that should cause continuous action
        keys = pygame.key.get_pressed()

        # Continuous position movement
        # (x, y plane movement)
        if keys[pygame.K_w]:
            self.pos[0] -= self._pos_step * self.pos_sensitivity
        if keys[pygame.K_s]:
            self.pos[0] += self._pos_step * self.pos_sensitivity
        if keys[pygame.K_a]:
            self.pos[1] -= self._pos_step * self.pos_sensitivity
        if keys[pygame.K_d]:
            self.pos[1] += self._pos_step * self.pos_sensitivity

        # (z-axis movement)
        if keys[pygame.K_r]:
            self.pos[2] += self._pos_step * self.pos_sensitivity
        if keys[pygame.K_f]:
            self.pos[2] -= self._pos_step * self.pos_sensitivity

        # Continuous orientation movement
        # Rotate about x-axis
        if keys[pygame.K_y]:
            drot = rotation_matrix(
                angle=0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[1] -= 0.1 * self.rot_sensitivity

        if keys[pygame.K_x]:
            drot = rotation_matrix(
                angle=-0.1 * self.rot_sensitivity, direction=[1.0, 0.0, 0.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[1] += 0.1 * self.rot_sensitivity

        # Rotate about y-axis
        if keys[pygame.K_t]:
            drot = rotation_matrix(
                angle=0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[0] += 0.1 * self.rot_sensitivity

        if keys[pygame.K_g]:
            drot = rotation_matrix(
                angle=-0.1 * self.rot_sensitivity, direction=[0.0, 1.0, 0.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[0] -= 0.1 * self.rot_sensitivity

        # Rotate about z-axis
        if keys[pygame.K_c]:
            drot = rotation_matrix(
                angle=0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[2] += 0.1 * self.rot_sensitivity

        if keys[pygame.K_v]:
            drot = rotation_matrix(
                angle=-0.1 * self.rot_sensitivity, direction=[0.0, 0.0, 1.0]
            )[:3, :3]
            self.rotation = self.rotation.dot(drot)
            self.raw_drotation[2] -= 0.1 * self.rot_sensitivity
