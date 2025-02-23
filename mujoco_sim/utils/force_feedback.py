import tkinter as tk
import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ForceFeedback:
    def __init__(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Polar Rectangles")

        # Create a canvas widget
        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        # Center of the canvas
        center_x, center_y = 200, 200

        # Radius for the rectangles
        radius = 100

        # List to store rectangle IDs
        self.force_rectangles = []
        self.torque_rectangles = []

        # Labels for the rectangles
        labels = ["D", "F", "S", "A", "R", "W"]
        angles = [20, 90, 160, 200, 270, 340]
        rect_width, rect_height = 150, 25

        # Add vertical rectangles at the outer ends of "W", "A", "S", "D"
        vertical_rect_height = 50
        vertical_rect_width = 20

        # Create 6 force-rectangles
        for i in range(6):
            angle = math.radians(angles[i])
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            if i in [0, 2, 3, 5]:
                # Calculate the outer endpoint
                outer_x = x + rect_width / 2 * math.cos(angle)
                outer_y = y + rect_width / 2 * math.sin(angle)

                # Adjust position for the vertical rectangle
                vx1 = outer_x - vertical_rect_width / 2
                vy1 = outer_y
                vx2 = outer_x + vertical_rect_width / 2
                vy2 = outer_y + vertical_rect_height

                # Create the vertical rectangle
                rect = self.canvas.create_rectangle(
                    vx1, vy1, vx2, vy2, fill="white", outline="black"
                )
                self.torque_rectangles.append(rect)

            # Calculate the orientation of the rectangle
            x1 = (
                x - rect_width / 2 * math.cos(angle) + rect_height / 2 * math.sin(angle)
            )
            y1 = (
                y - rect_width / 2 * math.sin(angle) - rect_height / 2 * math.cos(angle)
            )
            x2 = (
                x + rect_width / 2 * math.cos(angle) + rect_height / 2 * math.sin(angle)
            )
            y2 = (
                y + rect_width / 2 * math.sin(angle) - rect_height / 2 * math.cos(angle)
            )

            rect = self.canvas.create_polygon(
                x1,
                y1,
                x2,
                y2,
                x2 - rect_height * math.sin(angle),
                y2 + rect_height * math.cos(angle),
                x1 - rect_height * math.sin(angle),
                y1 + rect_height * math.cos(angle),
                fill="white",
                outline="black",
            )
            self.force_rectangles.append(rect)

            # Add label to the rectangle
            self.canvas.create_text(x, y, text=labels[i], font=("Arial", 12, "bold"))

        # Add the second window for goal and current positions
        self.create_position_window()
        # Define goal positions
        self.goal_x, self.goal_y = 200, 200  # Example goal position
        self.xy_history = []
        # Draw cross for goal position
        # self.goal_cross = self.draw_cross(self.goal_x, self.goal_y, color="red")
        self.rand_x = np.random.rand()
        self.rand_y = np.random.rand()
        self.current_cross = self.draw_cross(0, 0, color="blue")

        self.z_poses = []
        # Data for the line chart
        x_data = list(range(min(len(self.z_poses), 10)))
        y_data = self.z_poses[-min(len(self.z_poses), 10) :]

        # Create a Matplotlib figure
        fig = Figure(figsize=(3, 2), dpi=100)
        self.ax = fig.add_subplot(111)
        (self.line,) = self.ax.plot(x_data, y_data)
        self.ax.set_ylim(0.073, 0.074)

        # Embed the Matplotlib figure into the Tkinter window
        self.canvas_mptl = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = self.canvas_mptl.get_tk_widget()
        canvas_widget.pack()

    # Function to convert force to red intensity with gradient from white to red
    def value_to_red_intensity(self, value, max_value):
        # Assuming force ranges from 0 to 100 for simplicity
        # intensity = int(min(max(force, 0), 100) * 2.55)
        # return f'#{intensity:02x}0000'
        red_value = 255 - int(min(max(value, 0), max_value) * 255 / max_value)
        color = f"#ee{red_value:02x}{red_value:02x}"
        return color

    # Function to update rectangles based on forces
    def update_force_rectangles(self, forces, max_force):
        # Initialize forces in 6 directions
        forces_6_directions = [0] * 6
        forces_6_directions[2] = max(forces[0], 0)
        forces_6_directions[5] = max(-forces[0], 0)
        forces_6_directions[0] = max(forces[1], 0)
        forces_6_directions[3] = max(-forces[1], 0)
        forces_6_directions[4] = max(forces[2], 0)
        forces_6_directions[1] = max(-forces[2], 0)
        for i, force in enumerate(forces_6_directions):
            color = self.value_to_red_intensity(force, max_force)
            self.canvas.itemconfig(
                self.force_rectangles[i], fill=color, outline="black"
            )

    # Function to update rectangles based on torques
    def update_torque_rectangles(self, torques, max_torques):
        # Initialize forces in 6 directions
        torques_4_directions = [0] * 4
        torques_4_directions[0] = max(torques[0], 0)
        torques_4_directions[2] = max(-torques[0], 0)
        torques_4_directions[1] = max(-torques[1], 0)
        torques_4_directions[3] = max(torques[1], 0)
        for i, torque in enumerate(torques_4_directions):
            color = self.value_to_red_intensity(torque, max_torques)
            self.canvas.itemconfig(
                self.torque_rectangles[i], fill=color, outline="black"
            )

    def create_position_window(self):
        # Create a new Toplevel window
        self.position_window = tk.Toplevel(self.root)
        self.position_window.title("Position Window")
        self.position_window.geometry("400x400")

        # Create a canvas in the new window
        self.position_canvas = tk.Canvas(self.position_window, width=400, height=400)
        self.position_canvas.pack()

    def draw_cross(self, x, y, color):
        # Length of cross arms
        arm_length = 20
        # Draw horizontal line
        h_line = self.position_canvas.create_line(
            x - arm_length, y, x + arm_length, y, fill=color, width=1
        )
        # Draw vertical line
        v_line = self.position_canvas.create_line(
            x, y - arm_length, x, y + arm_length, fill=color, width=1
        )

        return {"h_line": h_line, "v_line": v_line}
    
    def draw_path(self):
        c_max = len(self.xy_history)
        c = 0
        if len(self.xy_history) > 1:
            xy_0 = self.xy_history[0]
            for xy in self.xy_history[1:]:
                blue_value = min(2*(c_max-c), 217)
                color = f"#{blue_value:02x}{blue_value:02x}{217:02x}"
                self.position_canvas.create_line(xy_0[0], xy_0[1], xy[0], xy[1], fill=color, width=1)
                xy_0 = xy
                c += 1

    def update_xy_pos(self, x1, y1):
        y = (x1 - 0.403) / 0.02 * self.goal_x + self.goal_x     # + (self.rand_x*50-25)
        x = (y1 - 0.001) / 0.02 * self.goal_y + self.goal_y     # + (self.rand_y*50-25)
        self.xy_history.append([x, y])
        # Length of cross arms
        arm_length = 20
        # Update horizontal line
        self.position_canvas.coords(
            self.current_cross["h_line"], x - arm_length, y, x + arm_length, y
        )
        # Update vertical line
        self.position_canvas.coords(
            self.current_cross["v_line"], x, y - arm_length, x, y + arm_length
        )
        self.draw_path()

    def update_z_pos(self, z_pos):
        self.z_poses.append(z_pos)
        # Update the line data
        x_data = list(range(min(len(self.z_poses), 10)))
        y_data = self.z_poses[-min(len(self.z_poses), 10) :]
        self.line.set_xdata(x_data)
        self.line.set_ydata(y_data)

        # Rescale the axes if necessary
        self.ax.relim()
        self.ax.set_ylim(0.073, 0.074)
        self.ax.autoscale_view()

        # Redraw the canvas
        self.canvas_mptl.draw()
