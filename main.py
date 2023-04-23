#!/usr/bin/env python 

__author__ = "OKUNOWO, Similoluwa Adetoyosi (EEG/2016/095)"
__email__ = "rexsimiloluwa@gmail.com"

import sys
import ast
import warnings 
from typing import Optional, List, Tuple, NamedTuple

import control
import matplotlib

matplotlib.use("QtAgg")
import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QPushButton,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QGridLayout,
    QFormLayout,
    QGroupBox,
    QSplitter,
    QApplication,
    QStyleFactory,
    QMainWindow,
    QSlider,
    QTextEdit,
    QSizePolicy
)
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from scipy.integrate import odeint

warnings.filterwarnings("ignore")

DEFAULT_TF = ([3], [1, 2, 3, 1])

# PID controller gains obtained after tuning in MATLAB
DEFAULT_KP_VALUE = 0.6752
DEFAULT_KI_VALUE = 0.3233
DEFAULT_KD_VALUE = 0.3470

DEFAULT_SIMULATION_TIME = 10  # seconds

STEP_INFO_MAP = {
    "RiseTime": ("Rise Time", "seconds"),
    "SettlingTime": ("SettlingTime", "seconds"),
    "SettlingMin": ("Settling Min", ""),
    "SettlingMax": ("Settling Max", ""),
    "Overshoot": ("Overshoot", ""),
    "Undershoot": ("Undershoot", ""),
    "Peak": ("Peak", ""),
    "PeakTime": ("Peak Time", "seconds"),
    "SteadyStateValue": ("Steady State Value", ""),
}

plant_tf = DEFAULT_TF


class TfPoly(NamedTuple):
    num: List[float]
    den: List[float]


def compute_step_response(
    plant_tf: TfPoly,
    controller_tf: TfPoly,
    simulation_time: Optional[int] = DEFAULT_SIMULATION_TIME,
):
    """Compute the step response for a unity feedback system.

    Args:
        plant_tf (tuple): The plant transfer function
        controller_tf (tuple): The controller transfer function
        simulation_time (int): Simulation time in seconds, defaults to 10s

    Returns:
        tuple: A tuple containing
        - t_out (List[float]): The time
        - y_out (List[float]): The response
        - step_info: Contains performance metrics for the step response i.e. rise time
    """
    # Convert to 'control' form
    plant_tf = control.TransferFunction(plant_tf[0], plant_tf[1])
    controller_tf = control.TransferFunction(controller_tf[0], controller_tf[1])

    # Combine the plant and the controller
    ol_system = control.series(controller_tf, plant_tf)
    cl_system = control.feedback(ol_system, control.TransferFunction(1, 1))

    # Compute the step response of the closed-loop system
    t = np.linspace(0, simulation_time, 1000)
    t_out, y_out = control.step_response(cl_system, t)

    # Compute the step info (performance metrics)
    try:
        step_info = control.step_info(cl_system)
    except:
        step_info = None

    return t_out, y_out, step_info

def autotune(sys: TfPoly) -> Tuple[float, float, float]:
    """Obtain the tuned PID gains.

    Args: 
        sys (tuple): A tuple containing the system's transfer function numerator and denominator polynomials

    Returns:
        tuple: A tuple containing 
        - Kp (float): Value for the proportional gain
        - Ki (float): Value for the integral gain
        - Kd (float): Value for the derivative gain
    """
    return 0, 0, 0

class MplCanvas(FigureCanvas):
    """Matplotlib Figure Canvas for Realtime plotting."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class ChangeTfDialog(QDialog):
    """Dialog Widget for changing the plant transfer function."""

    tf_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        global plant_tf

        self.setFixedWidth(300)
        self.setWindowTitle("Change Transfer Function")

        # Create a form layout for the dialog
        layout = QFormLayout(self)

        # Add the form fields form the tf numerator and denominator
        self.num = QLineEdit(self)
        self.den = QLineEdit(self)

        self.num.setText(str(plant_tf[0]))
        self.den.setText(str(plant_tf[1]))

        layout.addRow("Numerator: ", self.num)
        layout.addRow("Denominator: ", self.den)

        # Add OK and Cancel buttons to the layout
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            Qt.Orientation.Horizontal,
            self,
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow(buttons)

    def values(self):
        return ast.literal_eval(self.num.text()), ast.literal_eval(self.den.text())

    def accept(self):
        global plant_tf
        self.tf_changed.emit(True)

        plant_tf = self.values()

        self.done(1)
        self.close()

    def reject(self):
        self.done(1)
        self.close()


class SimTimeGroup(QGroupBox):
    """Simulation Time Group Widget.

    Used to change the simulation time.
    """

    sim_time_changed = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Simulation Time")
        self.value = float(DEFAULT_SIMULATION_TIME)

        layout = QGridLayout()

        self.sim_time_field = QLineEdit()
        self.sim_time_field.setReadOnly(True)
        self.sim_time_field.setText(str(self.value))

        # Increment and Decrement buttons
        self.increment_btn = QPushButton("+")
        self.decrement_btn = QPushButton("-")

        self.increment_btn.clicked.connect(lambda: self.update_value(5))
        self.decrement_btn.clicked.connect(lambda: self.update_value(-5))

        layout.addWidget(self.decrement_btn, 0, 0, 1, 1)
        layout.addWidget(self.sim_time_field, 0, 1)
        layout.addWidget(self.increment_btn, 0, 2, 1, 1)

        self.setStyleSheet(
            """
            QPushButton{
                background: orange;
                color: #FFF;
                font-size: 18px;
                font-weight: bold;
                padding: 2px 10px;
                text-align: center;
            }

            QPushButton:hover{
                background: blue;
            }

            QLineEdit{
                padding: 4px;
                text-align: center;
            }
        """
        )

        self.setLayout(layout)

    def update_value(self, count: int) -> None:
        """Update the value of the simulation time.

        Args:
            count (int): The increment value
        """
        new_value = self.value + count

        # Ensure that it does not go below 5s
        if new_value >= 5:
            self.value = new_value
            self.sim_time_field.setText(str(self.value))

            # Emit the updated value
            self.sim_time_changed.emit(int(self.value))


class TransferFunctionGroup(QGroupBox):
    """Transfer Function Group Widget.

    Used to render the plant transfer function.

    Args:
        num (list): A list containing the coefficients of the numerator polynomial of the tf
        den (list): A list containing the coefficients of the denominator polynomial of the tf
    """

    tf_update = QtCore.pyqtSignal(bool)

    def __init__(self, num: List[float], den: List[float], parent=None):
        super().__init__(parent)
        self.setTitle("Plant Transfer Function")
        self.num = num
        self.den = den
        self.dialog = ChangeTfDialog(self)

        vbox_layout = QVBoxLayout()

        # Create the transfer function
        tf = control.TransferFunction(num, den)

        # Label for displaying the transfer function
        self.tf_label = QLabel(str(tf))
        self.tf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tf_label.setStyleSheet(
            "background: #FFF; padding: 10px; text-align: center;"
        )

        # Button for changing the transfer function
        change_tf_button = QPushButton("Change")
        change_tf_button.clicked.connect(lambda: self.open_dialog())

        vbox_layout.addWidget(self.tf_label)
        vbox_layout.addWidget(change_tf_button)
        vbox_layout.setSpacing(18)

        self.setStyleSheet(
            """
            QPushButton{
                background: orange;
                color: #FFF;
            }

            QPushButton:hover{
                background: blue;
            }
        """
        )
        self.setLayout(vbox_layout)

    def open_dialog(self) -> None:
        """"""
        self.dialog.tf_changed.connect(
            lambda: self.update_tf_label(self.dialog.values())
        )
        self.dialog.exec()

    def update_tf_label(self, values: Tuple[List[float], List[float]]) -> None:
        self.num, self.den = values
        tf = control.TransferFunction(self.num, self.den)
        self.tf_label.setText(str(tf))
        self.tf_update.emit(True)

    def values(self):
        return self.num, self.den


class SliderGroup(QGroupBox):
    """Slider Group Widget.

    Used to render the sliders for controlling the PID gains i.e. Kp, Ki, and Kd

    Args:
        title (str): Title of the slider group
        min_slider_value (Optional[int]): Minimum slider value, defaults to -10
        max_slider_value (Optional[int]): Maximum slider value, defaults to 10
        default_slider_value (Optional[int]): Default slider value, defaults to 0
        resolution (Optional[float]): Resolution or step size of the slider
    """

    def __init__(
        self,
        title: str,
        min_slider_value: Optional[int] = -10,
        max_slider_value: Optional[int] = 10,
        default_slider_value: Optional[int] = 0,
        resolution: Optional[float] = 0.01,
        parent=None,
    ):
        super().__init__(parent)
        self.setTitle(title)
        self.resolution = resolution

        vbox_layout = QVBoxLayout()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_slider_value / resolution)
        self.slider.setMaximum(max_slider_value / resolution)
        self.slider.setValue(default_slider_value / resolution)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        vbox_layout.addWidget(self.slider)
        self.setLayout(vbox_layout)

    def slider_value(self) -> float:
        """Returns the slider value."""
        # Transform the slider value using the resolution
        return float(self.slider.value()) * self.resolution


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrudePID")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon("icon.png"))

        # Create the sidebar panel
        self.sidebar = QFrame()
        self.sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        self.sidebar.setFixedWidth(250)

        # Create the sidebar layout
        self.sidebar_layout = QVBoxLayout()

        # Create the child widgets for the sidebar layout
        # Transfer function
        self.plant_tf = TransferFunctionGroup(*DEFAULT_TF)

        # Sliders
        self.Kp_slider = SliderGroup(
            "Kp (Proportional Gain)", default_slider_value=DEFAULT_KP_VALUE
        )
        self.Ki_slider = SliderGroup(
            "Ki (Integral Gain)", default_slider_value=DEFAULT_KI_VALUE
        )
        self.Kd_slider = SliderGroup(
            "Kd (Derivative Gain)", default_slider_value=DEFAULT_KD_VALUE
        )

        self.sim_time_group = SimTimeGroup()

        self.autotune_button = QPushButton()
        self.autotune_button.setText("Tune")
        self.autotune_button.setStyleSheet("""
            background: crimson;
            color: #FFF;
            font-size: 14px;
            font-weight: 500;
        """)

        # Add the widgets to the sidebar layout
        self.sidebar_layout.addWidget(self.plant_tf)
        self.sidebar_layout.addWidget(self.Kp_slider)
        self.sidebar_layout.addWidget(self.Ki_slider)
        self.sidebar_layout.addWidget(self.Kd_slider)
        self.sidebar_layout.addWidget(self.sim_time_group)
        self.sidebar_layout.addWidget(self.autotune_button)

        self.sidebar_layout.addStretch(5)
        self.sidebar_layout.setSpacing(20)
        self.sidebar.setLayout(self.sidebar_layout)

        # Create the main view
        self.main = QFrame()
        self.main.setFrameShape(QFrame.Shape.StyledPanel)

        # Create the layout for the main view
        self.main_layout = QVBoxLayout()
        self.main.setLayout(self.main_layout)

        # Create the figure and axes for the plot
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        # Create the canvas toolbar
        toolbar = NavigationToolbar(self.canvas, self)

        self.step_info = QTextEdit()
        self.step_info.setReadOnly(True)
        self.step_info.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred
        )

        # Add the toolbar and canvas to the main layout
        self.main_layout.addWidget(toolbar)
        self.main_layout.addWidget(self.canvas)
        self.main_layout.addWidget(self.step_info)
        self.main_layout.addWidget(QLabel("By OKUNOWO Similoluwa (EEG/2016/095)"))

        self.main_layout.addStretch(5)
        self.main_layout.setSpacing(20)

        # Add the sidebar and main part to the main window
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(self.main)
        self.setCentralWidget(self.splitter)

        # Initialize the step response plot
        self.init_response()

        # Callback: When the sliders are changed
        self.Kp_slider.slider.valueChanged.connect(self.update_response)
        self.Ki_slider.slider.valueChanged.connect(self.update_response)
        self.Kd_slider.slider.valueChanged.connect(self.update_response)

        # Callback: for tuning the PID controller
        self.autotune_button.clicked.connect(self.tune)

        # Callback: When the plant transfer function is changed
        self.plant_tf.tf_update.connect(self.update_response)

        # Callback: When the simulation time is changed
        self.sim_time_group.sim_time_changed.connect(self.update_response)

    def tune(self):
        """Determine the optimal P, I, and D gains for the PID controller.
        
        The gains should satisfy the following performance criteria:
        1. < 20% overshoot 
        2. < 10s settling time 
        3. < 3s rise time 
        """
        plant_sys = self.plant_tf.values()
        Kp_tuned, Ki_tuned, Kd_tuned = autotune(plant_sys)

        # Update the slider values 
        self.Kp_slider.slider.setValue(Kp_tuned)
        self.Ki_slider.slider.setValue(Ki_tuned)
        self.Kd_slider.slider.setValue(Kd_tuned)

    def plot_response(
        self, t_out: List[float], y_out: List[float], Kp: float, Ki: float, Kd: float
    ):
        """Plot the response on the canvas."""
        self.canvas.axes.cla()
        self.canvas.axes.plot(t_out, y_out)
        self.canvas.axes.set_title(
            f"Step Response (Kp = {Kp:.3f}, Ki = {Ki:.3f}, Kd = {Kd:.3f})"
        )
        self.canvas.axes.grid(True)
        self.canvas.draw()

    def render_step_info(self, step_info):
        """Render the step response info."""
        if step_info:
            self.step_info.setText(
                "\n\n".join(
                    list(
                        map(
                            lambda x: f"{STEP_INFO_MAP[x][0]}: {step_info[x]:.4f} {STEP_INFO_MAP[x][1]}",
                            step_info.keys(),
                        )
                    )
                )
            )

    def init_response(self):
        """Initialize the step response plot with the default values."""
        # Get the slider values
        Kp_value = self.Kp_slider.slider_value()
        Ki_value = self.Ki_slider.slider_value()
        Kd_value = self.Kd_slider.slider_value()

        pid_tf = ([Kd_value, Kp_value, Ki_value], [1, 0])
        t_out, y_out, step_info = compute_step_response(DEFAULT_TF, pid_tf)

        # Render the step info
        self.render_step_info(step_info)

        self.plot_response(t_out, y_out, Kp_value, Ki_value, Kd_value)

    def update_response(self):
        """Callback for updating the response."""
        # Get the slider values
        Kp_value = self.Kp_slider.slider_value()
        Ki_value = self.Ki_slider.slider_value()
        Kd_value = self.Kd_slider.slider_value()

        pid_tf = ([Kd_value, Kp_value, Ki_value], [1, 0])

        t_out, y_out, step_info = compute_step_response(
            self.plant_tf.values(), pid_tf, self.sim_time_group.value
        )

        # Render the step info
        self.render_step_info(step_info)

        self.plot_response(t_out, y_out, Kp_value, Ki_value, Kd_value)


if __name__ == "__main__":
    # Set the style of the application
    QApplication.setStyle(QStyleFactory.keys()[2])
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Execute the application
    sys.exit(app.exec())
