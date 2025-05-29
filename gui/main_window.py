# Hasan Taha Bağcı 150210338
# gui/main_window.py

import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFormLayout, QGroupBox, QLabel, QComboBox, QLineEdit, QPushButton, 
    QTextEdit, QSplitter, QCheckBox, QSpinBox, QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt, QProcess, QTimer

from .plot_widget import PlotWidget 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import get_available_devices
from models.cnn_models import AVAILABLE_MODELS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Trainer GUI - Hasan Taha Bagci 150210338")
        self.setGeometry(50, 50, 1200, 800) # Increased window size

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.training_process = None # For QProcess
        self.is_training_running = False

        self._init_ui()

    def _init_ui(self):
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setAlignment(Qt.AlignTop) # Align widgets to the top

        # Model Selection Group
        model_group = QGroupBox("Model Configuration")
        model_form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(list(AVAILABLE_MODELS.keys()))
        model_form.addRow("Select Model:", self.model_combo)
        
        self.pretrained_checkbox = QCheckBox("Use Pretrained Weights")
        self.pretrained_checkbox.setChecked(True)
        model_form.addRow(self.pretrained_checkbox)
        self.model_combo.currentTextChanged.connect(self._toggle_pretrained_option) # Enable/disable based on model
        self._toggle_pretrained_option(self.model_combo.currentText()) # Initial check

        model_group.setLayout(model_form)
        config_layout.addWidget(model_group)

        # Training Parameters Group
        params_group = QGroupBox("Training Parameters")
        params_form = QFormLayout()
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 500)
        self.epochs_spinbox.setValue(10)
        params_form.addRow("Epochs:", self.epochs_spinbox)

        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.00001, 0.1)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setValue(0.001)
        params_form.addRow("Learning Rate:", self.lr_spinbox)

        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(16, 512) # Common batch sizes
        self.batch_size_spinbox.setSingleStep(16)
        self.batch_size_spinbox.setValue(64)
        params_form.addRow("Batch Size:", self.batch_size_spinbox)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "SGD"])
        params_form.addRow("Optimizer:", self.optimizer_combo)
        
        self.augment_data_checkbox = QCheckBox("Augment Training Data")
        self.augment_data_checkbox.setChecked(True)
        params_form.addRow(self.augment_data_checkbox)

        params_group.setLayout(params_form)
        config_layout.addWidget(params_group)

        # Device Selection Group
        device_group = QGroupBox("Device Configuration")
        device_form = QFormLayout()
        self.device_combo = QComboBox()
        self.available_devices = get_available_devices()
        self.device_combo.addItems(self.available_devices)
        # Try to pre-select a GPU if available
        if 'mps' in self.available_devices:
            self.device_combo.setCurrentText('mps')
        elif 'cuda' in self.available_devices:
            self.device_combo.setCurrentText('cuda')
        device_form.addRow("Select Device:", self.device_combo)
        device_group.setLayout(device_form)
        config_layout.addWidget(device_group)
        
        # Control Buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; border-radius: 5px;")
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; padding: 10px; border-radius: 5px;")
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_group.setLayout(control_layout)
        config_layout.addWidget(control_group)

        output_panel = QWidget()
        output_layout = QVBoxLayout(output_panel)

        # Splitter for plots and logs
        splitter = QSplitter(Qt.Vertical)

        self.plot_widget = PlotWidget() # Our custom Matplotlib widget
        splitter.addWidget(self.plot_widget)

        self.log_output_area = QTextEdit()
        self.log_output_area.setReadOnly(True)
        self.log_output_area.setFontFamily("Courier") # Monospaced font for logs
        self.log_output_area.setLineWrapMode(QTextEdit.NoWrap) # No line wrapping for logs
        
        log_group = QGroupBox("Training Logs")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_output_area)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)
        
        splitter.setSizes([400, 200]) # Initial sizes for plot and log areas

        output_layout.addWidget(splitter)

        # Add panels to main layout
        self.main_layout.addWidget(config_panel, 1) # Configuration panel takes 1 part of space
        self.main_layout.addWidget(output_panel, 3) # Output panel takes 3 parts of space

    def _toggle_pretrained_option(self, model_name): # Enable/disable pretrained checkbox
        if model_name == "CustomCNN":
            self.pretrained_checkbox.setChecked(False)
            self.pretrained_checkbox.setEnabled(False)
        else: # ResNet50, VGG16
            self.pretrained_checkbox.setEnabled(True)

    def _collect_config(self): # Collects configuration from GUI elements
        config = {
            "model_name": self.model_combo.currentText(),
            "pretrained": self.pretrained_checkbox.isChecked() if self.pretrained_checkbox.isEnabled() else False,
            "num_epochs": self.epochs_spinbox.value(),
            "learning_rate": self.lr_spinbox.value(),
            "batch_size": self.batch_size_spinbox.value(),
            "optimizer_name": self.optimizer_combo.currentText(),
            "device": self.device_combo.currentText(),
            "augment_data": self.augment_data_checkbox.isChecked(),
        }
        return config

    def start_training(self):
        if self.is_training_running:
            QMessageBox.warning(self, "Training in Progress", "A training session is already running.")
            return

        self.log_output_area.clear() # Clear previous logs
        self.plot_widget.reset_plot_data() # Reset plots
        
        config = self._collect_config()
        self.log_output_area.append("Starting training with configuration:")
        self.log_output_area.append(json.dumps(config, indent=2))
        self.log_output_area.append("-" * 50 + "\n")

        self.training_process = QProcess(self)
        script_path = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), "..", "run_training.py")
        
        # If running from source (e.g. python main.py)
        if not getattr(sys, 'frozen', False): # not in a PyInstaller bundle
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            script_path = os.path.join(project_root, "run_training.py")
            python_executable = sys.executable # Use the same python interpreter
        else: # If bundled with PyInstaller
            base_path = os.path.dirname(sys.executable)
            script_path = os.path.join(base_path, "run_training.py")
            python_executable = sys.executable 

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            script_path = os.path.join(project_root, "run_training.py")
            python_executable = sys.executable


        self.training_process.readyReadStandardOutput.connect(self.handle_stdout)
        self.training_process.readyReadStandardError.connect(self.handle_stderr)
        self.training_process.finished.connect(self.training_finished)
        self.training_process.errorOccurred.connect(self.training_error)

        # Construct arguments for run_training.py
        args = []
        for key, value in config.items():
            args.append(f"--{key.replace('_', '-')}") # e.g., --model-name
            args.append(str(value))
        
        self.log_output_area.append(f"Executing: {python_executable} {script_path} {' '.join(args)}\n")
        
        try:
            self.training_process.start(python_executable, [script_path] + args)
            if self.training_process.waitForStarted(5000): # Wait 5s for process to start
                self.is_training_running = True
                self.start_button.setEnabled(False)
                self.stop_button.setEnabled(True)
                self.log_output_area.append("Training process started...\n")
            else:
                self.log_output_area.append("Error: Training process failed to start.")
                self.log_output_area.append(f"Process error: {self.training_process.errorString()}")
                self.is_training_running = False
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
        except Exception as e:
            self.log_output_area.append(f"Exception starting process: {str(e)}")
            QMessageBox.critical(self, "Process Error", f"Could not start training process: {str(e)}")
            self.is_training_running = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)


    def handle_stdout(self):
        data = self.training_process.readAllStandardOutput().data().decode().strip()
        if data:
            self.log_output_area.append(data)
            # Check for our specific metrics update prefix
            for line in data.splitlines():
                if line.startswith("METRICS_UPDATE:"):
                    try:
                        metrics_json = line.replace("METRICS_UPDATE:", "")
                        metrics = json.loads(metrics_json)
                        self.plot_widget.update_plot(
                            epoch=metrics["epoch"],
                            train_loss=metrics["train_loss"],
                            val_loss=metrics["val_loss"],
                            train_acc=metrics["train_acc"],
                            val_acc=metrics["val_acc"]
                        )
                    except json.JSONDecodeError as e:
                        self.log_output_area.append(f"Error decoding metrics JSON: {e} - Line: {line}")
                    except KeyError as e:
                        self.log_output_area.append(f"KeyError in metrics data: {e} - Data: {metrics_json}")
        self.log_output_area.verticalScrollBar().setValue(self.log_output_area.verticalScrollBar().maximum())


    def handle_stderr(self):
        data = self.training_process.readAllStandardError().data().decode().strip()
        if data:
            self.log_output_area.append(f"<font color='red'>{data}</font>") # Show errors in red
        self.log_output_area.verticalScrollBar().setValue(self.log_output_area.verticalScrollBar().maximum())

    def training_finished(self, exit_code, exit_status):
        self.log_output_area.append("-" * 50)
        if exit_status == QProcess.NormalExit and exit_code == 0:
            self.log_output_area.append("Training process finished successfully.")
        elif exit_status == QProcess.CrashExit:
            self.log_output_area.append(f"<font color='red'>Training process crashed.</font>")
        else:
            self.log_output_area.append(f"<font color='red'>Training process finished with exit code {exit_code}.</font>")
        
        self.is_training_running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.training_process = None # Clear the process

    def training_error(self, error):
        error_map = {
            QProcess.FailedToStart: "Failed to start",
            QProcess.Crashed: "Crashed",
            QProcess.Timedout: "Timed out",
            QProcess.ReadError: "Read error",
            QProcess.WriteError: "Write error",
            QProcess.UnknownError: "Unknown error"
        }
        error_string = error_map.get(error, "An unknown error occurred")
        self.log_output_area.append(f"<font color='red'>Training process error: {error_string} ({self.training_process.errorString()})</font>")


    def stop_training(self):
        if self.training_process and self.is_training_running:
            self.log_output_area.append("Attempting to stop training process...")
           
            self.training_process.terminate() # Sends SIGTERM
            
            # Give it a moment to terminate gracefully
            if not self.training_process.waitForFinished(3000): # Wait 3 seconds
                self.log_output_area.append("Process did not terminate gracefully, killing...")
                self.training_process.kill() # Sends SIGKILL

            # training_finished slot will handle UI updates
            self.stop_button.setEnabled(False) # Disable stop button immediately
        else:
            self.log_output_area.append("No training process to stop.")
    
    def closeEvent(self, event): # Handle closing the window while training
        if self.is_training_running and self.training_process:
            reply = QMessageBox.question(self, 'Training in Progress',
                                           "A training process is still running. Do you want to stop it and exit?",
                                           QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training()
                if self.training_process:
                    self.training_process.waitForFinished(1000) 
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
