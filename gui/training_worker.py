# Hasan Taha Bağcı
# 150210338
# QThread Worker for Running Training Script via QProcess

from PyQt5.QtCore import QThread, pyqtSignal, QProcess, QProcessEnvironment
import json
import sys
import os

class TrainingWorker(QThread): # Training worker class (inherits QThread)
    log_message = pyqtSignal(str) # Signal for log messages
    tqdm_output_signal = pyqtSignal(str) # Signal for raw tqdm output lines
    epoch_metric_received = pyqtSignal(dict) # Signal for epoch metrics
    epoch_progress_update = pyqtSignal(int, int) # Signal for epoch progress (current, total)
    training_finished = pyqtSignal(str) # Signal for when training is finished (model_name)
    training_error = pyqtSignal(str) # Signal for errors during training

    def __init__(self, training_params, parent=None): # Constructor
        super().__init__(parent)
        self.training_params = training_params # Store training parameters
        self.process = None # QProcess instance

    def run(self): # Executed when thread starts
        self.log_message.emit("DEBUG: TrainingWorker thread 'run' method started.")
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.finished.connect(self._handle_finished)
        self.process.errorOccurred.connect(self._handle_error)
        
        project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(project_root_dir, "training", "train_script.py")

        self.log_message.emit(f"DEBUG: Project root determined as: {project_root_dir}")
        self.log_message.emit(f"DEBUG: Attempting to run script at path: {script_path}")
        script_exists = os.path.exists(script_path)
        self.log_message.emit(f"DEBUG: Script at path exists: {script_exists}")

        if not script_exists:
            self.log_message.emit(f"ERROR: train_script.py not found at calculated path: {script_path}")
            self.training_error.emit(f"Critical error: train_script.py not found at {script_path}. Please check file structure.")
            return

        python_executable = sys.executable
        self.log_message.emit(f"DEBUG: Using Python interpreter: {python_executable}")
        
        # Explicitly set the working directory for the QProcess
        self.process.setWorkingDirectory(project_root_dir)
        self.log_message.emit(f"DEBUG: QProcess working directory explicitly set to: {project_root_dir}")

        # Log current environment that QProcess will inherit (or we can set a custom one)
        current_env = QProcessEnvironment.systemEnvironment()
        self.log_message.emit(f"DEBUG SysEnv: Python Executable from sys: {sys.executable}")
        self.log_message.emit(f"DEBUG SysEnv: PATH from system env: {current_env.value('PATH', 'Not set')}")
        self.log_message.emit(f"DEBUG SysEnv: PYTHONPATH from system env: {current_env.value('PYTHONPATH', 'Not set or empty')}")
        # For MPS debugging on macOS, certain env vars can be relevant if set
        self.log_message.emit(f"DEBUG SysEnv: METAL_DEVICE_WRAPPER_TYPE: {current_env.value('METAL_DEVICE_WRAPPER_TYPE', 'Not set')}")
        self.log_message.emit(f"DEBUG SysEnv: PYTORCH_ENABLE_MPS_FALLBACK: {current_env.value('PYTORCH_ENABLE_MPS_FALLBACK', 'Not set')}")
        
        # self.process.setProcessEnvironment(current_env) # You can explicitly set it if needed

        command_args = [ # Arguments for the script
            script_path, # Script itself is the first argument to python_executable for QProcess if not using shell
            "--model_name", str(self.training_params["model_name"]),
            "--epochs", str(self.training_params["epochs"]),
            "--batch_size", str(self.training_params["batch_size"]),
            "--learning_rate", str(self.training_params["learning_rate"]),
            "--device", str(self.training_params["device"])
        ]
        if self.training_params["use_data_augmentation"]:
            command_args.append("--use_data_augmentation")
        if self.training_params["use_pretrained"]:
            command_args.append("--use_pretrained")
        
        self.log_message.emit(f"DEBUG: QProcess executable: {python_executable}")
        self.log_message.emit(f"DEBUG: QProcess arguments: {command_args}")
        self.log_message.emit("DEBUG: Attempting to start QProcess...")
        
        # Note: QProcess takes executable and a list of arguments.
        # The script_path is an argument to the python_executable.
        self.process.start(python_executable, command_args)
        
        self.log_message.emit("DEBUG: QProcess.start() has been called. Waiting for signals.")

    def _handle_stdout(self): # Handler for stdout data
        data = self.process.readAllStandardOutput().data().decode(errors='replace').strip()
        if not data: return

        self.log_message.emit(f"RAW_STDOUT: {data}") # Log all raw output
        lines = data.split('\n')
        for line in lines:
            line = line.strip()
            if not line: continue

            if line.startswith("EPOCH_METRIC:"):
                try:
                    metric_json = line.replace("EPOCH_METRIC:", "")
                    metrics = json.loads(metric_json)
                    self.epoch_metric_received.emit(metrics)
                    self.epoch_progress_update.emit(metrics.get("epoch", 0), metrics.get("total_epochs", 0))
                except json.JSONDecodeError as e:
                    self.log_message.emit(f"Error decoding metric JSON: {e} - Data: {line}")
            elif "Epoch" in line and ("Train" in line or "Val" in line) and ("it/s" in line or "%" in line):
                self.tqdm_output_signal.emit(line)
            else:
                # self.log_message.emit(line) # Already logged by RAW_STDOUT
                pass

    def _handle_finished(self, exitCode, exitStatus): # Handler for process finished
        status_string = "NormalExit" if exitStatus == QProcess.NormalExit else "CrashExit"
        self.log_message.emit(f"DEBUG: QProcess finished. ExitCode: {exitCode}, ExitStatus: {status_string}")
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            self.log_message.emit(f"Training process finished successfully for {self.training_params['model_name']}.")
            self.training_finished.emit(self.training_params['model_name'])
        elif exitStatus == QProcess.CrashExit:
            self.log_message.emit(f"Training process crashed for {self.training_params['model_name']}. ExitCode: {exitCode}")
            self.training_error.emit(f"Training process CRASHED (Exit code: {exitCode}) for {self.training_params['model_name']}.")
        else:
            self.log_message.emit(f"Training process for {self.training_params['model_name']} finished with non-zero exit code {exitCode}.")
            self.training_error.emit(f"Training process FAILED (Exit code: {exitCode}) for {self.training_params['model_name']}.")

    def _handle_error(self, error): # Handler for QProcess errors
        error_string = self.process.errorString()
        error_name_str = error.name if hasattr(error, 'name') else str(error) # QProcess.ProcessError enum
        self.log_message.emit(f"ERROR: QProcess errorOccurred: {error_name_str} - {error_string}")
        self.training_error.emit(f"QProcess error: {error_string} (Code: {error_name_str})")

    def stop_training(self): # Method to stop training
        if self.process and self.process.state() == QProcess.Running:
            self.log_message.emit("Attempting to stop training process...")
            self.process.terminate()
            if not self.process.waitForFinished(5000):
                self.log_message.emit("Process did not terminate gracefully, killing...")
                self.process.kill()
                self.process.waitForFinished()
            self.log_message.emit("Training process stopped by user.")
        elif self.process:
            self.log_message.emit(f"Stop requested, but process not running. State: {self.process.state()}")
        else:
            self.log_message.emit("Stop requested, but no process was found.")