# Hasan Taha Bağcı
# 150210338
# QThread Worker for Running Training Script via QProcess

from PyQt5.QtCore import QThread, pyqtSignal, QProcess
import json
import sys
import os

class TrainingWorker(QThread): # Training worker class (inherits QThread)
    log_message = pyqtSignal(str) # Signal for log messages
    tqdm_output_signal = pyqtSignal(str) # Signal for raw tqdm output lines
    epoch_metric_received = pyqtSignal(dict) # Signal for epoch metrics
    epoch_progress_update = pyqtSignal(int, int) # Signal for epoch progress (current, total)
    # batch_progress_update = pyqtSignal(int, int) # Optional: (current_batch, total_batches)
    training_finished = pyqtSignal(str) # Signal for when training is finished (model_name)
    training_error = pyqtSignal(str) # Signal for errors during training

    def __init__(self, training_params, parent=None): # Constructor
        super().__init__(parent)
        self.training_params = training_params # Store training parameters
        self.process = None # QProcess instance

    def run(self): # Executed when thread starts
        self.process = QProcess() # Create QProcess instance [cite: 18]
        self.process.setProcessChannelMode(QProcess.MergedChannels) # Merge stdout and stderr

        self.process.readyReadStandardOutput.connect(self._handle_stdout) # Connect stdout signal
        self.process.finished.connect(self._handle_finished) # Connect finished signal
        self.process.errorOccurred.connect(self._handle_error) # Connect error signal
        
        # Construct the command to run train_script.py
        script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training", "train_script.py") # Path to train_script.py
        # Ensure script_path is correct if main_app.py is in the root
        if not os.path.exists(script_path):
             # Assuming main_app.py is in comparative_cnn_analyzer/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Should be project root
            script_path = os.path.join(base_dir, "training", "train_script.py")


        python_executable = sys.executable # Path to current python interpreter
        
        command = [python_executable, script_path] # Command list
        command.extend([
            "--model_name", str(self.training_params["model_name"]),
            "--epochs", str(self.training_params["epochs"]),
            "--batch_size", str(self.training_params["batch_size"]),
            "--learning_rate", str(self.training_params["learning_rate"]),
            "--device", str(self.training_params["device"])
        ])
        if self.training_params["use_data_augmentation"]:
            command.append("--use_data_augmentation")
        if self.training_params["use_pretrained"]:
            command.append("--use_pretrained")
        
        self.log_message.emit(f"Starting process: {' '.join(command)}")
        self.process.start(command[0], command[1:]) # Start the process [cite: 21]

    def _handle_stdout(self): # Handler for stdout data
        data = self.process.readAllStandardOutput().data().decode().strip() # Read data
        lines = data.split('\n') # Split into lines
        for line in lines:
            line = line.strip() # Strip whitespace
            if not line: continue

            if line.startswith("EPOCH_METRIC:"): # Check for epoch metric JSON [cite: 18]
                try:
                    metric_json = line.replace("EPOCH_METRIC:", "")
                    metrics = json.loads(metric_json)
                    self.epoch_metric_received.emit(metrics) # Emit metric signal
                    self.epoch_progress_update.emit(metrics.get("epoch", 0), metrics.get("total_epochs", 0))
                except json.JSONDecodeError as e:
                    self.log_message.emit(f"Error decoding metric JSON: {e} - Data: {line}")
            elif "Epoch" in line and ("Train" in line or "Val" in line) and ("it/s" in line or "%" in line): # Heuristic for tqdm line
                self.tqdm_output_signal.emit(line) # Emit tqdm output
            else: # General log message
                self.log_message.emit(line)

    def _handle_finished(self, exitCode, exitStatus): # Handler for process finished
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            self.log_message.emit(f"Training process finished successfully for {self.training_params['model_name']}.")
            self.training_finished.emit(self.training_params['model_name'])
        elif exitStatus == QProcess.CrashExit:
            self.log_message.emit(f"Training process crashed for {self.training_params['model_name']}.")
            self.training_error.emit(f"Training process crashed (Exit code: {exitCode}) for {self.training_params['model_name']}.")
        else:
            self.log_message.emit(f"Training process finished with code {exitCode} for {self.training_params['model_name']}.")
            self.training_error.emit(f"Training process failed (Exit code: {exitCode}) for {self.training_params['model_name']}.")


    def _handle_error(self, error): # Handler for QProcess errors
        error_string = self.process.errorString()
        self.log_message.emit(f"QProcess Error: {error_string}")
        self.training_error.emit(f"Failed to start/run training process: {error_string}")

    def stop_training(self): # Method to stop training
        if self.process and self.process.state() == QProcess.Running:
            self.log_message.emit("Attempting to stop training process...")
            self.process.terminate() # Try to terminate gracefully
            if not self.process.waitForFinished(5000): # Wait 5s
                self.log_message.emit("Process did not terminate gracefully, killing...")
                self.process.kill() # Force kill
                self.process.waitForFinished() # Wait for it to be killed
            self.log_message.emit("Training process stopped by user.")