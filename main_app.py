# Hasan Taha Bağcı 150210338
import sys
import os
import json
import time
# import subprocess # Keep QProcess for better GUI integration
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QPushButton, QTextEdit, QLabel, QProgressBar, QGroupBox,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QTabWidget, QScrollArea, QMessageBox)
from PyQt5.QtCore import QProcess, Qt, QThread, pyqtSignal 
from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt 

from utils import load_metrics_from_log, plot_comparison, show_error_message, show_info_message

RESULTS_DIR = './results' 
os.makedirs(RESULTS_DIR, exist_ok=True) 

class MatplotlibCanvas(FigureCanvas): 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi) 
        self.axes = self.fig.add_subplot(111) 
        super(MatplotlibCanvas, self).__init__(self.fig) 
        self.setParent(parent) 

    def plot(self, x_data, y_data_list, labels, title, x_label, y_label): 
        self.axes.clear() 
        for y_data, label in zip(y_data_list, labels): 
            if x_data and y_data and len(x_data) == len(y_data):
                 self.axes.plot(x_data, y_data, label=label)
            elif y_data: 
                 self.axes.plot(y_data, label=label)

        self.axes.set_title(title) 
        self.axes.set_xlabel(x_label) 
        self.axes.set_ylabel(y_label) 
        if labels and any(labels): self.axes.legend() 
        self.axes.grid(True) 
        self.draw() 

class TrainingThread(QThread): 
    new_log_message = pyqtSignal(str) 
    # training_progress = pyqtSignal(int, int) # Original epoch progress signal (can be kept or removed if batch progress covers it)
    epoch_status_update = pyqtSignal(int, int) # For Epoch X/Y display
    batch_update = pyqtSignal(int, int, int, float) # epoch_num, current_batch, total_batches, batch_loss
    metric_update = pyqtSignal(dict) 
    training_finished = pyqtSignal(str, str, str) 
    training_error = pyqtSignal(str) 

    def __init__(self, model_name, epochs, lr, batch_size, augment, results_dir):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.augment = augment
        self.results_dir = results_dir
        self.run_id = f"{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.process = None 

    def run(self): 
        self.process = QProcess() 
        self.process.setProcessChannelMode(QProcess.MergedChannels) 

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.handle_finish)
        self.process.errorOccurred.connect(self.handle_error)

        python_executable = sys.executable 
        script_path = os.path.join(os.path.dirname(__file__), 'train.py') 

        if not os.path.exists(script_path):
            self.training_error.emit(f"Training script 'train.py' not found at {script_path}")
            return

        command_args = [ 
            '--model_name', self.model_name,
            '--epochs', str(self.epochs),
            '--lr', str(self.lr),
            '--batch_size', str(self.batch_size),
            '--augment', str(self.augment),
            '--results_dir', self.results_dir,
        ]
        
        self.new_log_message.emit(f"Starting training process for {self.model_name} with ID: {self.run_id}...")
        self.new_log_message.emit(f"Command: {python_executable} {script_path} {' '.join(command_args)}")
        
        try:
            self.process.start(python_executable, [script_path] + command_args) 
        except Exception as e:
            self.training_error.emit(f"Failed to start QProcess: {str(e)}")


    def handle_stdout(self): 
        data = self.process.readAllStandardOutput().data().decode().strip()
        for line in data.splitlines(): 
            self.new_log_message.emit(line) 
            try:
                parsed_json = json.loads(line) 
                msg_type = parsed_json.get("type")

                if msg_type == "metric":
                    self.metric_update.emit(parsed_json)
                    # Update epoch status when a full epoch metric is received
                    self.epoch_status_update.emit(parsed_json.get("epoch", 0), parsed_json.get("epochs", self.epochs))
                elif msg_type == "batch_progress":
                    epoch_num = parsed_json.get("epoch", 0)
                    current_batch = parsed_json.get("current_batch", 0)
                    total_batches = parsed_json.get("total_batches", 1) # Avoid division by zero
                    batch_loss = parsed_json.get("batch_loss", 0.0)
                    self.batch_update.emit(epoch_num, current_batch, total_batches, batch_loss)
                    # Also update the general epoch status label from batch progress
                    self.epoch_status_update.emit(epoch_num, parsed_json.get("epochs", self.epochs))

                elif msg_type == "error":
                    self.training_error.emit(f"Error from training script: {parsed_json.get('message', 'Unknown error')}")
                elif msg_type == "summary": 
                    acc_plot = parsed_json.get("accuracy_plot_path", "")
                    loss_plot = parsed_json.get("loss_plot_path", "")
                    metrics_log = parsed_json.get("metrics_log_path", "")
                    actual_run_id = os.path.basename(metrics_log).replace("_metrics.jsonl", "") if metrics_log else self.run_id
                    self.training_finished.emit(actual_run_id, acc_plot, loss_plot)
                # elif msg_type == "log": # General log messages, already handled by new_log_message

            except json.JSONDecodeError:
                pass # Not a JSON line, just a regular log message (already emitted)

    def handle_finish(self, exit_code, exit_status): 
        status_msg = "successfully" if exit_status == QProcess.NormalExit and exit_code == 0 else f"with errors (Code: {exit_code}, Status: {exit_status})"
        self.new_log_message.emit(f"Training process for {self.run_id} finished {status_msg}.")
        # If summary was not caught, might need to emit training_finished here
        # For now, assume summary will be sent on normal completion.

    def handle_error(self, error): 
        self.training_error.emit(f"QProcess Error for {self.run_id}: {self.process.errorString()}")

    def stop_training(self): 
        if self.process and self.process.state() == QProcess.Running:
            self.new_log_message.emit(f"Attempting to stop training for {self.run_id}...")
            self.process.terminate() 
            if not self.process.waitForFinished(3000): 
                self.process.kill() 
                self.new_log_message.emit(f"Training for {self.run_id} forcefully killed.")
            else:
                self.new_log_message.emit(f"Training for {self.run_id} terminated.")
        self.quit() 


class MainWindow(QMainWindow): 
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Image Classification Analysis - Hasan Taha Bağcı 150210338") 
        self.setGeometry(100, 100, 1200, 800) 

        self.current_training_thread = None 
        self.current_run_id = None 
        self.current_run_metrics = {'epochs':[], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 

        self.init_ui() 

    def init_ui(self): 
        main_widget = QWidget() 
        self.setCentralWidget(main_widget) 
        main_layout = QVBoxLayout(main_widget) 

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.train_tab = QWidget()
        self.tabs.addTab(self.train_tab, "Model Training")
        train_tab_layout = QHBoxLayout(self.train_tab) 

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        train_tab_layout.addWidget(left_panel, 1) 

        model_group = QGroupBox("1. Model Configuration")
        model_layout = QFormLayout()
        self.model_combo = QComboBox() 
        self.model_combo.addItems(['CustomCNN', 'ResNet50', 'VGG16'])
        model_layout.addRow("Select Model:", self.model_combo)
        
        self.epochs_spin = QSpinBox() 
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(10) 
        model_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox() 
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001) 
        model_layout.addRow("Learning Rate:", self.lr_spin)

        self.batch_size_spin = QSpinBox() 
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setSingleStep(16)
        self.batch_size_spin.setValue(64) 
        model_layout.addRow("Batch Size:", self.batch_size_spin)

        self.augment_checkbox = QCheckBox("Enable Data Augmentation") 
        self.augment_checkbox.setChecked(True) 
        model_layout.addRow(self.augment_checkbox)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        train_control_group = QGroupBox("2. Training Control")
        train_control_layout = QVBoxLayout()
        self.train_button = QPushButton("Start Training") 
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training") 
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False) 
        train_control_layout.addWidget(self.train_button)
        train_control_layout.addWidget(self.stop_button)
        train_control_group.setLayout(train_control_layout)
        left_layout.addWidget(train_control_group)

        log_group = QGroupBox("3. Training Log & Progress") # Renamed group
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit() 
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        # Progress display
        progress_info_layout = QHBoxLayout() # Layout for epoch status and batch loss
        self.epoch_status_label = QLabel("Epoch: -/-") # Label for Epoch X/Y
        self.batch_loss_label = QLabel("Batch Loss: -.----") # Label for current batch loss
        progress_info_layout.addWidget(self.epoch_status_label)
        progress_info_layout.addStretch()
        progress_info_layout.addWidget(self.batch_loss_label)
        log_layout.addLayout(progress_info_layout)

        self.batch_progress_bar = QProgressBar() # Progress bar for batch progress within an epoch
        self.batch_progress_bar.setFormat("%v/%m (%p%)") # Show current_batch/total_batches
        log_layout.addWidget(self.batch_progress_bar)
        
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        train_tab_layout.addWidget(right_panel, 2) 

        plot_group_training = QGroupBox("Live Training Metrics")
        plot_layout_training = QVBoxLayout()
        self.accuracy_canvas = MatplotlibCanvas(self, width=7, height=3.5) 
        self.loss_canvas = MatplotlibCanvas(self, width=7, height=3.5) 
        plot_layout_training.addWidget(self.accuracy_canvas)
        plot_layout_training.addWidget(self.loss_canvas)
        plot_group_training.setLayout(plot_layout_training)
        right_layout.addWidget(plot_group_training)

        self.compare_tab = QWidget()
        self.tabs.addTab(self.compare_tab, "Results Comparison")
        compare_tab_layout = QVBoxLayout(self.compare_tab) # Defined here
        # ... (rest of compare_tab UI - no changes needed here for batch progress) ...
        compare_controls_group = QGroupBox("Select Runs to Compare")
        compare_controls_layout = QFormLayout()
        self.select_runs_button = QPushButton("Select Result Files (.jsonl)")
        self.select_runs_button.clicked.connect(self.select_run_files_for_comparison)
        self.selected_files_label = QLabel("No files selected.")
        self.selected_files_label.setWordWrap(True)
        compare_controls_layout.addRow(self.select_runs_button)
        compare_controls_layout.addRow(QLabel("Selected:"), self.selected_files_label)
        
        self.compare_metric_combo = QComboBox()
        self.compare_metric_combo.addItems(["Validation Accuracy (val_acc)", "Validation Loss (val_loss)", "Training Accuracy (train_acc)", "Training Loss (train_loss)"])
        compare_controls_layout.addRow("Metric to Compare:", self.compare_metric_combo)

        self.generate_comparison_button = QPushButton("Generate Comparison Plot")
        self.generate_comparison_button.clicked.connect(self.generate_comparison_plot_action)
        compare_controls_layout.addRow(self.generate_comparison_button)
        compare_controls_group.setLayout(compare_controls_layout)
        compare_tab_layout.addWidget(compare_controls_group)

        self.comparison_plot_canvas = MatplotlibCanvas(self, width=8, height=6) 
        
        scroll_area_compare = QScrollArea()
        scroll_area_compare.setWidgetResizable(True)
        scroll_area_compare.setWidget(self.comparison_plot_canvas)
        compare_tab_layout.addWidget(scroll_area_compare)
        
        self.selected_run_files = [] 


    def start_training(self): 
        if self.current_training_thread and self.current_training_thread.isRunning():
            show_error_message(self, "Training Busy", "A training process is already running.")
            return

        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        lr = self.lr_spin.value()
        batch_size = self.batch_size_spin.value()
        augment = self.augment_checkbox.isChecked()

        self.log_display.clear() 
        self.batch_progress_bar.setValue(0) 
        self.batch_progress_bar.setMaximum(100) # Default max, will be updated
        self.epoch_status_label.setText("Epoch: -/-")
        self.batch_loss_label.setText("Batch Loss: -.----")
        self.current_run_metrics = {'epochs':[], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} 
        self.accuracy_canvas.axes.clear(); self.accuracy_canvas.draw()
        self.loss_canvas.axes.clear(); self.loss_canvas.draw()

        self.current_training_thread = TrainingThread(model_name, epochs, lr, batch_size, augment, RESULTS_DIR)
        self.current_run_id = self.current_training_thread.run_id 

        self.current_training_thread.new_log_message.connect(self.update_log_display)
        # self.current_training_thread.training_progress.connect(self.update_progress_bar) # Old signal
        self.current_training_thread.epoch_status_update.connect(self.update_epoch_status_display) # New signal for epoch label
        self.current_training_thread.batch_update.connect(self.update_batch_progress) # New signal for batch bar
        self.current_training_thread.metric_update.connect(self.update_live_plots)
        self.current_training_thread.training_finished.connect(self.on_training_finished)
        self.current_training_thread.training_error.connect(self.on_training_error)
        
        self.current_training_thread.start() 
        self.train_button.setEnabled(False) 
        self.stop_button.setEnabled(True) 

    def stop_training(self): 
        if self.current_training_thread and self.current_training_thread.isRunning():
            self.current_training_thread.stop_training()
        else:
            self.update_log_display("No active training process to stop.")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)


    def update_log_display(self, message): 
        self.log_display.append(message)

    def update_epoch_status_display(self, current_epoch, total_epochs): # New slot
        self.epoch_status_label.setText(f"Epoch: {current_epoch}/{total_epochs}")
        if current_epoch > 0 and total_epochs > 0 and current_epoch == total_epochs: # If it's the last epoch, and batch progress completes it
             if self.batch_progress_bar.value() == self.batch_progress_bar.maximum():
                 pass # Already 100%
        elif current_epoch > 0 and self.batch_progress_bar.value() == self.batch_progress_bar.maximum(): # Reset for new epoch
            self.batch_progress_bar.setValue(0)


    def update_batch_progress(self, epoch_num, current_batch, total_batches, batch_loss): # New slot
        if total_batches > 0: # Ensure total_batches is valid
            self.batch_progress_bar.setMaximum(total_batches)
            self.batch_progress_bar.setValue(current_batch)
            self.batch_loss_label.setText(f"Batch Loss: {batch_loss:.4f}")
        
        # Update epoch status label as well, as batch progress implies current epoch
        # This might be redundant if epoch_status_update is also called, but ensures it's current
        # total_epochs_overall = self.epochs_spin.value() # Get overall total epochs
        # self.epoch_status_label.setText(f"Epoch: {epoch_num}/{total_epochs_overall}")


    def update_live_plots(self, metric_data): 
        if metric_data.get("type") == "metric":
            epoch = metric_data.get("epoch")
            
            if epoch is None: 
                self.update_log_display(f"Warning: Received metric_data with no epoch: {metric_data}")
                return

            new_train_loss = metric_data.get("train_loss")
            new_train_acc = metric_data.get("train_acc")
            new_val_loss = metric_data.get("val_loss")
            new_val_acc = metric_data.get("val_acc")

            try:
                idx = self.current_run_metrics['epochs'].index(epoch)
                self.current_run_metrics['train_loss'][idx] = new_train_loss
                self.current_run_metrics['train_acc'][idx] = new_train_acc
                self.current_run_metrics['val_loss'][idx] = new_val_loss
                self.current_run_metrics['val_acc'][idx] = new_val_acc
            except ValueError:
                self.current_run_metrics['epochs'].append(epoch)
                self.current_run_metrics['train_loss'].append(new_train_loss)
                self.current_run_metrics['train_acc'].append(new_train_acc)
                self.current_run_metrics['val_loss'].append(new_val_loss)
                self.current_run_metrics['val_acc'].append(new_val_acc)

            if not self.current_run_metrics['epochs']: return

            temp_metrics_list = []
            for i in range(len(self.current_run_metrics['epochs'])):
                temp_metrics_list.append((
                    self.current_run_metrics['epochs'][i],
                    self.current_run_metrics['train_loss'][i],
                    self.current_run_metrics['train_acc'][i],
                    self.current_run_metrics['val_loss'][i],
                    self.current_run_metrics['val_acc'][i]
                ))
            
            temp_metrics_list.sort(key=lambda x: x[0]) 

            sorted_epochs = [m[0] for m in temp_metrics_list]
            sorted_train_loss = [m[1] for m in temp_metrics_list]
            sorted_train_acc = [m[2] for m in temp_metrics_list]
            sorted_val_loss = [m[3] for m in temp_metrics_list]
            sorted_val_acc = [m[4] for m in temp_metrics_list]
            
            self.accuracy_canvas.plot(
                x_data=sorted_epochs,
                y_data_list=[sorted_train_acc, sorted_val_acc], 
                labels=['Train Accuracy', 'Validation Accuracy'],
                title=f'{self.current_run_id} - Accuracy',
                x_label='Epoch', y_label='Accuracy (%)'
            )
            self.loss_canvas.plot(
                x_data=sorted_epochs,
                y_data_list=[sorted_train_loss, sorted_val_loss], 
                labels=['Train Loss', 'Validation Loss'],
                title=f'{self.current_run_id} - Loss',
                x_label='Epoch', y_label='Loss'
            )

    def on_training_finished(self, run_id, acc_plot_path, loss_plot_path): 
        self.update_log_display(f"Training finished for run: {run_id}.")
        if acc_plot_path and os.path.exists(acc_plot_path):
            self.update_log_display(f"Accuracy plot saved to: {acc_plot_path}")
        if loss_plot_path and os.path.exists(loss_plot_path):
            self.update_log_display(f"Loss plot saved to: {loss_plot_path}")
        
        self.train_button.setEnabled(True) 
        self.stop_button.setEnabled(False) 
        self.batch_progress_bar.setValue(self.batch_progress_bar.maximum()) # Ensure it shows 100% at the end
        # Consider setting epoch_status_label to final state if not already
        
        if self.current_training_thread and not self.current_training_thread.isRunning():
            self.current_training_thread = None 

        show_info_message(self, "Training Complete", f"Training for {run_id} has finished.\nPlots saved in {RESULTS_DIR}")


    def on_training_error(self, error_message): 
        self.update_log_display(f"ERROR: {error_message}")
        show_error_message(self, "Training Error", error_message)
        self.train_button.setEnabled(True) 
        self.stop_button.setEnabled(False) 
        self.batch_progress_bar.setValue(0) 
        self.epoch_status_label.setText("Epoch: Error")
        self.batch_loss_label.setText("Batch Loss: -.----")
        if self.current_training_thread:
            if self.current_training_thread.isRunning():
                self.current_training_thread.quit() 
                self.current_training_thread.wait() 
            self.current_training_thread = None

    def select_run_files_for_comparison(self): # Function to select .jsonl files for comparison
        files, _ = QFileDialog.getOpenFileNames(self, "Select Metric Log Files", RESULTS_DIR, "JSONL files (*.jsonl)")
        if files:
            self.selected_run_files = files
            self.selected_files_label.setText(f"{len(files)} file(s) selected: {', '.join(os.path.basename(f) for f in files)}")
        else:
            self.selected_run_files = []
            self.selected_files_label.setText("No files selected.")

    def generate_comparison_plot_action(self): # Function to generate comparison plot
        if not self.selected_run_files:
            show_error_message(self, "No Files Selected", "Please select at least one .jsonl result file to compare.")
            return

        histories_to_compare = {}
        for file_path in self.selected_run_files:
            run_id = os.path.basename(file_path).replace("_metrics.jsonl", "") # Extract run ID from filename
            history = load_metrics_from_log(file_path)
            if history:
                histories_to_compare[run_id] = history
            else:
                self.update_log_display(f"Warning: Could not load metrics from {file_path}")
        
        if not histories_to_compare:
            show_error_message(self, "Error Loading Metrics", "No valid metrics could be loaded from the selected files.")
            return

        metric_full_name = self.compare_metric_combo.currentText() # e.g. "Validation Accuracy (val_acc)"
        metric_short_name = metric_full_name[metric_full_name.find("(")+1:metric_full_name.find(")")]

        comparison_plot_filename = f"comparison_{metric_short_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        comparison_plot_path = os.path.join(RESULTS_DIR, comparison_plot_filename)

        plot_comparison(histories_to_compare, metric_short_name, comparison_plot_path)

        if os.path.exists(comparison_plot_path):
            pixmap = QPixmap(comparison_plot_path)
            self.comparison_plot_canvas.axes.clear()
            self.comparison_plot_canvas.axes.imshow(plt.imread(comparison_plot_path)) # Display saved image
            self.comparison_plot_canvas.axes.axis('off') # Hide axes for image display
            self.comparison_plot_canvas.draw()
            show_info_message(self, "Comparison Generated", f"Comparison plot saved to:\n{comparison_plot_path}\nand displayed.")
            self.update_log_display(f"Comparison plot for '{metric_short_name}' generated: {comparison_plot_path}")
        else:
            show_error_message(self, "Plot Error", f"Failed to generate or find comparison plot at:\n{comparison_plot_path}")
            self.update_log_display(f"Error generating comparison plot for '{metric_short_name}'.")


    def closeEvent(self, event): 
        if self.current_training_thread and self.current_training_thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit', 
                                       "A training process is currently running. Are you sure you want to exit? This will stop the training.",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training() 
                if self.current_training_thread: 
                    if self.current_training_thread.isRunning():
                         self.current_training_thread.wait() 
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv) 
    main_win = MainWindow() 
    main_win.show() 
    sys.exit(app.exec_()) 
