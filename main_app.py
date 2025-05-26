# Hasan Taha Bağcı 150210338
import sys
import os
import json
import time
import subprocess # Using subprocess instead of QProcess for simplicity in this example, QProcess is better for GUI integration.
                  # For a production app, QProcess is recommended for non-blocking execution and better signal/slot integration.
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QComboBox, QPushButton, QTextEdit, QLabel, QProgressBar, QGroupBox,
                             QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
                             QTabWidget, QScrollArea) 
from PyQt5.QtCore import QProcess, Qt, QThread, pyqtSignal # QProcess for external script, QThread for non-blocking tasks
from PyQt5.QtGui import QPixmap

# Matplotlib imports for plotting within PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt # For direct plot saving if needed

from utils import load_metrics_from_log, plot_comparison, show_error_message, show_info_message

RESULTS_DIR = './results' # Directory to store results
os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists

class MatplotlibCanvas(FigureCanvas): # Matplotlib canvas for embedding plots
    """A custom Matplotlib canvas to integrate with PyQt5."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi) # Create a figure
        self.axes = self.fig.add_subplot(111) # Add a subplot
        super(MatplotlibCanvas, self).__init__(self.fig) # Initialize parent class
        self.setParent(parent) # Set parent

    def plot(self, x_data, y_data_list, labels, title, x_label, y_label): # Plotting function
        self.axes.clear() # Clear previous plot
        for y_data, label in zip(y_data_list, labels): # Plot each dataset
            if x_data and y_data and len(x_data) == len(y_data):
                 self.axes.plot(x_data, y_data, label=label)
            elif y_data: # If no x_data, use index
                 self.axes.plot(y_data, label=label)

        self.axes.set_title(title) # Set title
        self.axes.set_xlabel(x_label) # Set x-axis label
        self.axes.set_ylabel(y_label) # Set y-axis label
        if labels and any(labels): self.axes.legend() # Show legend if labels exist
        self.axes.grid(True) # Show grid
        self.draw() # Redraw canvas

class TrainingThread(QThread): # QThread for running training process to keep GUI responsive
    new_log_message = pyqtSignal(str) # Signal for new log messages
    training_progress = pyqtSignal(int, int) # Signal for epoch progress (current, total)
    metric_update = pyqtSignal(dict) # Signal for new metric data (for plotting)
    training_finished = pyqtSignal(str, str, str) # Signal when training is done (run_id, acc_plot_path, loss_plot_path)
    training_error = pyqtSignal(str) # Signal for errors during training

    def __init__(self, model_name, epochs, lr, batch_size, augment, results_dir):
        super().__init__()
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.augment = augment
        self.results_dir = results_dir
        self.run_id = f"{self.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.process = None # QProcess instance

    def run(self): # Executed when thread starts
        self.process = QProcess() # Create QProcess
        self.process.setProcessChannelMode(QProcess.MergedChannels) # Merge stdout and stderr

        # Connect signals from QProcess
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.handle_finish)
        self.process.errorOccurred.connect(self.handle_error)

        python_executable = sys.executable # Get current python interpreter
        script_path = os.path.join(os.path.dirname(__file__), 'train.py') # Path to train.py

        if not os.path.exists(script_path):
            self.training_error.emit(f"Training script 'train.py' not found at {script_path}")
            return

        command_args = [ # Arguments for train.py
            '--model_name', self.model_name,
            '--epochs', str(self.epochs),
            '--lr', str(self.lr),
            '--batch_size', str(self.batch_size),
            '--augment', str(self.augment),
            '--results_dir', self.results_dir,
            # run_id is generated internally by train.py based on its args, but we pass our GUI run_id for consistency if train.py is modified to accept it
        ]
        
        self.new_log_message.emit(f"Starting training process for {self.model_name} with ID: {self.run_id}...")
        self.new_log_message.emit(f"Command: {python_executable} {script_path} {' '.join(command_args)}")
        
        try:
            self.process.start(python_executable, [script_path] + command_args) # Start the process
        except Exception as e:
            self.training_error.emit(f"Failed to start QProcess: {str(e)}")


    def handle_stdout(self): # Handle output from train.py
        data = self.process.readAllStandardOutput().data().decode().strip()
        for line in data.splitlines(): # Process each line of output
            self.new_log_message.emit(line) # Emit raw log
            try:
                parsed_json = json.loads(line) # Try to parse as JSON
                if parsed_json.get("type") == "metric":
                    self.metric_update.emit(parsed_json)
                    self.training_progress.emit(parsed_json.get("epoch", 0), parsed_json.get("epochs", self.epochs))
                elif parsed_json.get("type") == "error":
                    self.training_error.emit(f"Error from training script: {parsed_json.get('message', 'Unknown error')}")
                elif parsed_json.get("type") == "summary": # Check for summary message
                    acc_plot = parsed_json.get("accuracy_plot_path", "")
                    loss_plot = parsed_json.get("loss_plot_path", "")
                    # The run_id used by train.py might differ if it generates its own. We need the one it used.
                    # For now, we assume the metrics_log_path contains the run_id.
                    metrics_log = parsed_json.get("metrics_log_path", "")
                    actual_run_id = os.path.basename(metrics_log).replace("_metrics.jsonl", "") if metrics_log else self.run_id
                    self.training_finished.emit(actual_run_id, acc_plot, loss_plot)

            except json.JSONDecodeError:
                pass # Not a JSON line, just a regular log message

    def handle_finish(self, exit_code, exit_status): # Handle process finish
        status_msg = "successfully" if exit_status == QProcess.NormalExit and exit_code == 0 else f"with errors (Code: {exit_code}, Status: {exit_status})"
        self.new_log_message.emit(f"Training process for {self.run_id} finished {status_msg}.")
        # If summary was not caught, emit finished signal with potentially empty plot paths
        # This ensures the GUI knows the process ended.
        # self.training_finished.emit(self.run_id, "", "") # This might be redundant if summary is always sent

    def handle_error(self, error): # Handle QProcess errors
        self.training_error.emit(f"QProcess Error for {self.run_id}: {self.process.errorString()}")

    def stop_training(self): # Method to stop training
        if self.process and self.process.state() == QProcess.Running:
            self.new_log_message.emit(f"Attempting to stop training for {self.run_id}...")
            self.process.terminate() # Try to terminate gracefully
            if not self.process.waitForFinished(3000): # Wait 3s
                self.process.kill() # Force kill
                self.new_log_message.emit(f"Training for {self.run_id} forcefully killed.")
            else:
                self.new_log_message.emit(f"Training for {self.run_id} terminated.")
        self.quit() # Quit the thread


class MainWindow(QMainWindow): # Main application window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN Image Classification Analysis - Hasan Taha Bağcı 150210338") # Window title
        self.setGeometry(100, 100, 1200, 800) # Window size and position

        self.current_training_thread = None # To hold the current training thread
        self.current_run_id = None # ID of the current training run
        self.current_run_metrics = {'epochs':[], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # Metrics for current run

        self.init_ui() # Initialize UI elements

    def init_ui(self): # Initialize UI components
        main_widget = QWidget() # Main widget
        self.setCentralWidget(main_widget) # Set as central widget
        main_layout = QVBoxLayout(main_widget) # Main vertical layout

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Tab 1: Model Training ---
        self.train_tab = QWidget()
        self.tabs.addTab(self.train_tab, "Model Training")
        train_tab_layout = QHBoxLayout(self.train_tab) # Horizontal layout for training tab

        # Left Panel: Controls and Logs
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        train_tab_layout.addWidget(left_panel, 1) # Takes 1/3 of space

        # Model Selection Group
        model_group = QGroupBox("1. Model Configuration")
        model_layout = QFormLayout()
        self.model_combo = QComboBox() # Dropdown for model selection
        self.model_combo.addItems(['CustomCNN', 'ResNet50', 'VGG16'])
        model_layout.addRow("Select Model:", self.model_combo)
        
        self.epochs_spin = QSpinBox() # Spinbox for epochs
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(10) # Default epochs
        model_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox() # Spinbox for learning rate
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001) # Default learning rate
        model_layout.addRow("Learning Rate:", self.lr_spin)

        self.batch_size_spin = QSpinBox() # Spinbox for batch size
        self.batch_size_spin.setRange(16, 512)
        self.batch_size_spin.setSingleStep(16)
        self.batch_size_spin.setValue(64) # Default batch size
        model_layout.addRow("Batch Size:", self.batch_size_spin)

        self.augment_checkbox = QCheckBox("Enable Data Augmentation") # Checkbox for augmentation
        self.augment_checkbox.setChecked(True) # Default to enabled
        model_layout.addRow(self.augment_checkbox)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # Training Control Group
        train_control_group = QGroupBox("2. Training Control")
        train_control_layout = QVBoxLayout()
        self.train_button = QPushButton("Start Training") # Button to start training
        self.train_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training") # Button to stop training
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False) # Initially disabled
        train_control_layout.addWidget(self.train_button)
        train_control_layout.addWidget(self.stop_button)
        train_control_group.setLayout(train_control_layout)
        left_layout.addWidget(train_control_group)

        # Log Display Group
        log_group = QGroupBox("3. Training Log")
        log_layout = QVBoxLayout()
        self.log_display = QTextEdit() # Text area for logs
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        self.progress_bar = QProgressBar() # Progress bar for training
        log_layout.addWidget(self.progress_bar)
        log_group.setLayout(log_layout)
        left_layout.addWidget(log_group)
        left_layout.addStretch()

        # Right Panel: Plots for current training
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        train_tab_layout.addWidget(right_panel, 2) # Takes 2/3 of space

        plot_group_training = QGroupBox("Live Training Metrics")
        plot_layout_training = QVBoxLayout()
        self.accuracy_canvas = MatplotlibCanvas(self, width=7, height=3.5) # Canvas for accuracy plot
        self.loss_canvas = MatplotlibCanvas(self, width=7, height=3.5) # Canvas for loss plot
        plot_layout_training.addWidget(self.accuracy_canvas)
        plot_layout_training.addWidget(self.loss_canvas)
        plot_group_training.setLayout(plot_layout_training)
        right_layout.addWidget(plot_group_training)

        # --- Tab 2: Results Comparison ---
        self.compare_tab = QWidget()
        self.tabs.addTab(self.compare_tab, "Results Comparison")
        compare_tab_layout = QVBoxLayout(self.compare_tab)

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

        self.comparison_plot_canvas = MatplotlibCanvas(self, width=8, height=6) # Canvas for comparison plot
        
        # Use QScrollArea for the comparison plot canvas if it might be large
        scroll_area_compare = QScrollArea()
        scroll_area_compare.setWidgetResizable(True)
        scroll_area_compare.setWidget(self.comparison_plot_canvas)
        compare_tab_layout.addWidget(scroll_area_compare)
        
        self.selected_run_files = [] # List to store paths of selected .jsonl files

    def start_training(self): # Function to start training process
        if self.current_training_thread and self.current_training_thread.isRunning():
            show_error_message(self, "Training Busy", "A training process is already running.")
            return

        model_name = self.model_combo.currentText()
        epochs = self.epochs_spin.value()
        lr = self.lr_spin.value()
        batch_size = self.batch_size_spin.value()
        augment = self.augment_checkbox.isChecked()

        self.log_display.clear() # Clear previous logs
        self.progress_bar.setValue(0) # Reset progress bar
        self.current_run_metrics = {'epochs':[], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []} # Reset metrics
        self.accuracy_canvas.axes.clear() # Clear plots
        self.loss_canvas.axes.clear()
        self.accuracy_canvas.draw()
        self.loss_canvas.draw()

        self.current_training_thread = TrainingThread(model_name, epochs, lr, batch_size, augment, RESULTS_DIR)
        self.current_run_id = self.current_training_thread.run_id # Store the run_id from the thread

        # Connect signals from the thread
        self.current_training_thread.new_log_message.connect(self.update_log_display)
        self.current_training_thread.training_progress.connect(self.update_progress_bar)
        self.current_training_thread.metric_update.connect(self.update_live_plots)
        self.current_training_thread.training_finished.connect(self.on_training_finished)
        self.current_training_thread.training_error.connect(self.on_training_error)
        
        self.current_training_thread.start() # Start the thread
        self.train_button.setEnabled(False) # Disable start button
        self.stop_button.setEnabled(True) # Enable stop button

    def stop_training(self): # Function to stop training
        if self.current_training_thread and self.current_training_thread.isRunning():
            self.current_training_thread.stop_training()
            # GUI updates (button states) will be handled in on_training_finished or on_training_error
        else:
            self.update_log_display("No active training process to stop.")
            self.train_button.setEnabled(True)
            self.stop_button.setEnabled(False)


    def update_log_display(self, message): # Update log display
        self.log_display.append(message)

    def update_progress_bar(self, current_epoch, total_epochs): # Update progress bar
        if total_epochs > 0:
            progress = int((current_epoch / total_epochs) * 100)
            self.progress_bar.setValue(progress)

    def update_live_plots(self, metric_data): # Update live plots with new metric data
        if metric_data.get("type") == "metric":
            epoch = metric_data.get("epoch")
            if epoch is not None and epoch not in self.current_run_metrics['epochs']: # Avoid duplicates if multiple updates for same epoch
                 self.current_run_metrics['epochs'].append(epoch)
                 self.current_run_metrics['train_loss'].append(metric_data.get("train_loss"))
                 self.current_run_metrics['train_acc'].append(metric_data.get("train_acc"))
                 self.current_run_metrics['val_loss'].append(metric_data.get("val_loss"))
                 self.current_run_metrics['val_acc'].append(metric_data.get("val_acc"))

            # Sort by epoch before plotting to ensure lines are drawn correctly if data arrives out of order
            sorted_indices = sorted(range(len(self.current_run_metrics['epochs'])), key=lambda k: self.current_run_metrics['epochs'][k])
            sorted_epochs = [self.current_run_metrics['epochs'][i] for i in sorted_indices]
            
            sorted_train_acc = [self.current_run_metrics['train_acc'][i] for i in sorted_indices]
            sorted_val_acc = [self.current_run_metrics['val_acc'][i] for i in sorted_indices]
            
            sorted_train_loss = [self.current_run_metrics['train_loss'][i] for i in sorted_indices]
            sorted_val_loss = [self.current_run_metrics['val_loss'][i] for i in sorted_indices]

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

    def on_training_finished(self, run_id, acc_plot_path, loss_plot_path): # Handle training finished signal
        self.update_log_display(f"Training finished for run: {run_id}.")
        if acc_plot_path and os.path.exists(acc_plot_path):
            self.update_log_display(f"Accuracy plot saved to: {acc_plot_path}")
        if loss_plot_path and os.path.exists(loss_plot_path):
            self.update_log_display(f"Loss plot saved to: {loss_plot_path}")
        
        self.train_button.setEnabled(True) # Re-enable start button
        self.stop_button.setEnabled(False) # Disable stop button
        self.progress_bar.setValue(100) # Set progress to 100%
        self.current_training_thread = None # Clear current thread

        show_info_message(self, "Training Complete", f"Training for {run_id} has finished.\nPlots saved in {RESULTS_DIR}")


    def on_training_error(self, error_message): # Handle training error signal
        self.update_log_display(f"ERROR: {error_message}")
        show_error_message(self, "Training Error", error_message)
        self.train_button.setEnabled(True) # Re-enable start button
        self.stop_button.setEnabled(False) # Disable stop button
        self.progress_bar.setValue(0) # Reset progress bar
        if self.current_training_thread:
            if self.current_training_thread.isRunning():
                self.current_training_thread.quit() # Ensure thread is quit
                self.current_training_thread.wait() # Wait for it to finish
            self.current_training_thread = None

    def select_run_files_for_comparison(self): # Select .jsonl files for comparison
        files, _ = QFileDialog.getOpenFileNames(self, "Select Metric Log Files", RESULTS_DIR, "JSONL files (*.jsonl)")
        if files:
            self.selected_run_files = files
            self.selected_files_label.setText(f"{len(files)} file(s) selected: {', '.join(os.path.basename(f) for f in files)}")
        else:
            self.selected_run_files = []
            self.selected_files_label.setText("No files selected.")

    def generate_comparison_plot_action(self): # Generate comparison plot
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
        # Extract the short metric name like 'val_acc'
        metric_short_name = metric_full_name[metric_full_name.find("(")+1:metric_full_name.find(")")]

        comparison_plot_filename = f"comparison_{metric_short_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        comparison_plot_path = os.path.join(RESULTS_DIR, comparison_plot_filename)

        # Use utils.plot_comparison to generate and save the plot
        plot_comparison(histories_to_compare, metric_short_name, comparison_plot_path)

        if os.path.exists(comparison_plot_path):
            # Display the plot in the GUI
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


    def closeEvent(self, event): # Handle window close event
        if self.current_training_thread and self.current_training_thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit', 
                                       "A training process is currently running. Are you sure you want to exit? This will stop the training.",
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.stop_training() # Attempt to stop training
                if self.current_training_thread: # Wait for thread to finish
                    self.current_training_thread.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv) # Create QApplication instance
    main_win = MainWindow() # Create MainWindow instance
    main_win.show() # Show main window
    sys.exit(app.exec_()) # Start application event loop
