# Hasan Taha Bağcı
# 150210338
# Main Application Window for CNN Comparative Analysis

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSplitter, QMessageBox, QPushButton, QDialog, QLabel)
from PyQt5.QtCore import Qt, pyqtSlot
import collections

from .widgets.control_panel import ControlPanel         # Import control panel widget
from .widgets.log_viewer import LogViewer               # Import log viewer widget
from .widgets.plot_widget import PlotWidget             # Import plot widget
from .training_worker import TrainingWorker             # Import training worker

class MainWindow(QMainWindow): # Main window class
    def __init__(self, parent=None): # Constructor
        super().__init__(parent)
        self.setWindowTitle("Comparative Analysis of CNN Architectures for Image Classification") # Set window title [cite: 1]
        self.setGeometry(100, 100, 1200, 800) # Set window geometry (x, y, width, height)

        # Central widget and main layout
        self.central_widget = QWidget() # Create central widget
        self.setCentralWidget(self.central_widget) # Set central widget
        self.main_layout = QHBoxLayout(self.central_widget) # Main horizontal layout

        # Create main components
        self.control_panel = ControlPanel() # Create control panel [cite: 20]
        self.log_viewer = LogViewer() # Create log viewer
        self.plot_widget = PlotWidget() # Create plot widget [cite: 19]
        self.comparison_plot_widget = PlotWidget() # Separate plot for comparisons [cite: 25]
        self.comparison_plot_widget.setWindowTitle("Model Performance Comparison")

        # Store results for comparison
        self.all_training_results = collections.OrderedDict() # Stores {model_name: {epochs:[], val_acc:[], ...}}

        # Layouting with a splitter
        left_v_layout = QVBoxLayout() # Vertical layout for left panel
        left_v_layout.addWidget(self.control_panel)
        
        self.compare_button = QPushButton("Show/Refresh Comparison Plot") # Button for comparison plot
        self.compare_button.clicked.connect(self.show_comparison_plot)
        left_v_layout.addWidget(self.compare_button)
        left_v_layout.addStretch()
        
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_v_layout)

        right_splitter = QSplitter(Qt.Vertical) # Vertical splitter for log and plot
        right_splitter.addWidget(self.log_viewer)
        right_splitter.addWidget(self.plot_widget)
        right_splitter.setSizes([300, 500]) # Initial sizes for log and plot areas

        main_splitter = QSplitter(Qt.Horizontal) # Horizontal splitter for control and (log+plot)
        main_splitter.addWidget(left_panel_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([300, 900]) # Initial sizes for left and right panels

        self.main_layout.addWidget(main_splitter) # Add main splitter to layout
        self.central_widget.setLayout(self.main_layout) # Set main layout for central widget

        # Connections
        self.control_panel.start_training_signal.connect(self.on_start_training_clicked) # Connect start training signal

        self.current_training_worker = None # To hold the current training worker

    @pyqtSlot(dict)
    def on_start_training_clicked(self, params): # Slot for start training button click
        model_name = params.get("model_name", "UnknownModel")
        self.log_viewer.clear_logs() # Clear previous logs
        self.plot_widget.reset_plot(model_name=model_name) # Reset plot for new training
        self.control_panel.set_training_active(True) # Disable controls
        self.log_viewer.append_log(f"Preparing to train {model_name}...")

        # Store current model's data for comparison plot later
        if model_name not in self.all_training_results:
            self.all_training_results[model_name] = {
                'epochs': [], 'train_acc': [], 'val_acc': [],
                'train_loss': [], 'val_loss': [], 'params': params
            }
        else: # Reset data if re-training the same model in this session
            self.all_training_results[model_name]['epochs'].clear()
            self.all_training_results[model_name]['train_acc'].clear()
            self.all_training_results[model_name]['val_acc'].clear()
            self.all_training_results[model_name]['train_loss'].clear()
            self.all_training_results[model_name]['val_loss'].clear()
            self.all_training_results[model_name]['params'] = params


        self.current_training_worker = TrainingWorker(params) # Create training worker [cite: 21]
        self.current_training_worker.log_message.connect(self.log_viewer.append_log) # Connect log signal
        self.current_training_worker.tqdm_output_signal.connect(self.log_viewer.update_tqdm_output) # Connect tqdm signal
        self.current_training_worker.epoch_metric_received.connect(self.handle_epoch_metric) # Connect metric signal
        self.current_training_worker.epoch_progress_update.connect(self.log_viewer.update_epoch_progress)
        self.current_training_worker.training_finished.connect(self.handle_training_finished) # Connect finished signal
        self.current_training_worker.training_error.connect(self.handle_training_error) # Connect error signal
        self.current_training_worker.start() # Start the training thread

    @pyqtSlot(dict)
    def handle_epoch_metric(self, metrics): # Slot to handle epoch metrics [cite: 24]
        self.plot_widget.update_plot(metrics) # Update live plot
        
        # Store data for comparison plot
        model_name = self.control_panel.model_combo.currentText() # Get current model name
        if model_name in self.all_training_results:
            self.all_training_results[model_name]['epochs'].append(metrics['epoch'])
            self.all_training_results[model_name]['train_acc'].append(metrics['train_acc'])
            self.all_training_results[model_name]['val_acc'].append(metrics['val_acc'])
            self.all_training_results[model_name]['train_loss'].append(metrics['train_loss'])
            self.all_training_results[model_name]['val_loss'].append(metrics['val_loss'])

    @pyqtSlot(str)
    def handle_training_finished(self, model_name): # Slot for training finished
        self.log_viewer.append_log(f"SUCCESS: Training for {model_name} completed.")
        QMessageBox.information(self, "Training Complete", f"Training for {model_name} has finished successfully.")
        self.control_panel.set_training_active(False) # Re-enable controls
        self.current_training_worker = None

    @pyqtSlot(str)
    def handle_training_error(self, error_message): # Slot for training error
        self.log_viewer.append_log(f"ERROR: {error_message}")
        QMessageBox.critical(self, "Training Error", error_message)
        self.control_panel.set_training_active(False) # Re-enable controls
        if self.current_training_worker: # Clean up worker if it still exists
             if self.current_training_worker.isRunning():
                  self.current_training_worker.quit()
                  self.current_training_worker.wait()
             self.current_training_worker = None


    def show_comparison_plot(self): # Method to show comparison plot [cite: 25]
        self.comparison_plot_widget.reset_plot("Model Comparison") # Reset comparison plot
        
        if not self.all_training_results:
            QMessageBox.information(self, "No Data", "No training results available to compare. Please train one or more models.")
            return

        # Create a dialog to host the comparison plot widget if it's not already visible
        # Or, just update it if it's already part of a persistent dialog/window
        # For simplicity, we'll make it a modal dialog for now
        
        dialog = QDialog(self) # Create a dialog
        dialog.setWindowTitle("Model Performance Comparison")
        layout = QVBoxLayout()
        
        # Recreate a new plot widget for the dialog to avoid issues with reparenting
        temp_comparison_plot = PlotWidget()
        temp_comparison_plot.reset_plot("Overall Model Comparison")

        for model_name, data in self.all_training_results.items(): # Iterate through results
            if data['epochs']: # Check if data exists
                epochs = data['epochs']
                # For clarity, plot only validation accuracy and validation loss on the comparison
                val_acc = data['val_acc']
                val_loss = data['val_loss']
                
                temp_comparison_plot.add_comparison_data(
                    label=f"{model_name}", 
                    epochs=epochs, 
                    train_acc=[], # Not plotting train_acc here for clarity
                    val_acc=val_acc, 
                    train_loss=[], # Not plotting train_loss here
                    val_loss=[],   # Filled by next call
                    plot_type='accuracy'
                )
                temp_comparison_plot.add_comparison_data(
                    label=f"{model_name}",
                    epochs=epochs,
                    train_acc=[],
                    val_acc=[], # Filled by previous call
                    train_loss=[],
                    val_loss=val_loss,
                    plot_type='loss'
                )
        
        layout.addWidget(temp_comparison_plot) # Add plot to dialog layout
        dialog.setLayout(layout)
        dialog.resize(800, 600) # Set dialog size
        dialog.exec_() # Show dialog modally


    def closeEvent(self, event): # Handle window close event
        if self.current_training_worker and self.current_training_worker.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit',
                                         "A training process is currently running. Are you sure you want to exit? This will stop the training.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.current_training_worker.stop_training() # Attempt to stop worker
                event.accept() # Accept close event
            else:
                event.ignore() # Ignore close event
        else:
            event.accept() # Accept close event