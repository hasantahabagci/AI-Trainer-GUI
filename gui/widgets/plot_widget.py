# Hasan Taha Bağcı
# 150210338
# Matplotlib Plotting Widget for PyQt5

from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class PlotWidget(QWidget): # Plotting widget class
    def __init__(self, parent=None): # Constructor
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 8)) # Create a Matplotlib figure
        self.canvas = FigureCanvas(self.figure) # Create a canvas for the figure
        
        # Create two subplots: one for accuracy, one for loss
        self.ax_accuracy = self.figure.add_subplot(211) # Accuracy subplot
        self.ax_loss = self.figure.add_subplot(212) # Loss subplot

        self.data = { # Dictionary to store plot data
            'train_acc': [], 'val_acc': [],
            'train_loss': [], 'val_loss': [],
            'epochs': []
        }
        
        self._initial_plot_setup() # Initial plot setup

        layout = QVBoxLayout() # Vertical layout
        layout.addWidget(self.canvas) # Add canvas to layout
        self.setLayout(layout) # Set layout

    def _initial_plot_setup(self): # Setup initial plot appearance
        self.ax_accuracy.set_title("Model Accuracy")
        self.ax_accuracy.set_xlabel("Epoch")
        self.ax_accuracy.set_ylabel("Accuracy (%)")
        self.ax_accuracy.grid(True)
        self.line_train_acc, = self.ax_accuracy.plot([], [], 'r-o', label='Training Accuracy')
        self.line_val_acc, = self.ax_accuracy.plot([], [], 'b-o', label='Validation Accuracy')
        self.ax_accuracy.legend(loc='lower right')

        self.ax_loss.set_title("Model Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)
        self.line_train_loss, = self.ax_loss.plot([], [], 'r-o', label='Training Loss')
        self.line_val_loss, = self.ax_loss.plot([], [], 'b-o', label='Validation Loss')
        self.ax_loss.legend(loc='upper right')
        
        self.figure.tight_layout() # Adjust layout to prevent overlap
        self.canvas.draw() # Draw the canvas

    def update_plot(self, epoch_data): # Update plot with new data from an epoch [cite: 24]
        """
        epoch_data: a dictionary like {'epoch': e, 'train_loss': tl, 'val_loss': vl, 'train_acc': ta, 'val_acc': va}
        """
        self.data['epochs'].append(epoch_data['epoch'])
        self.data['train_acc'].append(epoch_data['train_acc'])
        self.data['val_acc'].append(epoch_data['val_acc'])
        self.data['train_loss'].append(epoch_data['train_loss'])
        self.data['val_loss'].append(epoch_data['val_loss'])

        # Update accuracy plot
        self.line_train_acc.set_data(self.data['epochs'], self.data['train_acc'])
        self.line_val_acc.set_data(self.data['epochs'], self.data['val_acc'])
        self.ax_accuracy.relim() # Recompute the limits
        self.ax_accuracy.autoscale_view(True,True,True) # Autoscale

        # Update loss plot
        self.line_train_loss.set_data(self.data['epochs'], self.data['train_loss'])
        self.line_val_loss.set_data(self.data['epochs'], self.data['val_loss'])
        self.ax_loss.relim() # Recompute the limits
        self.ax_loss.autoscale_view(True,True,True) # Autoscale
        
        self.figure.tight_layout()
        self.canvas.draw() # Redraw the canvas

    def reset_plot(self, model_name=""): # Reset plot for a new training session
        self.data = {
            'train_acc': [], 'val_acc': [],
            'train_loss': [], 'val_loss': [],
            'epochs': []
        }
        
        self.ax_accuracy.set_title(f"{model_name} Accuracy" if model_name else "Model Accuracy")
        self.ax_loss.set_title(f"{model_name} Loss" if model_name else "Model Loss")

        self.line_train_acc.set_data([], [])
        self.line_val_acc.set_data([], [])
        self.line_train_loss.set_data([], [])
        self.line_val_loss.set_data([], [])

        # Reset limits
        self.ax_accuracy.relim()
        self.ax_accuracy.autoscale_view(True,True,True)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view(True,True,True)

        self.figure.tight_layout()
        self.canvas.draw()

    def add_comparison_data(self, label, epochs, train_acc, val_acc, train_loss, val_loss, plot_type='accuracy'): # For model comparison [cite: 25]
        """Adds a new line to the specified plot for comparison."""
        if plot_type == 'accuracy':
            ax = self.ax_accuracy
            ax.plot(epochs, val_acc, marker='o', linestyle='--', label=f'{label} Val Acc')
            # ax.plot(epochs, train_acc, marker='x', linestyle=':', label=f'{label} Train Acc')
        elif plot_type == 'loss':
            ax = self.ax_loss
            ax.plot(epochs, val_loss, marker='o', linestyle='--', label=f'{label} Val Loss')
            # ax.plot(epochs, train_loss, marker='x', linestyle=':', label=f'{label} Train Loss')
        else:
            return

        ax.legend(loc='best') # Update legend
        ax.relim()
        ax.autoscale_view(True,True,True)
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__': # Example Usage
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    main_plot = PlotWidget()
    main_plot.setWindowTitle("Plot Test")
    main_plot.show()
    # Example update
    main_plot.update_plot({'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.4, 'train_acc': 70, 'val_acc': 75})
    main_plot.update_plot({'epoch': 2, 'train_loss': 0.3, 'val_loss': 0.25, 'train_acc': 80, 'val_acc': 85})
    sys.exit(app.exec_())