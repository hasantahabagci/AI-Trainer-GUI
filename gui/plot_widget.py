# Hasan Taha Bağcı 150210338
# gui/plot_widget.py

import matplotlib
matplotlib.use('Qt5Agg') # Set the backend for PyQt5 compatibility
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout
import numpy as np

class PlotWidget(QWidget): # A custom widget to display Matplotlib plots
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super(PlotWidget, self).__init__(parent)
        
        self.fig = Figure(figsize=(width, height), dpi=dpi) # Create a Matplotlib Figure
        self.canvas = FigureCanvas(self.fig) # Create a FigureCanvas to render the Figure
        
        # Create subplots for loss and accuracy
        self.ax_loss = self.fig.add_subplot(2, 1, 1) 
        self.ax_acc = self.fig.add_subplot(2, 1, 2)  
        
        self.init_plots() # Initialize plot appearance

        layout = QVBoxLayout() # Use a QVBoxLayout to hold the canvas
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        # Data storage for plotting
        self.epochs_data = []
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []

    def init_plots(self): # Initialize or reset the plots
        self.ax_loss.clear()
        self.ax_loss.set_title('Training and Validation Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True) # Add a grid for better readability
        self.loss_lines = { # Store line objects for updating
            'train': self.ax_loss.plot([], [], 'r-o', label='Train Loss')[0], # Red line with circles for train loss
            'val': self.ax_loss.plot([], [], 'b-s', label='Validation Loss')[0] # Blue line with squares for val loss
        }
        self.ax_loss.legend()

        self.ax_acc.clear()
        self.ax_acc.set_title('Training and Validation Accuracy')
        self.ax_acc.set_xlabel('Epoch')
        self.ax_acc.set_ylabel('Accuracy')
        self.ax_acc.grid(True)
        self.acc_lines = { # Store line objects for updating
            'train': self.ax_acc.plot([], [], 'r-o', label='Train Accuracy')[0],
            'val': self.ax_acc.plot([], [], 'b-s', label='Validation Accuracy')[0]
        }
        self.ax_acc.legend()
        
        self.fig.tight_layout() 
        self.canvas.draw() 

    def reset_plot_data(self): 
        self.epochs_data = []
        self.train_loss_data = []
        self.val_loss_data = []
        self.train_acc_data = []
        self.val_acc_data = []
        self.init_plots() # Re-initialize plots to clear lines

    def update_plot(self, epoch, train_loss, val_loss, train_acc, val_acc): 
        self.epochs_data.append(epoch)
        self.train_loss_data.append(train_loss)
        self.val_loss_data.append(val_loss)
        self.train_acc_data.append(train_acc)
        self.val_acc_data.append(val_acc)

        # Update loss plot
        self.loss_lines['train'].set_data(self.epochs_data, self.train_loss_data)
        self.loss_lines['val'].set_data(self.epochs_data, self.val_loss_data)
        self.ax_loss.relim() 
        self.ax_loss.autoscale_view(True, True, True) # Autoscale axes

        # Update accuracy plot
        self.acc_lines['train'].set_data(self.epochs_data, self.train_acc_data)
        self.acc_lines['val'].set_data(self.epochs_data, self.val_acc_data)
        self.ax_acc.relim()
        self.ax_acc.autoscale_view(True, True, True)

        self.fig.tight_layout() 
        self.canvas.draw() 

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication, QMainWindow
    import sys
    import time

    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("PlotWidget Test")
            self.plot_widget = PlotWidget(self)
            self.setCentralWidget(self.plot_widget)
            self.setGeometry(100, 100, 800, 600)
            self.test_update()

        def test_update(self): # Simulate receiving data over time
            self.plot_widget.reset_plot_data() # Start with a clean plot
            
            # Simulate a few epochs of data
            simulated_epochs = 10
            for i in range(1, simulated_epochs + 1):
                # Simulate some data (replace with actual data in your app)
                dummy_train_loss = np.random.rand() * 0.5 + 0.1 / i
                dummy_val_loss = np.random.rand() * 0.4 + 0.15 / i
                dummy_train_acc = 0.5 + np.random.rand() * 0.1 + i * 0.03
                dummy_val_acc = 0.55 + np.random.rand() * 0.08 + i * 0.025
                
                # Ensure accuracy doesn't exceed 1.0
                dummy_train_acc = min(dummy_train_acc, 0.95 + np.random.rand()*0.04)
                dummy_val_acc = min(dummy_val_acc, 0.93 + np.random.rand()*0.04)

                self.plot_widget.update_plot(
                    epoch=i,
                    train_loss=dummy_train_loss,
                    val_loss=dummy_val_loss,
                    train_acc=dummy_train_acc,
                    val_acc=dummy_val_acc
                )
                QApplication.processEvents()
                time.sleep(0.5) 
            
            print("Test update finished.")

    app = QApplication(sys.argv)
    main_win = TestWindow()
    main_win.show()
    sys.exit(app.exec_())
