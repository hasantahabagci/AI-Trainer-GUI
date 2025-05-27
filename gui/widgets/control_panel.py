# Hasan Taha Bağcı
# 150210338
# Control Panel Widget for Model Selection and Training Parameters

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QFormLayout, 
                             QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
                             QCheckBox, QLabel)
from PyQt5.QtCore import pyqtSignal
import torch # Import torch to check for MPS/CUDA availability for default selection

from models import MODEL_FACTORIES # Import model factories

class ControlPanel(QWidget): # Control panel class
    start_training_signal = pyqtSignal(dict) # Signal emitted when training starts

    def __init__(self, parent=None): # Constructor
        super().__init__(parent)
        
        main_layout = QVBoxLayout() # Main vertical layout
        self.setLayout(main_layout)

        # Model Selection Group
        model_group = QGroupBox("Model Configuration") # Group box for model selection
        model_layout = QFormLayout() # Form layout for model parameters

        self.model_combo = QComboBox() # Combo box for model selection
        self.model_combo.addItems(MODEL_FACTORIES.keys()) # Add model names to combo box
        model_layout.addRow("Select Model:", self.model_combo)

        self.use_pretrained_checkbox = QCheckBox("Use Pretrained Weights (if available)") # Checkbox for pretrained weights
        self.use_pretrained_checkbox.setChecked(True) # Default to checked
        model_layout.addRow(self.use_pretrained_checkbox)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)

        # Training Parameters Group
        params_group = QGroupBox("Training Parameters") # Group box for training parameters
        params_layout = QFormLayout() # Form layout for parameters

        self.epochs_spinbox = QSpinBox() # Spin box for epochs
        self.epochs_spinbox.setRange(1, 200) # Set range for epochs
        self.epochs_spinbox.setValue(10) # Default value for epochs
        params_layout.addRow("Epochs:", self.epochs_spinbox)

        self.batch_size_spinbox = QSpinBox() # Spin box for batch size
        self.batch_size_spinbox.setRange(4, 512) # Set range for batch size
        self.batch_size_spinbox.setStepType(QSpinBox.AdaptiveDecimalStepType) # Allow steps like 4, 8, 16, ...
        self.batch_size_spinbox.setValue(64) # Default value for batch size
        params_layout.addRow("Batch Size:", self.batch_size_spinbox)

        self.lr_spinbox = QDoubleSpinBox() # Double spin box for learning rate
        self.lr_spinbox.setRange(0.00001, 0.1) # Set range for learning rate
        self.lr_spinbox.setSingleStep(0.0001) # Set step for learning rate
        self.lr_spinbox.setDecimals(5) # Set number of decimals
        self.lr_spinbox.setValue(0.001) # Default value for learning rate
        params_layout.addRow("Learning Rate:", self.lr_spinbox)

        self.data_aug_checkbox = QCheckBox("Enable Data Augmentation") # Checkbox for data augmentation
        self.data_aug_checkbox.setChecked(True) # Default to checked
        params_layout.addRow(self.data_aug_checkbox)
        
        self.device_combo = QComboBox() # Combo box for device selection
        # Always list all three options
        device_options = ["cpu", "cuda", "mps"] 
        self.device_combo.addItems(device_options)
        
        # Set a sensible default based on availability
        default_device = "cpu"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check for MPS
            default_device = "mps"
        if torch.cuda.is_available(): # Check for CUDA (CUDA preferred over MPS if both, for broader hardware support)
             # If you prefer MPS to be default on Mac even if an older CUDA GPU is present, swap these checks
            default_device = "cuda" 
        
        # If MPS is available and no CUDA, MPS becomes default (handled by order above if CUDA is checked first)
        # This logic can be simplified:
        if torch.cuda.is_available():
            self.device_combo.setCurrentText("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device_combo.setCurrentText("mps")
        else:
            self.device_combo.setCurrentText("cpu")

        params_layout.addRow("Device:", self.device_combo)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Action Button
        self.start_button = QPushButton("Start Training") # Button to start training
        self.start_button.clicked.connect(self._on_start_training) # Connect button click to handler
        main_layout.addWidget(self.start_button)

        main_layout.addStretch() # Add stretch to push elements to the top

    def _on_start_training(self): # Handler for start training button click
        params = { # Gather parameters from UI elements
            "model_name": self.model_combo.currentText(),
            "use_pretrained": self.use_pretrained_checkbox.isChecked(),
            "epochs": self.epochs_spinbox.value(),
            "batch_size": self.batch_size_spinbox.value(),
            "learning_rate": self.lr_spinbox.value(),
            "use_data_augmentation": self.data_aug_checkbox.isChecked(),
            "device": self.device_combo.currentText()
        }
        self.start_training_signal.emit(params) # Emit signal with parameters

    def set_training_active(self, active): # Enable/disable controls during training
        self.model_combo.setEnabled(not active)
        self.use_pretrained_checkbox.setEnabled(not active)
        self.epochs_spinbox.setEnabled(not active)
        self.batch_size_spinbox.setEnabled(not active)
        self.lr_spinbox.setEnabled(not active)
        self.data_aug_checkbox.setEnabled(not active)
        self.device_combo.setEnabled(not active)
        self.start_button.setEnabled(not active)
        self.start_button.setText("Training in Progress..." if active else "Start Training")

if __name__ == '__main__': # Example Usage
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    panel = ControlPanel()
    panel.setWindowTitle("Control Panel Test")
    panel.start_training_signal.connect(lambda p: print(f"Start training with: {p}"))
    panel.show()
    sys.exit(app.exec_())