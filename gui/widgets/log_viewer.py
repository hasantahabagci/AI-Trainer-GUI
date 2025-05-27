# Hasan Taha Bağcı
# 150210338
# Log Viewer and Progress Bar Widget

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QProgressBar, QLabel
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

class LogViewer(QWidget): # Log viewer class
    def __init__(self, parent=None): # Constructor
        super().__init__(parent)
        
        layout = QVBoxLayout() # Vertical layout
        self.log_output_label = QLabel("Training Log & Progress:") # Label for log output
        self.log_output = QTextEdit() # Text edit for logs
        self.log_output.setReadOnly(True) # Make log output read-only
        
        self.epoch_progress_label = QLabel("Epoch Progress: 0/0") # Label for epoch progress
        self.epoch_progress_bar = QProgressBar() # Progress bar for epochs
        self.epoch_progress_bar.setValue(0) # Initialize progress bar value
        self.epoch_progress_bar.setTextVisible(True) # Show text on progress bar

        self.batch_progress_label = QLabel("Batch Progress:") # Label for batch progress (from tqdm)
        # self.batch_progress_bar = QProgressBar() # Alternative finer-grained progress bar (optional)
        # self.batch_progress_bar.setValue(0)
        # self.batch_progress_bar.setTextVisible(True)

        layout.addWidget(self.log_output_label)
        layout.addWidget(self.log_output)
        layout.addWidget(self.epoch_progress_label)
        layout.addWidget(self.epoch_progress_bar)
        layout.addWidget(self.batch_progress_label)
        # layout.addWidget(self.batch_progress_bar)
        self.setLayout(layout) # Set layout

    @pyqtSlot(str)
    def append_log(self, message): # Slot to append log messages [cite: 18]
        self.log_output.append(message) # Append message to log output
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum()) # Auto-scroll

    @pyqtSlot(str)
    def update_tqdm_output(self, line): # Slot to display tqdm output directly
        # For tqdm, we might just append it or try to parse it if a specific format is used.
        # For simplicity, we append the relevant part of the tqdm line.
        if "\r" in line: # Handle carriage returns from tqdm
            current_text = self.log_output.toPlainText()
            last_line_break = current_text.rfind('\n')
            if last_line_break != -1 and current_text[last_line_break:].startswith("Epoch "): # Crude check if last line is a tqdm line
                self.log_output.setPlainText(current_text[:last_line_break+1] + line.strip())
            else:
                self.log_output.append(line.strip())
        else:
             self.log_output.append(line.strip()) # Append stripped line to log output
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())


    @pyqtSlot(int, int)
    def update_epoch_progress(self, current_epoch, total_epochs): # Slot to update epoch progress
        self.epoch_progress_label.setText(f"Epoch Progress: {current_epoch}/{total_epochs}")
        if total_epochs > 0:
            progress_percent = int((current_epoch / total_epochs) * 100) if current_epoch <= total_epochs else 100
            self.epoch_progress_bar.setValue(progress_percent) # Set progress bar value
            self.epoch_progress_bar.setFormat(f"{current_epoch}/{total_epochs} (%p%)")
        else:
            self.epoch_progress_bar.setValue(0)
            self.epoch_progress_bar.setFormat(f"0/0 (%p%)")
            
    # @pyqtSlot(int, int) # If implementing batch progress bar
    # def update_batch_progress(self, current_batch, total_batches):
    #     if total_batches > 0:
    #         progress_percent = int((current_batch / total_batches) * 100)
    #         self.batch_progress_bar.setValue(progress_percent)
    #     else:
    #         self.batch_progress_bar.setValue(0)

    def clear_logs(self): # Clear logs and reset progress
        self.log_output.clear()
        self.update_epoch_progress(0, 0)
        # self.update_batch_progress(0,0)
        self.batch_progress_label.setText("Batch Progress:")

if __name__ == '__main__': # Example Usage
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    viewer = LogViewer()
    viewer.setWindowTitle("Log Viewer Test")
    viewer.show()
    viewer.append_log("This is a test log message.")
    viewer.append_log("Another message.")
    viewer.update_epoch_progress(1, 10)
    viewer.update_tqdm_output("Epoch 1/10 [Train]: 10%|█         | 10/100 [00:01<00:09,  9.90it/s, Loss: 1.23, Acc: 0.55]")
    sys.exit(app.exec_())