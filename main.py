# Hasan Taha Bağcı
# 150210338
# Main Application Runner

import sys
from PyQt5.QtWidgets import QApplication
# Need to ensure gui package is discoverable if running from project root
# If main_app.py is in the root directory 'comparative_cnn_analyzer':
from gui.main_window import MainWindow 

def main(): # Main function
    app = QApplication(sys.argv) # Create application instance
    app.setStyle('Fusion') # Optional: set a style

    main_window = MainWindow() # Create main window instance
    main_window.show() # Show main window

    sys.exit(app.exec_()) # Start application event loop

if __name__ == '__main__':
    # This structure assumes you run from the 'comparative_cnn_analyzer' directory
    # e.g., python main_app.py
    # If 'gui' is not found, you might need to adjust sys.path or how you import MainWindow
    # One way is to ensure the project root is in PYTHONPATH or run as a module if packaged.
    
    # Simple check to ensure PWD is project root for easier module discovery
    # import os
    # print(f"Current working directory: {os.getcwd()}")
    # print(f"Python path: {sys.path}")
    main()