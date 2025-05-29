# Hasan Taha Bağcı 150210338
# main.py - Main entry point for the CNN Analyzer application

import sys
import os
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from gui.main_window import MainWindow

def start_application():
    app = QApplication(sys.argv)
    

    try:
        app.setStyle("Fusion")
    except Exception as e:
        print(f"Could not set Fusion style: {e}. Using default.")


    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    start_application()
