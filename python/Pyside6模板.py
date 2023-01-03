# This Python file uses the following encoding: utf-8
import sys

from ui import Ui_MainWindow

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QPushButton, QHBoxLayout, QMessageBox,
    QTableWidgetItem
)

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    ui = window.ui

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()