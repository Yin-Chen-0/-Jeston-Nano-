from PyQt5 import QtWidgets # import PyQt5 widgets
from PyQt5.QtWidgets import QWidget

from MainWindow import Ui_mainWindow
import sys


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
