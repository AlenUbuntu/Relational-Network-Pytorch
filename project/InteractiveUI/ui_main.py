import sys
from InteractiveUI.interactiveUI import Ui_Dialog
from PyQt5.QtWidgets import QApplication, QMainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    path = '/home/alan/Dropbox/UTD Course/2018 Spring/Relational Statistical AI/' \
           'project/DataGenerator/datasets/Sort-of-CLEVR_default'
    ui = Ui_Dialog(path)
    ui.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())