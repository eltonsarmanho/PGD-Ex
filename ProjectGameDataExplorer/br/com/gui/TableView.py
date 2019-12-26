    
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QIcon, QAbstractItemView
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget, QTableWidgetItem, QVBoxLayout
import sys


class TableView(QTableWidget):

    def __init__(self, data, title, *args):
        QTableWidget.__init__(self, *args)
        self.data = data
        self.setWindowTitle(title)
        self.setData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

        # self#.show()
    def setModeMultiple(self):
        self.setSelectionMode(QAbstractItemView.MultiSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)

    def setData(self): 
        horHeaders = []
        item1 = QTableWidgetItem()
        item1.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEditable | 
              Qt.ItemIsEnabled)
        for n, key in enumerate(sorted(self.data.keys(), reverse=True)):
        # for n, key in enumerate((self.data.keys())):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                newitem = QTableWidgetItem(item)
                newitem.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)

 
def main(args):
    data = {'Metrics':['Mean', 'Median', 'Max', 'Var', 'Std_dev', 'Kurtosis', 'skewness'],
           'Value': [str(1), str(2), str(3),
                    str(4), str(5), str(6), str(7)]}
    # data = {'Metrics':['1','2','3','4'],
    #    'Value':['1','2','1','3']}
    app = QApplication(args)
    table = TableView(data, 7, 2)
    table.show()
    sys.exit(app.exec_())

 
if __name__ == "__main__":
    main(sys.argv)

