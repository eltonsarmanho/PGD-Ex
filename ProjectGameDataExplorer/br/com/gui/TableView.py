    
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import sys


 
class TableView(QTableWidget):
    def __init__(self, data, *args):
        QTableWidget.__init__(self, *args)
        self.data = data
        self.setWindowTitle("EDA Metrics")
        self.setData()
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        #self.show()
 
    def setData(self): 
        horHeaders = []
        for n, key in enumerate(sorted(self.data.keys())):
            horHeaders.append(key)
            for m, item in enumerate(self.data[key]):
                newitem = QTableWidgetItem(item)
                self.setItem(m, n, newitem)
        self.setHorizontalHeaderLabels(horHeaders)
 
def main(args):
    data ={'Metrics':['Mean','Median','Max','Var','Std_dev','Kurtosis','skewness'],
           'Value': [str(1),str(2),str(3),
                    str(4),str(5),str(6),str(7)]}
    #data = {'Metrics':['1','2','3','4'],
    #    'Value':['1','2','1','3']}
    app = QApplication(args)
    table = TableView(data, 7, 2)
    table.show()
    sys.exit(app.exec_())
 
if __name__=="__main__":
    main(sys.argv)




