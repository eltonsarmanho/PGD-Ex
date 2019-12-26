'''
Created on 24 de abr de 2019

@author: eltonss
'''
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QDate, QDateTime, Qt, QTime
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QBrush, QColor, QSlider
from PyQt5.QtWidgets import QWidget, QMessageBox, QSplitter
from _datetime import datetime
from pyqtgraph.Qt import PYQT5
from pyqtgraph.Qt import QtCore, QtGui, QT_LIB


class TimeWidget(QtGui.QTimeEdit):
    '''
    classdocs
    '''

    def __init__(self, params):
        super(TimeWidget, self).__init__()
        '''
        Constructor
        '''

    def stepBy(self, steps):    
      
        if (self.time().minute() == 59 and steps > 0):
            self.setTime(QTime(self.time().hour() + 1, 0, self.time().second(), self.time().msec()));
        elif (self.time().minute() == 0 and steps < 0):
            self.setTime(QTime(self.time().hour() - 1, 59, self.time().second(), self.time().msec()));
        elif (self.time().second() == 59 and steps > 0):
            self.setTime(QTime(self.time().hour(), self.time().minute() + 1, 0, self.time().msec()));
        elif (self.time().second() == 0 and steps < 0):
            self.setTime(QTime(self.time().hour(), self.time().minute() - 1, 59, self.time().msec()));
        else:  super(TimeWidget, self).stepBy(steps);
        
