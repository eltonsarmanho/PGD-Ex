
'''
Created on 15 de abr de 2019

@author: eltonss
'''

import matplotlib

from PyQt5.QtCore import  Qt, QUrl,QTime
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMessageBox,QWidget,QSplitter,QStyle, QCheckBox,QListWidget
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtCore import pyqtSlot,Qt
from PyQt5 import QtCore, QtGui, QtWidgets

from datetime import datetime
from pyqtgraph import AxisItem
import sys, os
import math
import numpy
from PyQt5.QtGui import QIcon,QAbstractItemView

import scipy.signal as scisig
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pyqtgraph as pg
from datetime import datetime, timedelta
from time import mktime
import peakutils
import biosppy
from builtins import int
import glob

matplotlib.use('Agg')

__all__ = ['QRangeSlider']
    
DEFAULT_CSS = """
    QRangeSlider * {
        border: 0px;
        padding: 0px;
    }
    QRangeSlider #Head {
        background: #222;
    }
    QRangeSlider #Span {
        background: #393;
    }
    QRangeSlider #Span:active {
        background: #282;
    }
    QRangeSlider #Tail {
        background: #222;
    }
    QRangeSlider > QSplitter::handle {
        background: #797D7F;
    }
    QRangeSlider > QSplitter::handle:vertical {
        height: 0px;
    }
    QRangeSlider > QSplitter::handle:pressed {
        background: #ca5;
    }
    """

class FlowChartGame(QtGui.QMainWindow):
    
    _TIME_TAG_END = 0;
    _TIME_TAG_INITIAL = 0;
    _DURATION_SESSION = 0;
    _POSITION_INITIAL_SESSION = 0;
    
    _FILE_BVP = "";
    _FILE_EDA = "" ;
    _FILE_EMOTION = "";
    _FILE_PATH_SESSION = "";
    _VIDEO_SC = ""
    _VIDEO_EXTRA = ""
    _TAG_FILE_VIDEO =""
    _TAG_FILE = ""
    _EVALUATOR_NAME =""
    _NUMBER_PARTICIPANT = ""
    _NUMBER_SESSION = ""
    _DISCONNECT_RANGE_SLIDER = False;
    _ANNOTATION = False
    
    _LIST_EMOTION = ['Happiness', 'Sadness', 'Anger',
                      'Fear', 'Surprise', 'Disgust']
    
    def __init__(self, buffer_size=0, data_buffer=[], graph_title="", parent=None):
        super(FlowChartGame, self).__init__(parent)               

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self._SD = self.SourceData()

        self.createMediaPlayer();
        self.windowPlots()
        
    def createMediaPlayer(self):
       
        self._ANNOTATION = False;
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer2 = QMediaPlayer(None, QMediaPlayer.VideoSurface)       
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)       
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.error.connect(self.handleError)
        try:
            self.splitter.setSizes([1000, 200])
        except: pass;
        
    def destroyMediaPlayer(self):
        self.mediaPlayer.stop()
        self.mediaPlayer2.stop()
        btPlayer.setEnabled(False)    
        self.splitter.setSizes([1000, 0])    
    
    def windowPlots(self):
        
        self.mainbox = QtGui.QWidget()
        self.mainbox.setGeometry(0, 0, 600, 800)
        self.setCentralWidget(self.mainbox)
        
        layout = QtGui.QGridLayout()
        self.mainbox.setLayout(layout)        
        self.canvas = pg.GraphicsLayoutWidget(border=(100, 100, 100))        
       
        self.mainbox.layout().addLayout(self.uiMainPanel(), 0, 0, 1, 1)
        self.MenuBar();
      
        
    def MenuBar(self):
        
        layout = QtGui.QHBoxLayout()
        bar = self.menuBar()
        file = bar.addMenu("File")
        tools = bar.addMenu("Tools")
                
        
        resetPB = QtGui.QAction("Reset Progress Bar Timer", self)
        resetPB.setShortcut("Ctrl+T")
        
        metricEDA = QtGui.QAction("EDA Metrics", self)
        timeIntervals = QtGui.QAction("Time Intervals(Session)", self)
        emotionalComponents = QtGui.QAction("Emotional Components", self)
        
        open = QtGui.QAction("Open E4 Data File with Video", self)
        open.setShortcut("Ctrl+O")
        annotation = QtGui.QAction("Annotation", self)
        annotation.setShortcut("Ctrl+J")

        file.addAction(open)
        # file.addAction(openSV)
                    
        quit = QtGui.QAction("Quit", self)
        quit.setShortcut("Ctrl+Q") 
        
        restart = QtGui.QAction("Restart", self)
        restart.setShortcut("Ctrl+R") 
        
        file.addAction(restart)
        file.addAction(quit)
        
        tools.addAction(resetPB)
        tools.addAction(metricEDA)
        tools.addAction(emotionalComponents) 
        tools.addAction(timeIntervals)
        tools.addAction(annotation)


        file.triggered[QtGui.QAction].connect(self.processtrigger)
        tools.triggered[QtGui.QAction].connect(self.processTools)

        self.setLayout(layout)
    
    def processTools(self, q):
        if(q.text() == "EDA Metrics"):
           
            data = {'Metrics':['Mean', 'Median', 'Max', 'Var', 'Std_dev', 'Kurtosis', 'skewness'],
           'Value': [str(metricsEDA['mean']), str(metricsEDA['median']), str(metricsEDA['max']),
                    str(metricsEDA['var']), str(metricsEDA['std_dev']),
                    str(metricsEDA['kurtosis']), str(metricsEDA['skewness'])]}
          
            self.tv = self.TableView(data, "EDA Metrics", 7, 2)
            
            self.tv.show()
        elif(q.text() == "Emotional Components"):
            _list = ['Happiness', 'Sadness', 'Anger',
                      'Fear', 'Surprise', 'Disgust']
            data = {'Emotional components': _list}
            
            self.tv = self.TableView(data, "Time Intervals", 6, 1)            
            
            self.tv.setModeMultiple()
            self.tv.resizeColumnsToContents()
            self.tv.horizontalHeader().setSectionResizeMode(QtGui.QHeaderView.Stretch)                
            self.win = QWidget()
            
            def getSelectedInterval():
                indexes = self.tv.selectionModel().selectedRows()
                listSelected = []
                for index in sorted(indexes):
                    print("index %s Emotion %s" % (index.row(), _list[index.row()])) 
                    listSelected.append(_list[index.row()])
                self._LIST_EMOTION = listSelected;
                self.workloadPlot()
                destroyTable()
            def destroyTable():
                self.win.destroy()
               
            btSubmit = QtGui.QPushButton("Select") 
            btCancel = QtGui.QPushButton("Cancel");
            btSubmit.clicked.connect(getSelectedInterval)
            btCancel.clicked.connect(destroyTable)
            vbox = QtGui.QVBoxLayout()
            vbox.addWidget(self.tv)
            hbox = QtGui.QVBoxLayout()
            hbox.addWidget(btSubmit)
            hbox.addWidget(btCancel)
            vbox.addLayout(hbox)
            
            
            self.win.setLayout(vbox)
             
            self.win.setWindowTitle("PyQt")
            self.win.adjustSize()
            fg = self.frameGeometry()
            cp = QtGui.QDesktopWidget().availableGeometry().center()
            fg.moveCenter(cp)
            self.win.move(fg.topLeft())
            self.win.show()            
          
        elif(q.text() == "Time Intervals(Session)"):
                
            ut = self.UnixTime();
            timeLeft = []
            timeRight = []
            for (t1, t2) in zip(arrayTagsInitial, arrayTagsEnd):                    
                timeLeft.append(ut.time_inc(t1, 0).strftime('%H:%M:%S'))
                timeRight.append(ut.time_reduce(t2, 0).strftime('%H:%M:%S'))    
                
            data = {'Initial Time':timeLeft, 'End Time': timeRight}
          
            self.tv = self.TableView(data, "Time Intervals", len(arrayTagsInitial), 2)            
            self.tv.resizeColumnsToContents()
            self.tv.horizontalHeader().setSectionResizeMode(QtGui.QHeaderView.Stretch)                


            self.win = QWidget()

            def getSelectedInterval():
                indexes = self.tv.selectionModel().selectedRows()
                for index in sorted(indexes):

                    self.setConfigureTimeInterval(arrayTagsInitial[index.row()],
                                                  arrayTagsEnd[index.row()])  
                
                self.workloadPlot()
                self._DISCONNECT_RANGE_SLIDER = True;
                self.updateRangerSlider()
                self.setPositionInPlayer(self._POSITION_INITIAL_SESSION)
                destroyTable()
            def destroyTable():
                self.win.destroy()


               
            btSubmit = QtGui.QPushButton("Select") 
            btCancel = QtGui.QPushButton("Cancel");
            btSubmit.clicked.connect(getSelectedInterval)
            btCancel.clicked.connect(destroyTable)
            vbox = QtGui.QVBoxLayout()
            vbox.addWidget(self.tv)
            hbox = QtGui.QVBoxLayout()
            hbox.addWidget(btSubmit)
            hbox.addWidget(btCancel)
            vbox.addLayout(hbox)
            
            
            self.win.setLayout(vbox)
             
            self.win.setWindowTitle("PyQt")
            self.win.adjustSize()
            fg = self.frameGeometry()
            cp = QtGui.QDesktopWidget().availableGeometry().center()
            fg.moveCenter(cp)
            self.win.move(fg.topLeft())
            self.win.show()                
            
                
        elif(q.text() == "Reset Progress Bar Timer"):
            self.reset()
        elif (q.text() == 'Annotation'):
            try:
                self._ANNOTATION = True  
                  
                self.openButtonsEmotions()
                self.openButtonsActions()
                            
                self.getWindowsAnnotation()
                self.splitter.setSizes([0, 1])
                self.uiMainPanelAnnotation()
            except: 
                print("Oops!", sys.exc_info()[0], "occured.")
                print("Erro in Annotation")
             
        pass;  
   
    def reset (self):
        self.mediaPlayer.pause()
        self.mediaPlayer2.pause()  
        self.durationChanged(0)
        self.updateRangerSlider()
        self.clearLinearRegion()
    def processtrigger(self, q):
        
        if(q.text() == 'Open E4 Data File with Video'):

            self.loadingVisualization()       
        elif (q.text() == "Quit"):
            sys.exit(app.exec_())
        elif (q.text() == "Restart"):
            os.execl(sys.executable, sys.executable, *sys.argv) 
    
    def getWindowsAnnotation(self):
        self.currentWindow = 0
        self.totalWindows = math.ceil((float(self._TIME_TAG_END) - float(self._TIME_TAG_INITIAL))/10)
        self.windowEffects = []

        for i in range(self.totalWindows):
            self.windowEffects.append("none-none")
    
    def loadingVisualization(self):
        if(self.workload()):
            self.updateRangerSlider()            
            

    def workload(self):
               
        dlg = QtGui.QFileDialog()       
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        filenames = []
        try:
        
            if dlg.exec_():
                filenames = dlg.selectedFiles()               
            else:
                QMessageBox.information(self, "Message", "No appropriate file Located or no file selected"); 
                return False

            for file in filenames:
                filename = os.path.basename(file)               
                self._FILE_PATH_SESSION=os.path.dirname(file)
                
                if(not(self.is_video_file(filename) or filename.endswith('.csv'))):
                    QMessageBox.information(self, "Message", "No appropriate file Located");
                    return False;
                
                if(self.is_video_file(filename) and ("SC") in filename):
                    _VIDEO_SC = file;
                elif(self.is_video_file(filename) and (("WC") in filename or ("HV") in filename)):
                    _VIDEO_EXTRA = file;   
                elif(filename == 'timevideo.csv'):                    
                    _TAG_FILE_VIDEO = file;
                elif(filename == 'tags.csv'):                    
                    _TAG_FILE = file;
                elif filename == 'BVP.csv':
                    self._FILE_BVP = file;                
                elif filename == 'EDA.csv':
                    self._FILE_EDA = file; 
                elif ("EMOCAO") in filename:
                    self._FILE_EMOTION = file;            
            

            if _TAG_FILE_VIDEO:
                self.setTagVideo(_TAG_FILE_VIDEO);
            else: 
                QMessageBox.information(self, "Message", "Tag Video not selected");
                return False;
                    
            if _TAG_FILE:                
                self.setTags(_TAG_FILE)
            else: 
                QMessageBox.information(self, "Message", "Tag File not selected");
                return False
            
            if _VIDEO_SC:
                self.openFile(_VIDEO_SC)
            else: 
                QMessageBox.information(self, "Message", "Error in Loading Media: Screen Capture video not found");
                return False;
            
            if _VIDEO_EXTRA:
                self.openFile(_VIDEO_EXTRA)
            else: 
                QMessageBox.information(self, "Message", "Error Loading Media: Face or hand's video Not Found");
                return False;
            self.workloadPlot()
            #if(self._FILE_BVP or self._FILE_EDA or self._FILE_EMOTION):
            #   QMessageBox.information(self, "Message", "The files were loaded successfully");
            #    
            #else: 
            #    QMessageBox.information(self, "Message", "The Empatica E4 output or Emotion file not found");
            #    return False;


            return True;
    
        except IOError:
            print('An error occurred trying to read the file.')
            
        except ValueError:
            print('Non-numeric data found in the file.')
        
        except ImportError:
            print ("NO module found")
            
        except EOFError:
            print('Why did you do an EOF on me?')
        
        except KeyboardInterrupt:
            print('You cancelled the operation.')
        
        except:            
            QMessageBox.information(self, "Message", "An error occurred")
  
    def workloadPlot(self):
        self.PlotEmotion(self._FILE_EMOTION)
        self.PlotHRFromBVP(self._FILE_BVP)
        self.PlotEda(self._FILE_EDA)

    def setConfigureTimeInterval(self, initialTime, endTime):
        self._TIME_TAG_INITIAL = initialTime;
        self._TIME_TAG_END = endTime;
        self._DURATION_SESSION = self.UnixTime().diffTimeStampTags(self._TIME_TAG_INITIAL, self._TIME_TAG_END)
        self._POSITION_INITIAL_SESSION = self.UnixTime().diffTimeStamp(timeVideo, self._TIME_TAG_INITIAL) * 1000  
        self.loadingTimeProgressaBar(0, 0)

    def loadingTimeProgressaBar(self, shiftLeft, shiftRight):
        ut = self.UnixTime();
        timeLeft = ut.time_inc(self._TIME_TAG_INITIAL, shiftLeft)   
        timeRight = ut.time_reduce(self._TIME_TAG_END, shiftRight)     
        time = '{} / {}'.format(timeLeft.strftime('%H:%M:%S'), timeRight.strftime('%H:%M:%S'))
        timeProgressBar.setText(time)
        timeLabel.setText(time)
        timeLabel2.setText(time) 
        self.time = timeLeft

        return time;
    
    def setTags(self, path):

        global arrayTagsInitial;
        global arrayTagsEnd;

        try:

            
            tags = self._SD.LoadDataTags(path)
            if(len(tags) == 0):
                QMessageBox.information(self, "Message", "Does not exist tags");
                print("Does not exist tags");
                sys.exit(app.exec_()); 
            elif(len(tags) % 2 == 1):
                QMessageBox.information(self, "Message", "No match between Tags");
                print("No match between Tags");
                sys.exit(app.exec_());   
            elif (len(tags) % 2 == 0):
                arrayTagsInitial = tags[0::2];
                arrayTagsEnd = tags[1::2];
                self.setConfigureTimeInterval(arrayTagsInitial[0], arrayTagsEnd[0]);
            
        except:

            print("Erro during Loading Tags")
            sys.exit(app.exec_());
            
    def setTagVideo(self, path):
        global timeVideo;
        try:
            f = open(path, "r")
            timeVideo = float(f.read());           


        except:
            QMessageBox.information(self, "Message", "Error during Loading Video Tag");

            print("Erro during Loading Video Tag")
            sys.exit(app.exec_());
            
    def uiMainPanelAnnotation(self):
        
            self.win = self.PainelAnnotation(self)
            return self.win.build();
    
    def closeEvent(self, *args, **kwargs):
        sys.exit(app.exec_())
        return QtGui.QMainWindow.closeEvent(self, *args, **kwargs)
    
    def checkButton(self, emotion):
        if emotion:
            for button in self.buttonsEmotions:
                if button.isChecked():
                    self.selectedEmotion = button.text()
        else:
            for button in self.buttonsActions:
                if button.isChecked():
                    self.selectedAction = button.text()

    def unCheckButtons(self):
        self.buttonsEmotions[0].toggle()
        self.buttonsActions[0].toggle()

    def uiButtons(self):
        top = QtGui.QFrame()

        layout = QtGui.QGridLayout()

        self.buttonsEmotions = []
        index = 0

        #print("num emotions: " + str(len(self.emotions)))

        for n in range(len(self.emotions)):
            self.buttonsEmotions.append(QCheckBox(self.emotions[index]))
            self.buttonsEmotions[index].setChecked(False)
            self.buttonsEmotions[index].stateChanged.connect(lambda:self.checkButton(True))
            self.buttonsEmotions[index].setAutoExclusive(True)
            layout.addWidget(self.buttonsEmotions[index], n % 10, n / 10)
            index = index + 1


        top.setLayout(layout)
        top.setFrameShape(QtGui.QFrame.StyledPanel)
        top.setFrameShadow(QtGui.QFrame.Raised)

        self.buttonsActions = []
        index = 0
        bottom = QtGui.QFrame()
        layout = QtGui.QGridLayout()

        for n in range(len(self.actions)):
            button = QCheckBox(self.actions[index])
            button.setChecked(False)
            button.stateChanged.connect(lambda:self.checkButton(False))
            button.setAutoExclusive(True)
            layout.addWidget(button, n % 10, n / 10)
            self.buttonsActions.append(button)
            index = index + 1

        index = 0
        confirmButton = QtGui.QPushButton("")
        confirmButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
        confirmButton.setEnabled(True)
        confirmButton.setCheckable(True)
        confirmButton.clicked.connect(lambda:self.eventBtstateAn(3))

        #layout.addWidget(confirmButton)
        
        bottom.setLayout(layout)
        bottom.setFrameShape(QtGui.QFrame.StyledPanel)
        bottom.setFrameShadow(QtGui.QFrame.Raised)
        
        splitter = QtGui.QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.addWidget(confirmButton)
        splitter.setSizes([75,25,25])
        
        vbox = QtGui.QHBoxLayout()        
        
        vbox.addWidget(splitter)
        return vbox;

    def uiList(self):
        self.listWidget = QListWidget()
    
        # Resize width and height
        self.listWidget.resize(300, 120)

            
        for n in range(self.totalWindows):
            self.listWidget.addItem(self.listNameGenerator(n, 'none', 'none')[0])
            
        self.listWidget.itemSelectionChanged.connect(lambda:self.eventBtstateAn(4))
        self.listWidget.setWindowTitle('PyQT QListwidget Demo')

        gridl = QtGui.QGridLayout()
        gridl.addWidget(self.listWidget)
        return gridl;

    def uiTimeBarAnnotation(self):
               
        global positionRangeSlider;
        global endTimeEdit;
        global btLastAn;
        global btNextAn;
        global btAgainAn;

        positionRangeSlider = QRangeSlider()
    
        positionRangeSlider.handle.setTextColor(150)
        positionRangeSlider.setFixedHeight(30)
        
        btPlayer = QtGui.QPushButton("")
        btPlayer.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        btPlayer.setEnabled(True)
        btPlayer.setCheckable(True)
        btPlayer.clicked.connect(lambda:self.eventBtstate(btPlayer))

        btLastAn = QtGui.QPushButton("")
        btLastAn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        btLastAn.setEnabled(True)
        btLastAn.setCheckable(True)
        btLastAn.clicked.connect(lambda:self.eventBtstateAn(0))

        btNextAn = QtGui.QPushButton("")
        btNextAn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        btNextAn.setEnabled(True)
        btNextAn.setCheckable(True)
        btNextAn.clicked.connect(lambda:self.eventBtstateAn(1))

        btAgainAn = QtGui.QPushButton("")
        btAgainAn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        btAgainAn.setEnabled(True)
        btAgainAn.clicked.connect(lambda:self.eventBtstateAn(2))
        
        editsTimeLayout = QtGui.QVBoxLayout()        
        initialTimeLabel = QtGui.QLabel()

        timeProgressBar.setText("00:00:00/00:00:00")
        timeProgressBar.setSizePolicy(QtGui.QSizePolicy.Preferred,
                QtGui.QSizePolicy.Maximum)    
        editsTimeLayout.addStretch()    
        editsTimeLayout.addWidget(timeProgressBar)
        # editsTimeLayout.addStretch()
        
        
        vbox = QtGui.QHBoxLayout()
        vbox.addWidget(btLastAn)
        vbox.addWidget(btPlayer)
        vbox.addWidget(btNextAn)
        vbox.addWidget(btAgainAn)
        vbox.addWidget(positionRangeSlider)

        box = QtGui.QVBoxLayout()
        box.addLayout(vbox)
        box.addLayout(editsTimeLayout)
        return box;

    def eventBtstateAn(self, type):
        if type == 0:
            self.currentWindow = (self.currentWindow - 1) % self.totalWindows
            self.returnInitWindow()
            self.selectWindowList()

        if type == 1:
            self.currentWindow = (self.currentWindow + 1) % self.totalWindows
            self.returnInitWindow()
            self.selectWindowList()
        
        if type == 2:
            self.returnInitWindow()
            print("rewind")

        if type == 3:
            # windowEffects[currentWindow] = (currentEffect)
            texto,csv = self.listNameGenerator(self.currentWindow, 
                                           self.selectedEmotion,
                                           self.selectedAction)
            self.listWidget.item(self.currentWindow).setText(texto)
            self.windowEffects[self.currentWindow] = csv
            self.exportAffections()
            self.unCheckButtons()
            self.selectedEmotion = 'none'
            self.selectedAction = 'none'
            self.currentWindow = (self.currentWindow + 1) % self.totalWindows
            self.listWidget.setCurrentItem(self.listWidget.item(self.currentWindow))
        
        if type == 4:  # Select which member of the list will be changed
            for n in range(self.listWidget.count()):
                if self.listWidget.item(n).isSelected():
                    self.currentWindow = n
                    self.returnInitWindow()
        
    def exportAffections(self):
        #_START= datetime.fromtimestamp(float(self._TIME_TAG_INITIAL)).strftime('%H:%M:%S');
        #_END= datetime.fromtimestamp(float(self._TIME_TAG_END)).strftime('%H:%M:%S');
              
        _NAME = "S{0}_{1}_P{2}.csv".format(self._NUMBER_SESSION,self._EVALUATOR_NAME,self._NUMBER_PARTICIPANT)
        _FILE = "{0}/{1}".format(self._FILE_PATH_SESSION,_NAME)
        #_HEADER = "{0},{1},{2},{3}".format("Start","End","Emotion","Action")
        fileAffection = open(_FILE, "w")
        #fileAffection.write(_HEADER + "\n")
        for text in self.windowEffects:
            fileAffection.write(text + "\n")

    def selectWindowList(self):
        self.listWidget.setCurrentItem(self.listWidget.item(self.currentWindow))
        
    def getWindowInit(self):
        return float(self._TIME_TAG_INITIAL) + self.currentWindow * 10
    
    def getWindowEnd(self):
        endTime = float(self._TIME_TAG_INITIAL) + (self.currentWindow + 1) * 10
        if endTime > float(self._TIME_TAG_END):
            endTime = float(self._TIME_TAG_END)
        return endTime

    def listNameGenerator(self, index, emo, act):
        initTime = float(self._TIME_TAG_INITIAL) + index * 10
        endTime = float(self._TIME_TAG_INITIAL) + (index + 1) * 10
        if endTime > float(self._TIME_TAG_END):
            endTime = float(self._TIME_TAG_END)
        hours, minutes, seconds = self.getTimeDetails(datetime.fromtimestamp(float(initTime)))
        t = QTime(hours, minutes, seconds);
        hours2, minutes2, seconds2 = self.getTimeDetails(datetime.fromtimestamp(float(endTime)))
        t2 = QTime(hours2, minutes2, seconds2);  
        texto = "{0}-{1}|{2}|{3}".format(t.toString(),t2.toString(),emo,act);
        texto_csv = "{0};{1};{2};{3}".format(initTime,endTime,emo,act);
        return  ( texto,texto_csv )

    def openButtonsEmotions(self):
            self.selectedEmotion = 'none'
            try:
                
#                 self.emotions = ["Raiva","Insuficiencia","Pavor","Tristeza","Suavidade",
#                                 "Felicidade","Horror","Furia","Pesar","Nausea",
#                                 "Ansiedade","Descontracao","Desejo","Nervosismo","Solidao",
#                                 "Assustado","Loucura","Satisfacao","Maldisposicao","Vazio",
#                                 "Desejo","Panico","Saudade","Calma","Medo","Tranquilidade",
#                                 "Nojo","Preocupacao","Diversao","Simpatia","Frustracao",
#                                 "Determinacao","Surpresa",
#                                 "Desanimo","Concentracao","Stress"]
                self.emotions = ["Raiva","Loucura","Furia", "Stress",
                                 "Nojo", "Repulsa", "Maldisposicao", "Nausea",
                                 "Horror", "Assustado", "Medo","Panico",
                                 "Preocupacao", "Ansiedade","Pavor","Nervosismo",
                                 "Solidao", "Pesar","Tristeza","Vazio", "Desanimo","Frustracao",
                                 "Insuficiencia","Desejo", "Saudade",
                                 "Calma", "Tranquilidade","Descontracao", "Suavidade","Concentracao",
                                 "Felicidade", "Diversao", "Satisfacao","Simpatia"]
                
                self.emotions = sorted(self.emotions);
                self.emotions.insert(0, "Nenhum")
            except:
                print("Erro during Loading Emotion Buttons")
                sys.exit(app.exec_());
    
    def openButtonsActions(self):
        self.selectedAction = 'none'
        try:
            self.actions = ["Colisao","Acelerando","Perdendo Posicao","Cambio",
                            "Frenagem","Ganhando Posicao","Drift","Off Road","Roll over"]
            self.actions = sorted(self.actions);
            self.actions.insert(0, "Nenhum")

        except:
            print("Erro during Loading Action Buttons")

            sys.exit(app.exec_());

    def uiMainPanel(self):
        self.splitter = QtGui.QSplitter(Qt.Horizontal)    
        self.splitter.setSizes([1000, 200])    
        containerLeft = QtGui.QWidget()
        containerLeft.setLayout(self.uiPanelPlot())
        
        containerRight = QtGui.QWidget()
        containerRight.setLayout(self.uiPanelVideo())
        
        self.splitter.addWidget(containerLeft)
        self.splitter.addWidget(containerRight)
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.splitter)
        containerbottom = QtGui.QWidget()
        containerbottom.setLayout(self.uiTimeBar())
        vbox.addWidget(containerbottom)
        return vbox;
                 
    def uiPanelPlot(self):
        global pwEDA;        
        global isCreatedPlotEda;
        global pwHR;
        global isCreatedPlotHR;
        global pwEmotion;
        global isCreatedPlotBVP;
        self.isCreatedPlotEda = False;
        self.isCreatedPlotHR = False;
        self.isCreatedPlotEmotion = False;
        
        
        # global pwTemp;
        splitter = QtGui.QSplitter(Qt.Vertical)
        
        vbox = QtGui.QVBoxLayout()
        
        pwEDA = pg.PlotWidget(name='Plot1', title='EDA plot')    
        pwEDA.setLabel('bottom', 'Time (Seconds)')
        pwEDA.setLabel('left', 'EDA value')
        splitter.addWidget(pwEDA)       
        
        pwHR = pg.PlotWidget(name='Plot2', title='HR plot')
        pwHR.setLabel('bottom', 'Time (Seconds)')
        pwHR.setLabel('left', 'HR value')        
        splitter.addWidget(pwHR)
        
        pwEmotion = pg.PlotWidget(name='Plot3', title='Emotion plot')
        pwEmotion.setLabel('bottom', 'Time (Seconds)')
        pwEmotion.setLabel('left', 'P(E)')
        splitter.addWidget(pwEmotion)
        
        
        # containerbottom = QtGui.QWidget()
        # containerbottom.setLayout(self.uiTimeBar())
        # splitter.addWidget(containerbottom)
        vbox.addWidget(splitter)
       
        
        return vbox
    
    def uiTimeBar(self):
               
        global positionRangeSlider;
        global timeProgressBar;    
        global btPlayer;        
        positionRangeSlider = QRangeSlider()
    
        positionRangeSlider.handle.setTextColor(150)
        positionRangeSlider.setFixedHeight(30)
        # positionRangeSlider.setFixedWidth(1000)
        
        btPlayer = QtGui.QPushButton("")
        btPlayer.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        btPlayer.setEnabled(False)
        btPlayer.setCheckable(True)
        btPlayer.clicked.connect(lambda:self.eventBtstate(btPlayer))
        
        editsTimeLayout = QtGui.QHBoxLayout()        
        timeProgressBar = QtGui.QLabel()
        
        timeProgressBar.setText("00:00:00/00:00:00")
        timeProgressBar.setSizePolicy(QtGui.QSizePolicy.Preferred,
                QtGui.QSizePolicy.Maximum)    
        editsTimeLayout.addStretch()    
        editsTimeLayout.addWidget(timeProgressBar)
        # editsTimeLayout.addStretch()
        
        vbox = QtGui.QHBoxLayout()
        vbox.addWidget(btPlayer)
        vbox.addWidget(positionRangeSlider)
        
        
        box = QtGui.QVBoxLayout()
        box.addLayout(vbox)
        box.addLayout(editsTimeLayout)
        return box;

    def uiPanelVideo(self):
        print("uiPanelVideo")
        global timeLabel;
        global timeLabel2;
        

        timeLabel = QtGui.QLabel()
        timeLabel.setSizePolicy(QtGui.QSizePolicy.Preferred,
                QtGui.QSizePolicy.Maximum)
        timeLabel.setText("00:00:00")
        videoWidget = QVideoWidget()
        self.mediaPlayer.setVideoOutput(videoWidget)
        top = QtGui.QFrame();
       
        layout = QtGui.QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addWidget(timeLabel)
        top.setLayout(layout)
        top.setFrameShape(QtGui.QFrame.StyledPanel)
        top.setFrameShadow(QtGui.QFrame.Raised)
        
        videoWidget = QVideoWidget()
        self.mediaPlayer2.setVideoOutput(videoWidget)
        
        bottom = QtGui.QFrame()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(videoWidget)
        bottom.setLayout(layout)
        bottom.setFrameShape(QtGui.QFrame.StyledPanel)
        bottom.setFrameShadow(QtGui.QFrame.Raised)
        timeLabel2 = QtGui.QLabel()
        timeLabel2.setSizePolicy(QtGui.QSizePolicy.Preferred,
                QtGui.QSizePolicy.Maximum)
        timeLabel2.setText("00:00:00")
        layout.addWidget(timeLabel2)
        # bottom.layout().setContentsMargins(0, 0, 150, 150)
        # bottom.layout().setSpacing(1)
        
        splitter = QtGui.QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setSizes([100, 100])
        
        vbox = QtGui.QHBoxLayout()        
        
        vbox.addWidget(splitter)
        return vbox;
        
    def openFile(self, filename):        
        if(("SC") in filename):
            self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename)))
        else: self.mediaPlayer2.setMedia(
                    QMediaContent(QUrl.fromLocalFile(filename)))
       
        btPlayer.setEnabled(True)
            
    def play(self):
        try:
           
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause();
                self.mediaPlayer2.pause()
            else:
                positionRangeSlider.setMoved(False) 
                self.mediaPlayer2.play()
                self.mediaPlayer.play()
                
        except: 
            print("No media player") 
                
    def mediaStateChanged(self, state):
        try:
            
                if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                    btPlayer.setIcon(
                            self.style().standardIcon(QStyle.SP_MediaPause))
                else:
                    btPlayer.setIcon(
                            self.style().standardIcon(QStyle.SP_MediaPlay))     
        except: 
            print("No media player");
            
    def positionChanged(self, position):       
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            # print("positionChanged: Position:: %s" % position)
            # if(position>=self._POSITION_INITIAL_SESSION):
            positionRangeSlider.setStart(position)
            self.addLinearRegionInPlotWidget()
            if self._ANNOTATION:
                if(self.time >= datetime.fromtimestamp(float(self.getWindowEnd()))):
                    self.returnInitWindow()
                    
    def durationChanged(self, duration):  
        print("Duration Video in miliseconds: %s" % duration)
        self.setPositionInPlayer(self._POSITION_INITIAL_SESSION)
     
    def setPositionInPlayer(self, position):

        # print("setPositionInPlayer")
        self.mediaPlayer.setPosition(position)
        self.mediaPlayer2.setPosition(position)


    def handleError(self):
        # selfm.playButton.setEnabled(False)
        timeLabel.setText("Error: " + self.mediaPlayer.errorString())
            
    def updateRangerSlider(self):
        
        if(self._DISCONNECT_RANGE_SLIDER):
            positionRangeSlider.endValueChanged.disconnect(self.eventChangeRightRangeValue)
            positionRangeSlider.startValueChanged.disconnect(self.eventChangeLeftRangeValue)
        
        global positionEndSession;
        positionEndSession = self._POSITION_INITIAL_SESSION + self._DURATION_SESSION;
        print("Duration session in miliseconds: %s" % (self._DURATION_SESSION))
        print("Initial Point: %s" % self._POSITION_INITIAL_SESSION)
        print("End Point: %s" % (positionEndSession))
        
        positionRangeSlider.setMin(self._POSITION_INITIAL_SESSION)
        positionRangeSlider.setMax(positionEndSession)
        positionRangeSlider.setRange(self._POSITION_INITIAL_SESSION, positionEndSession)   
        
        positionRangeSlider.endValueChanged.connect(self.eventChangeRightRangeValue)
        positionRangeSlider.startValueChanged.connect(self.eventChangeLeftRangeValue)
        self._DISCONNECT_RANGE_SLIDER = False;                
    
    def eventChangeRightRangeValue(self, index):
        
        # print("eventChangeRightRangeValue : %s - %s" % (index,positionEndSession))
        if(index <= positionRangeSlider.start()):
            time = self.loadingTimeProgressaBar(0, 0)
        elif ((positionEndSession - index) >= 0):
            time = self.loadingTimeProgressaBar(positionRangeSlider.start() - self._POSITION_INITIAL_SESSION,
                                                positionEndSession - index)
        if(positionRangeSlider.getMoved()):
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause();
                self.mediaPlayer2.pause()
            
            self.setPositionInPlayer(positionRangeSlider.start())
    
    def eventChangeLeftRangeValue(self, index):
        # print("eventChangeLeftRangeValue: %s - %s = %s" % (index,self._POSITION_INITIAL_SESSION,(index-self._POSITION_INITIAL_SESSION)))
        # print(positionRangeSlider.getMoved())
        if(index >= positionRangeSlider.end() and not self._ANNOTATION):

            positionRangeSlider.setRange(self._POSITION_INITIAL_SESSION, positionEndSession)
            self.mediaPlayer.pause();
            self.mediaPlayer2.pause();
            self.setPositionInPlayer(self._POSITION_INITIAL_SESSION)
            time = self.loadingTimeProgressaBar(0, 0)
            self.clearLinearRegion()

        elif(index - self._POSITION_INITIAL_SESSION >= 0):
            time = self.loadingTimeProgressaBar(index - self._POSITION_INITIAL_SESSION,
                                                positionEndSession - positionRangeSlider.end())
        if(positionRangeSlider.getMoved()):
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause();
                self.mediaPlayer2.pause();           
            self.setPositionInPlayer(index)
            
    def eventBtstate(self, b):
        
        if b.isChecked():
            print ("Pressed Play") 
            self.play()
           
        else:
            print ("Pressed Pause")
            positionRangeSlider.setMoved(False)
            self.play()                   
             
    def returnInitWindow(self):
            newPosition = self.UnixTime().diffTimeStamp(timeVideo, self.getWindowInit())
            self.setPositionInPlayer(newPosition * 1000)

    def PlotEda(self, path):    
        print("PlotEda")   
       
        if(not path):
            return
        global ts;
        global metricsEDA;
        global plotEDA
            
        _UT = self.UnixTime();

        eda = self.EDAPeakDetectionScript(self)

        ts, raw_eda, filtered_eda, peaks, amp = eda.processEDA(path,
                                                               _UT.run(self._TIME_TAG_INITIAL),
                                                               _UT.run(self._TIME_TAG_END))

        metricsEDA = self.ProcessingData().getMetricsEDA(raw_eda)
        normalize_data_eda = self.ProcessingData().normalize(filtered_eda)
        
        if(self.isCreatedPlotEda):
            plotEDA.clear()
            pwEDA.clear();

            # pwEDA.remove(kind='item') 
        plotEDA = pwEDA.plot(title='EDA Pike', pen='r')
        
        if(not self.isCreatedPlotEda):
            pwEDA.getPlotItem().addLegend(offset=(10, 10))        
            pwEDA.addItem(pg.PlotDataItem(pen='r', name='GSR Value', antialias=False))
            pwEDA.addItem(pg.PlotDataItem(pen='b', name='GSR Peak', antialias=False))       
            pwEDA.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            pwEDA.setMouseEnabled(x=False, y=False)
            axis = self.DateAxis(orientation='bottom')
            axis.attachToPlotItem(pwEDA.getPlotItem())
            
        
        plotEDA.setData(x=ts, y=normalize_data_eda)        
        
        for peak in peaks:
            aux = str(float("{0:.2f}".format(filtered_eda[peak])));            
            l = pwEDA.addLine(x=ts[peak], y=None, pen='b')
            label = pg.InfLineLabel(l, aux, position=float(aux), rotateAxis=(0, 0), anchor=(2, 1))
        
        self.lrEDA = pg.LinearRegionItem([ts[0], ts[len(ts) - 1]], bounds=[ts[0], ts[len(ts) - 1]])  
        self.lrEDA.setZValue(-10)  
        pwEDA.addItem(self.lrEDA) 
        self.lrEDA.setRegion([ts[0], ts[0]])
        
        self.isCreatedPlotEda = True;
      
    def createHR(self, ts_hr, hr, filteredBVP):
        print("createHR")    
        global plotHR
        
        timeHR =self.UnixTime().timeFrom(self._TIME_TAG_INITIAL, ts_hr)
        
        n_array = list(zip(timeHR, hr))        
        df = pd.DataFrame(n_array, columns=['timeHR', 'hr'])        
        df['timeHR'] = [datetime.fromtimestamp(ts) for ts in df['timeHR']]  
        df = df[(df['timeHR'] >= self.UnixTime().run(self._TIME_TAG_INITIAL)) & 
             (df['timeHR'] <= self.UnixTime().run(self._TIME_TAG_END))]       
        
        hr = df['hr'].tolist()
        timeHR = [datetime.timestamp(dt) for dt in df['timeHR']]
       
        if(self.isCreatedPlotHR):
            plotHR.clear()
            pwHR.clear();       
        

        plotHR = pwHR.plot(title="HR", pen='r')
        if(not self.isCreatedPlotHR):
            pwHR.getPlotItem().addLegend()
            pwHR.addItem(pg.PlotDataItem(pen='r', name='HR Value', antialias=False))
            pwHR.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)

            pwHR.setMouseEnabled(x=False, y=False)
            axis = self.DateAxis(orientation='bottom')
            axis.attachToPlotItem(pwHR.getPlotItem())
        
        normalize_data_hr = self.ProcessingData().normalize(df['hr'])
        plotHR.setData(x=timeHR, y=normalize_data_hr.tolist())

        
        plotHR = pwHR.plot(title="HRV", pen='b')
        if(not self.isCreatedPlotHR):
            pwHR.addItem(pg.PlotDataItem(pen='b', name='HRV Value', antialias=False))
        
        RRI_DF = self.EmpaticaHRV().getRRI(filteredBVP, self._TIME_TAG_INITIAL, 64)
        HRV_DF = self.EmpaticaHRV().getHRV(RRI_DF, np.mean(hr))
        timeHR = HRV_DF['Timestamp'].tolist()
        normalize_data_hrv = self.ProcessingData().normalize(HRV_DF['HRV'])
        plotHR.setData(x=timeHR, y=normalize_data_hrv.tolist())
       
        self.lrHR = pg.LinearRegionItem([timeHR[0], timeHR[len(timeHR) - 1]],
                                        bounds=[timeHR[0], timeHR[len(timeHR) - 1]])  
        self.lrHR.setZValue(-10)  
        pwHR.addItem(self.lrHR)
        self.lrHR.setRegion([timeHR[0], timeHR[0]])
        
        self.isCreatedPlotHR = True;
    
    def PlotHRFromBVP(self, path):
        print("PlotHRFromBVP")
        if(not path):  # If not exist file then dont loading
            return;
       
        
        count, data, startTime, samplingRate = self._SD.LoadDataBVP(path)       
        filteredBVP, ts_hr, hr = self.ProcessingData().ProcessedBVPDataE4(data) 
        ut = self.UnixTime();        
        endTime, tsBVP = ut.time_array(self._TIME_TAG_INITIAL, count, samplingRate)
                
        n_array = list(zip(tsBVP, filteredBVP))        
        df = pd.DataFrame(n_array, columns=['tsBVP', 'filteredBVP'])        
        df['tsBVP'] = [datetime.fromtimestamp(ts) for ts in df['tsBVP']]
        # Cut in time
        df = df[(df['tsBVP'] >= self.UnixTime().run(self._TIME_TAG_INITIAL)) & 
                (df['tsBVP'] <= self.UnixTime().run(self._TIME_TAG_END))]       
        filteredBVP = df['filteredBVP'].tolist()       
       
        self.createHR(ts_hr, hr, filteredBVP)
   
    def PlotEmotion(self, url):
        print("PlotEmotion")

        if(not url):  # If not exist file then dont loading
            return;
            
        try:
            
            
            df = self._SD.LoadDataFacialExpression(indexSession=None, path=url);
            
            d1 = list(zip(df['Time'], df['Happiness'], df['Sadness'], df['Anger'],
                                     df['Surprise'], df['Fear'], df['Disgust']))        
            dataframe = pd.DataFrame(d1, columns=['tsEmotion', 'Happiness', 'Sadness',
                                                 "Anger", "Surprise", "Fear", 'Disgust'])   
            
            # Cut in time
            dataframe = dataframe[(dataframe['tsEmotion'] >= self.UnixTime().run(self._TIME_TAG_INITIAL)) & 
                                  (dataframe['tsEmotion'] <= self.UnixTime().run(self._TIME_TAG_END))]
    
            array1 = dataframe['Happiness'].tolist()

            array2 = dataframe['Sadness'].tolist()
            array3 = dataframe['Anger'].tolist()
            array4 = dataframe['Surprise'].tolist()
            array5 = dataframe['Fear'].tolist()
            array6 = dataframe['Disgust'].tolist()
            
            tsEmotion = [datetime.timestamp(dt) for dt in dataframe['tsEmotion']]
           
            if(self.isCreatedPlotEmotion):
                pwEmotion.clear();

            if(not self.isCreatedPlotEmotion):
                pwEmotion.addLegend((50, 60), offset=(30, 30)) 
                pwEmotion.addItem(pg.PlotDataItem(pen='b', name='Happiness', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='c', name='Sadness', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='y', name='Anger', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='g', name='Surprise', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='k', name='Fear', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='m', name='Disgust', antialias=False))
                pwEmotion.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)

                axis = self.DateAxis(orientation='bottom')
                axis.attachToPlotItem(pwEmotion.getPlotItem()) 

            # b: blue,g: green,r: red,c: cyan,m: magenta,y: yellow,k: black,w: white
            for nameEmotion in self._LIST_EMOTION:
                if(nameEmotion == 'Happiness'):
                    plotEmotion = pwEmotion.plot(title='Happiness', pen='b')                           
                    plotEmotion.setData(x=tsEmotion, y=array1)  
                if(nameEmotion == 'Sadness'):  
                    plotEmotion = pwEmotion.plot(title='Sadness', pen='c')

                    plotEmotion.setData(x=tsEmotion, y=array2)
                if(nameEmotion == 'Anger'): 
                    plotEmotion = pwEmotion.plot(title='Anger', pen='y')
                    plotEmotion.setData(x=tsEmotion, y=array3)
                if(nameEmotion == 'Surprise'):
                    plotEmotion = pwEmotion.plot(title='Surprise', pen='g')
                    plotEmotion.setData(x=tsEmotion, y=array4)
                if(nameEmotion == 'Fear'):
                    plotEmotion = pwEmotion.plot(title='Fear', pen='k')
                    plotEmotion.setData(x=tsEmotion, y=array5)
                if(nameEmotion == 'Disgust'):
                    plotEmotion = pwEmotion.plot(title='Disgust', pen='m')
                    plotEmotion.setData(x=tsEmotion, y=array6)           

            # Plot was created

            self.lrEmotion = pg.LinearRegionItem([tsEmotion[0], tsEmotion[len(tsEmotion) - 1]],
                                                 bounds=[tsEmotion[0], tsEmotion[len(tsEmotion) - 1]])  
            self.lrEmotion.setZValue(-10)  
            pwEmotion.addItem(self.lrEmotion) 
            self.lrEmotion.setRegion([tsEmotion[0], tsEmotion[0]])
            
            self.isCreatedPlotEmotion = True;
            self._LIST_EMOTION = ['Happiness', 'Sadness', 'Anger',
                      'Fear', 'Surprise', 'Disgust']
            return True;
        except: 
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro in PlotEmotion")
            return False;
 
    def clearLinearRegion(self):
        ut = self.UnixTime();

        indexInitial = datetime.timestamp(ut.time_inc(self._TIME_TAG_INITIAL, 0))


        self.printRegion(indexInitial, indexInitial)        
            
    def addLinearRegionInPlotWidget(self):
        if self.mediaPlayer.state() != QMediaPlayer.PausedState:

            ut = self.UnixTime();
            indexInitial = datetime.timestamp(ut.time_inc(self._TIME_TAG_INITIAL,
                                                          positionRangeSlider.start() - self._POSITION_INITIAL_SESSION))

            indexEnd = datetime.timestamp(ut.time_reduce(self._TIME_TAG_END,
                                                          positionEndSession - positionRangeSlider.end()))
            
            self.printRegion(indexInitial, indexEnd) 
                
    def printRegion(self, indexInitial, indexEnd):
        try:
            if(self.isCreatedPlotEda):
                self.lrEDA.setRegion([indexInitial, indexEnd])

            if(self.isCreatedPlotHR):
                self.lrHR.setRegion([indexInitial, indexEnd]) 
            if(self.isCreatedPlotEmotion):            
                self.lrEmotion.setRegion([indexInitial, indexEnd])         
        except ValueError:
            print('Non-numeric data found in the file.')           
        except EOFError:
            print('Why did you do an EOF on me?')      
        except:            
            print('Linear Region null') 
    
    def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        qp.end()
    
    def getTimeDetails(self, duration):
        
        hours = duration.hour
        minutes = duration.minute                
        seconds = duration.second 
        return (hours, minutes, seconds) 

    
    def is_video_file(self, filename):
        video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.avi', '.dv-avi',
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2')

        if filename.endswith(video_file_extensions):
            return True
        return False;
    
    class PainelAnnotation(QWidget):
        
        def __init__(self, obj, **kwargs):
            QWidget.__init__(self, **kwargs)
            self.obj = obj
            FlowChartGame.reset(self.obj)
        def build(self):
            
            e1 = QtGui.QLineEdit()
            e1.setValidator(QtGui.QIntValidator())
            e1.setMaxLength(4)
            e1.setAlignment(Qt.AlignLeft)
            e1.setFont(QtGui.QFont("Arial",12))
            e1.textChanged.connect(self.textchangedNumber)
            
            e2 = QtGui.QLineEdit()
            e2.textChanged.connect(self.textchangedName)
            
            e3 = QtGui.QLineEdit()
            e3.textChanged.connect(self.textchangedNumberSession)
            e3.setValidator(QtGui.QIntValidator())
            e3.setMaxLength(1)
            e3.setAlignment(Qt.AlignLeft)
            e3.setFont(QtGui.QFont("Arial",12))
            
            flo = QtGui.QFormLayout()
            flo.addRow("Number of Participant", e1)
            flo.addRow("Number of Session",e3)
            flo.addRow("Evaluator's name",e2)
          
            splitter = QtGui.QSplitter(Qt.Horizontal)  
            splitter.setSizes([300, 300, 300])
    
            vertSplitter = QtGui.QSplitter(Qt.Vertical)
            vertSplitter.setSizes([500, 100])
    
            containerLeft = QtGui.QWidget()
            containerLeft.setLayout(self.obj.uiList())
    
            containerMid = QtGui.QWidget()
            containerMid.setLayout(self.obj.uiButtons())
           
            splitter.addWidget(containerLeft)
            splitter.addWidget(containerMid)
    
            vertSplitter.addWidget(splitter)
    
            gridl = QtGui.QVBoxLayout()
            gridl.addLayout(flo)
            gridl.addWidget(vertSplitter)
    
            self.setLayout(gridl)
    
            self.setWindowTitle("Annotation")
            self.adjustSize()
            fg = self.frameGeometry()
            cp = QtGui.QDesktopWidget().availableGeometry().center()
            fg.moveCenter(cp)
            self.move(fg.topLeft())
            self.show()
            return vertSplitter;
        
        
        def closeEvent(self, *args, **kwargs):
            FlowChartGame.reset(self.obj)
            return QWidget.closeEvent(self, *args, **kwargs)
        
        def textchangedNumber(self,text):
            self.obj._NUMBER_PARTICIPANT = text
        def textchangedNumberSession(self,text):
            self.obj._NUMBER_SESSION = text   
        def textchangedName(self,text):
            self.obj._EVALUATOR_NAME = text;
            
    class EDAPeakDetectionScript:
        global SAMPLE_RATE
        SAMPLE_RATE = 8
        
        def __init__(self,obj): 
            self.obj = obj
            pass;
        
        def datetime_to_float(self,d):
            epoch = datetime.datetime.utcfromtimestamp(0)
            total_seconds =  (d - epoch).total_seconds()
            # total_seconds will be in decimals (millisecond precision)
            return total_seconds
        
        def processEDA(self,signal,startTime,endTime):
            thresh = 0.02;
            offset = 1;
            start_WT = 4;
            end_WT = 4;
            

            data = self.loadData_E4(signal)
            df = self.calcPeakFeatures(data,offset,thresh,start_WT,end_WT)
            peakData = df[(df.index >= startTime) & (df.index <= endTime)]
            ts = []
            peak = []
            raw_eda = []
            filtered_eda = []
            amp = []
            for item in peakData.index:
                #print(type(item))
                ts.append(item.to_pydatetime().timestamp())
            index = 0;
            
            for p in peakData['peaks']:
                if(p == 1):
                    peak.append(index)
                    amp.append(peakData['amp'][index])
                filtered_eda.append(peakData['filtered_eda'][index])
                raw_eda.append(peakData['EDA'][index])
                index = index + 1;
                
            return ts,raw_eda,filtered_eda,peak,amp;
        def findPeaks(self,data, offset, start_WT, end_WT, thres=0, sampleRate=SAMPLE_RATE):
            '''
                This function finds the peaks of an EDA signal and returns basic properties.
                Also, peak_end is assumed to be no later than the start of the next peak. (Is this okay??)
        
                ********* INPUTS **********
                data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
                offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak
                start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
                end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
                thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
                sampleRate:  number of samples per second, default=8
        
                ********* OUTPUTS **********
                peaks:               list of binary, 1 if apex of SCR
                peak_start:          list of binary, 1 if start of SCR
                peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
                peak_end:            list of binary, 1 if rec.t/2 of SCR
                peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
                amplitude:           list of floats,  value of EDA at apex - value of EDA at start
                max_deriv:           list of floats, max derivative within 1 second of apex of SCR
        
            '''
            EDA_deriv = data['filtered_eda'][1:].values - data['filtered_eda'][:-1].values
            peaks = np.zeros(len(EDA_deriv))
            peak_sign = np.sign(EDA_deriv)
            for i in range(int(offset), int(len(EDA_deriv) - offset)):
                if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
                    peaks[i] = 1
                    for j in range(1, int(offset)):
                        if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                            #if peak_sign[i-j]==-1 or peak_sign[i+j]==1:
                            peaks[i] = 0
                            break
        
            # Finding start of peaks
            peak_start = np.zeros(len(EDA_deriv))
            peak_start_times = [''] * len(data)
            max_deriv = np.zeros(len(data))
            rise_time = np.zeros(len(data))
        
            for i in range(0, len(peaks)):
                if peaks[i] == 1:
                    temp_start = max(0, i - sampleRate)
                    max_deriv[i] = max(EDA_deriv[temp_start:i])
                    start_deriv = .01 * max_deriv[i]
        
                    found = False
                    find_start = i
                    # has to peak within start_WT seconds
                    while found == False and find_start > (i - start_WT * sampleRate):
                        if EDA_deriv[find_start] < start_deriv:
                            found = True
                            peak_start[find_start] = 1
                            peak_start_times[i] = data.index[find_start]
                            rise_time[i] = self.get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))
        
                        find_start = find_start - 1
        
                    # If we didn't find a start
                    if found == False:
                        peak_start[i - start_WT * sampleRate] = 1
                        peak_start_times[i] = data.index[i - start_WT * sampleRate]
                        rise_time[i] = start_WT
        
                    # Check if amplitude is too small
                    if thres > 0 and (data['EDA'].iloc[i] - data['EDA'][peak_start_times[i]]) < thres:
                        peaks[i] = 0
                        peak_start[i] = 0
                        peak_start_times[i] = ''
                        max_deriv[i] = 0
                        rise_time[i] = 0
        
            # Finding the end of the peak, amplitude of peak
            peak_end = np.zeros(len(data))
            peak_end_times = [''] * len(data)
            amplitude = np.zeros(len(data))
            decay_time = np.zeros(len(data))
            half_rise = [''] * len(data)
            SCR_width = np.zeros(len(data))
        
            for i in range(0, len(peaks)):
                if peaks[i] == 1:
                    peak_amp = data['EDA'].iloc[i]
                    start_amp = data['EDA'][peak_start_times[i]]
                    amplitude[i] = peak_amp - start_amp
        
                    half_amp = amplitude[i] * .5 + start_amp
        
                    found = False
                    find_end = i
                    # has to decay within end_WT seconds
                    while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                        if data['EDA'].iloc[find_end] < half_amp:
                            found = True
                            peak_end[find_end] = 1
                            peak_end_times[i] = data.index[find_end]
                            decay_time[i] = self.get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])
        
                            # Find width
                            find_rise = i
                            found_rise = False
                            while found_rise == False:
                                if data['EDA'].iloc[find_rise] < half_amp:
                                    found_rise = True
                                    half_rise[i] = data.index[find_rise]
                                    SCR_width[i] = self.get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                                find_rise = find_rise - 1
        
                        elif peak_start[find_end] == 1:
                            found = True
                            peak_end[find_end] = 1
                            peak_end_times[i] = data.index[find_end]
                        find_end = find_end + 1
        
                    # If we didn't find an end
                    if found == False:
                        min_index = np.argmin(data['EDA'].iloc[i:(i + end_WT * sampleRate)].tolist())
                        peak_end[i + min_index] = 1
                        peak_end_times[i] = data.index[i + min_index]
        
            peaks = np.concatenate((peaks, np.array([0])))
            peak_start = np.concatenate((peak_start, np.array([0])))
            max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)
        
            return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise
    
        def get_seconds_and_microseconds(self,pandas_time):
            return pandas_time.seconds + pandas_time.microseconds * 1e-6
    
        def calcPeakFeatures(self,data,offset,thresh,start_WT,end_WT):
            returnedPeakData = self.findPeaks(data, offset*SAMPLE_RATE, start_WT, end_WT, thresh, SAMPLE_RATE)
            data['peaks'] = returnedPeakData[0]
            data['peak_start'] = returnedPeakData[1]
            data['peak_end'] = returnedPeakData[3]
        
            data['peak_start_times'] = returnedPeakData[2]
            data['peak_end_times'] = returnedPeakData[4]
            data['half_rise'] = returnedPeakData[10]
            # Note: If an SCR doesn't decrease to 50% of amplitude, then the peak_end = min(the next peak's start, 15 seconds after peak)
            data['amp'] = returnedPeakData[5]
            data['max_deriv'] = returnedPeakData[6]
            data['rise_time'] = returnedPeakData[7]
            data['decay_time'] = returnedPeakData[8]
            data['SCR_width'] = returnedPeakData[9]
        
            featureData = data[data.peaks==1][['EDA','rise_time','max_deriv','amp','decay_time','SCR_width']]
        
            # Replace 0s with NaN, this is where the 50% of the peak was not found, too close to the next peak
            featureData[['SCR_width','decay_time']]=featureData[['SCR_width','decay_time']].replace(0, np.nan)
            featureData['AUC']=featureData['amp']*featureData['SCR_width']
        
            
        
            return data
        
    
    # draws a graph of the data with the peaks marked on it
    # assumes that 'data' dataframe already contains the 'peaks' column
        def plotPeaks(self,data, x_seconds, sampleRate = SAMPLE_RATE):
            if x_seconds:
                time_m = np.arange(0,len(data))/float(sampleRate)
            else:
                time_m = np.arange(0,len(data))/(sampleRate*60.)
        
            data_min = min(data['EDA'])
            data_max = max(data['EDA'])
        
            #Plot the data with the Peaks marked
            plt.figure(1,figsize=(20, 5))
            peak_height = data_max * 1.15
            data['peaks_plot'] = data['peaks'] * peak_height
            
            plt.plot(time_m,data['peaks_plot'],'#4DBD33')
            #plt.plot(time_m,data['EDA'])
            plt.plot(time_m,data['filtered_eda'])
            plt.xlim([0,time_m[-1]])
            y_min = min(0, data_min) - (data_max - data_min) * 0.1
            plt.ylim([min(y_min, data_min),peak_height])
            plt.title('EDA with Peaks marked')
            plt.ylabel('$\mu$S')
            if x_seconds:
                plt.xlabel('Time (s)')
            else:
                plt.xlabel('Time (min)')
        
            plt.show()
    
        def chooseValueOrDefault(self,str_input, default):
               
            if str_input == "":
                return default
            else:
                return float(str_input)
    
        def loadSingleFile_E4(self,filepath,list_of_columns, expected_sample_rate,freq):
            # Load data
            data = pd.read_csv(filepath)
            
            # Get the startTime and sample rate
            startTime = self.obj.UnixTime().run(data.columns.values[0])
            sampleRate = float(data.iloc[0][0])
            data = data[data.index!=0]
            data.index = data.index-1
            
            # Reset the data frame assuming expected_sample_rate
            data.columns = list_of_columns
            if sampleRate != expected_sample_rate:
                print('ERROR, NOT SAMPLED AT {0}HZ. PROBLEMS WILL OCCUR\n'.format(expected_sample_rate))
        
            # Make sure data has a sample rate of 8Hz
            data = self.interpolateDataTo8Hz(data,sampleRate,startTime)
            return data
        
        def loadData_E4(self,filepath):
            # Load EDA data
            eda_data = self.loadSingleFile_E4(filepath,["EDA"],4,"250L")
            # Get the filtered data using a low-pass butterworth filter (cutoff:1hz, fs:8hz, order:6)
            eda_data['filtered_eda'] =  self.butter_lowpass_filter(eda_data['EDA'], 1.0, 8, 6)
        
           
            return eda_data
        
        
        def interpolateDataTo8Hz(self,data,sample_rate,startTime):
            print("sample_rate %s " % (sample_rate))
            if sample_rate<8:
                # Upsample by linear interpolation
                if sample_rate==2:
                    data.index = pd.date_range(start=startTime, periods=len(data), freq='500L')
                elif sample_rate==4:
                    data.index = pd.date_range(start=startTime, periods=len(data), freq='250L')
                data = data.resample("125L").mean()
            else:
                if sample_rate>8:
                    # Downsample
                    idx_range = list(range(0,len(data))) # TODO: double check this one
                    data = data.iloc[idx_range[0::int(int(sample_rate)/8)]]
                # Set the index to be 8Hz
                data.index = pd.date_range(start=startTime, periods=len(data), freq='125L')
        
            # Interpolate all empty values
            data = self.interpolateEmptyValues(data)
            return data
        
        def interpolateEmptyValues(self,data):
            cols = data.columns.values
            for c in cols:
                data.loc[:, c] = data[c].interpolate()
        
            return data
        
        def butter_lowpass(self,cutoff, fs, order=5):
            # Filtering Helper functions
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = scisig.butter(order, normal_cutoff, btype='low', analog=False)
            return b, a
        
        def butter_lowpass_filter(self,data, cutoff, fs, order=5):
            # Filtering Helper functions
            b, a = self.butter_lowpass(cutoff, fs, order=order)
            y = scisig.lfilter(b, a, data)
            return y

    class DateAxis(pg.AxisItem):
        """
        A tool that provides a date-time aware axis. It is implemented as an
        AxisItem that interpretes positions as unix timestamps (i.e. seconds
        since 1970).
        The labels and the tick positions are dynamically adjusted depending
        on the range.
        It provides a  :meth:`attachToPlotItem` method to add it to a given
        PlotItem
        """
        
        # Max width in pixels reserved for each label in axis
        _pxLabelWidth = 10
    
        def __init__(self, *args, **kwargs):
            AxisItem.__init__(self, *args, **kwargs)
            self._oldAxis = None
    
        def tickValues(self, minVal, maxVal, size):
            """
            Reimplemented from PlotItem to adjust to the range and to force
            the ticks at "round" positions in the context of time units instead of
            rounding in a decimal base
            """
    
            maxMajSteps = int(size/self._pxLabelWidth)
           
            dt1 = datetime.fromtimestamp(minVal)
            dt2 = datetime.fromtimestamp(maxVal)
            #print("dt(%s,%s)" %(dt1,dt2))
            dx = maxVal - minVal
            #print("dx (%s): "%dx)
            majticks = []
    
            if dx > 63072001:  # 3600s*24*(365+366) = 2 years (count leap year)
                d = timedelta(days=366)
                for y in range(dt1.year + 1, dt2.year):
                    dt = datetime(year=y, month=1, day=1)
                    majticks.append(mktime(dt.timetuple()))
    
            elif dx > 5270400:  # 3600s*24*61 = 61 days
                d = timedelta(days=31)
                dt = dt1.replace(day=1, hour=0, minute=0,
                                 second=0, microsecond=0) + d
                while dt < dt2:
                    # make sure that we are on day 1 (even if always sum 31 days)
                    dt = dt.replace(day=1)
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 172800:  # 3600s24*2 = 2 days
                d = timedelta(days=1)
                dt = dt1.replace(hour=0, minute=0, second=0, microsecond=0) + d
                while dt < dt2:
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 7200:  # 3600s*2 = 2hours
                d = timedelta(hours=1)
                dt = dt1.replace(minute=0, second=0, microsecond=0) + d
                while dt < dt2:
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 1200:  # 60s*20 = 20 minutes
                d = timedelta(minutes=10)
                dt = dt1.replace(minute=(dt1.minute // 10) * 10,
                                 second=0, microsecond=0) + d
                while dt < dt2:
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 120:  # 60s*2 = 2 minutes [Para casso de sessões curtas]
                d = timedelta(seconds=30)
                dt = dt1.replace(second=0, microsecond=0) + d
                while dt < dt2:
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 20:  # 20s
                d = timedelta(seconds=10)
                dt = dt1.replace(second=(dt1.second // 10) * 10, microsecond=0) + d
                while dt < dt2:
                    majticks.append(mktime(dt.timetuple()))
                    dt += d
    
            elif dx > 2:  # 2s
                d = timedelta(seconds=1)
                majticks = range(int(minVal), int(maxVal))
    
            else:  # <2s , use standard implementation from parent
                return AxisItem.tickValues(self, minVal, maxVal, size)
    
            L = len(majticks)
            if L > maxMajSteps:
                majticks = majticks[::int(numpy.ceil(float(L) / maxMajSteps))]
    
            return [(d.total_seconds(), majticks)]
    
        def tickStrings(self, values, scale, spacing):
            """Reimplemented from PlotItem to adjust to the range"""
            ret = []
            if not values:
                return []
    
            if spacing >= 31622400:  # 366 days
                fmt = "%Y"
    
            elif spacing >= 2678400:  # 31 days
                fmt = "%Y %b"
    
            elif spacing >= 86400:  # = 1 day
                fmt = "%b/%d"
    
            elif spacing >= 3600:  # 1 h
                fmt = "%b/%d-%Hh"
    
            elif spacing >= 60:  # 1 m
                fmt = "%H:%M:%S"
    
            elif spacing >= 1:  # 1s
                fmt = "%H:%M:%S"
    
            else:
                # less than 2s (show microseconds)
                # fmt = '%S.%f"'
                fmt = '[+%fms]'  # explicitly relative to last second
    
            for x in values:
                try:
                    t = datetime.fromtimestamp(x)
                    ret.append(t.strftime(fmt))
                except ValueError:  # Windows can't handle dates before 1970
                    ret.append('')
    
            return ret
    
        def attachToPlotItem(self, plotItem):
            """Add this axis to the given PlotItem
            :param plotItem: (PlotItem)
            """
            self.setParentItem(plotItem)
            viewBox = plotItem.getViewBox()
            self.linkToView(viewBox)
            self._oldAxis = plotItem.axes[self.orientation]['item']
            self._oldAxis.hide()
            plotItem.axes[self.orientation]['item'] = self
            pos = plotItem.axes[self.orientation]['pos']
            plotItem.layout.addItem(self, *pos)
            self.setZValue(-1000)
    
        def detachFromPlotItem(self):
            """Remove this axis from its attached PlotItem
            (not yet implemented)
            """
            raise NotImplementedError()  # TODO

    class TableView(QTableWidget):
        def __init__(self, data,title, *args):
            QTableWidget.__init__(self, *args)
            self.data = data
            self.setWindowTitle(title)
            self.setData()
            self.resizeColumnsToContents()
            self.resizeRowsToContents()
            self.setSelectionMode(QAbstractItemView.SingleSelection)
            self.setSelectionBehavior(QAbstractItemView.SelectRows)
            #self#.show()
        def setModeMultiple(self):
            self.setSelectionMode(QAbstractItemView.MultiSelection)
            self.setSelectionBehavior(QAbstractItemView.SelectRows)
        def setData(self): 
            horHeaders = []
            item1 = QTableWidgetItem()
            item1.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEditable | 
                  Qt.ItemIsEnabled)
            for n, key in enumerate(sorted(self.data.keys(),reverse=True)):
            #for n, key in enumerate((self.data.keys())):
                horHeaders.append(key)
                for m, item in enumerate(self.data[key]):
                    newitem = QTableWidgetItem(item)
                    newitem.setFlags(Qt.ItemIsSelectable  | Qt.ItemIsEnabled)
                    self.setItem(m, n, newitem)
            self.setHorizontalHeaderLabels(horHeaders)

    class EmpaticaHRV:
        def __init__(self):
            pass
        def bvpPeaks(self,signal):
            cb = np.array(signal)
            x = peakutils.indexes(cb, thres=0.02/max(cb), min_dist=0.1)
            y = []
            i = 0
            while (i < (len(x)-1)):
                if x[i+1] - x[i] < 15:
                    y.append(x[i])
                    x = np.delete(x, i+1)
                else:
                    y.append(x[i])
                i += 1
            return y
        
        def getRRI(self, signal, start, sample_rate):
            peakIDX = self.bvpPeaks(signal)
            spr = 1 / sample_rate # seconds between readings
            start_time = float(start)
            timestamp = [start_time, (peakIDX[0] * spr) + start_time ] 
            ibi = [0, 0]
            for i in range(1, len(peakIDX)):
                timestamp.append(peakIDX[i] * spr + start_time)
                ibi.append((peakIDX[i] - peakIDX[i-1]) * spr)
        
            df = pd.DataFrame({'Timestamp': timestamp, 'IBI': ibi})
            return df
        
        def getHRV(self,data, avg_heart_rate):
            rri = np.array(data['IBI']) * 1000
            RR_list = rri.tolist()
            #RR_diff = []
            RR_sqdiff = []
            RR_diff_timestamp = []
            cnt = 2
            while (cnt < (len(RR_list)-1)): 
                #RR_diff.append(abs(RR_list[cnt+1] - RR_list[cnt])) 
                RR_sqdiff.append(math.pow(RR_list[cnt+1] - RR_list[cnt], 2)) 
                RR_diff_timestamp.append(data['Timestamp'][cnt])
                cnt += 1
            hrv_window_length = 10
            window_length_samples = int(hrv_window_length*(avg_heart_rate/60))
            #SDNN = []
            RMSSD = []
            index = 1
            for val in RR_sqdiff:
                if index < int(window_length_samples):
                    #SDNNchunk = RR_diff[:index:]
                    RMSSDchunk = RR_sqdiff[:index:]
                else:
                    #SDNNchunk = RR_diff[(index-window_length_samples):index:]
                    RMSSDchunk = RR_sqdiff[(index-window_length_samples):index:]
                #SDNN.append(np.std(SDNNchunk))
                RMSSD.append(math.sqrt(np.std(RMSSDchunk)))
                index += 1
            dt = np.dtype('Float64')
            #SDNN = np.array(SDNN, dtype=dt)
            RMSSD = np.array(RMSSD, dtype=dt)
            df = pd.DataFrame({'Timestamp': RR_diff_timestamp, 'HRV': RMSSD})
            return df
    
    class ProcessingData:
        
        def __init__(self):
            pass;
        
        def normalize(self,myarray):
            
            arrayNormalized = myarray;
            max_value = max(myarray);
            min_value = min(myarray);
            arrayNormalized = (arrayNormalized - min_value) / (max_value - min_value); 
            return arrayNormalized;
        
        def getMetricsEDA(self,signal):
            metric = biosppy.tools.signal_stats(signal)
            return metric;
        
        def ProcessedBVPDataE4(self,sig):
            ts, filtered, onsets, ts_hr, hr = biosppy.bvp.bvp(signal=sig, sampling_rate=64., show=True);       
            return  filtered, ts_hr, hr;

    class UnixTime(object):
        '''
        classdocs
        '''
    
    
        def __init__(self):
            '''
            Constructor
            '''
        def run(self,strTime):
            #ts = int("1284101485")
            ts = float(strTime)
    
            # if you encounter a "year is out of range" error the timestamp
            # may be in milliseconds, try `ts /= 1000` in that case
            #dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S');
            #print(dt)
            #print(datetime.fromtimestamp(ts) + timedelta(seconds=1/4))
            #return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S');
            return datetime.fromtimestamp(ts);
        
        def runGMT(self,strTime):
            #ts = int("1284101485")
            ts = float(strTime)
    
            # if you encounter a "year is out of range" error the timestamp
            # may be in milliseconds, try `ts /= 1000` in that case
            #dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S');
            #print(dt)
            #print(datetime.fromtimestamp(ts) + timedelta(seconds=1/4))
            return datetime.utcfromtimestamp(ts)
            #return datetime.fromtimestamp(ts);
        
        def time_array(self,strTime,count,samplingRate):
            aux = 0;
            array = []
            dt = self.run(strTime)
            print("Count: %s " % (len(count)/samplingRate))
                #diff = accumulate - startTime;
            for ts in count:
                #print(dt+aux*timedelta(seconds=1/4))
                accumulate = dt+aux*timedelta(seconds=1/samplingRate);
                #st= datetime.fromtimestamp(accumulate.timestamp()).strftime('%H:%M:%S');
                
                array.append(accumulate.timestamp())
                #array.append(str(diff))
                aux = aux + 1;
            #print(datetime.fromtimestamp(array[len(array)-1]))
            return array[len(array)-1],array;
        
        def time_inc(self,strTime,value):
            dt = self.run(strTime)
            accumulate = dt+timedelta(seconds=int(value/1000));
            return accumulate;
        
        def time_reduce(self,strTime,value):
            dt = self.run(strTime)
            accumulate = dt-timedelta(seconds=int(value/1000));
            return accumulate;
        
        def timeFrom(self,strTime,arraySecond):
            array = []
            dt = self.run(strTime)
            for ts in (arraySecond):
                #print(dt+aux*timedelta(seconds=1/4))
                accumulate = dt+timedelta(seconds=ts);
                #diff = accumulate - startTime;
                array.append(accumulate.timestamp())
                #array.append(str(diff))
                
            return array;
        
        def time_array_segment(self,strTime,count,samplingRate):
            array = []
            dt = self.run(strTime)
                #diff = accumulate - startTime;
            
            aux = 0;
            count_segment = int(len(count)/samplingRate)
            for ts in range(count_segment):
                accumulate = dt+aux*timedelta(seconds=1);
                t = datetime.fromtimestamp(accumulate.timestamp()).strftime('%H:%M:%S');
                #diff = accumulate - startTime;
                #print(accumulate)
                array.append(accumulate.timestamp())
                #array.append(str(diff))
                aux = aux + 1;
            print("Last Time: %s" %(datetime.fromtimestamp(array[len(array)-1])))
            return array[len(array)-1],array;  
        
        def time_1(self,strTime,samplingRate):
            aux = 0;
            array = []
            dt = self.run(strTime)
                #diff = accumulate - startTime;
            for ts in (range(1294)):
                #print(dt+aux*timedelta(seconds=1/4))
                accumulate = dt+aux*timedelta(seconds=1);
               
                array.append(accumulate.timestamp())
                #array.append(str(diff))
                aux = aux + 1;
            diff = accumulate - self.run(strTime);
            return array;    
      
        def diffTimeStamp(self,strTimeVideo, strT2):
            """
            Method that calculates difference between video time and arbitrary time
    
            Parameters
            ----------
            strTimeVideo: String
               video time.
            strT2: String
                arbitrary time.
            """
            tstamp1 = self.runGMT(strTimeVideo)
            tstamp2 = self.run(strT2)
    
            if tstamp1 > tstamp2:
                td = tstamp1 - tstamp2
            else:
                td = tstamp2 - tstamp1
            td_seconds = int(round(td.total_seconds()))
            return td_seconds;
        
        def diffTimeStampTags(self,strT1, strT2):
            """
            Method that calculates difference between video time and arbitrary time
    
            Parameters
            ----------
            strTimeVideo: String
               video time.
            strT2: String
                arbitrary time.
            """
            tstamp1 = self.run(strT1)
            tstamp2 = self.run(strT2)
    
            if tstamp1 > tstamp2:
                td = tstamp1 - tstamp2
            else:
                td = tstamp2 - tstamp1
            td_seconds = int(round(td.total_seconds()))
            return td_seconds*1000;

    class SourceData:
           
        def __init__(self):
            print('Constructor SourceData')
           
        
     
        def LoadDataFacialExpression(self, path,indexSession=None):
            
            if(indexSession != None):
                source = 'EMOCAO_*.csv';            
                url = path.format(indexSession,indexSession,source)
                file_ = glob.glob(url)[0]
            else: file_ = path;    
            list_=[];
            
            df = pd.read_csv(file_,index_col=None, header=0)
            list_.append(df)
                
            frame = pd.concat(list_)
           
                
            df = pd.DataFrame(columns=['Time','Neutral', 'Happiness','Sadness',
                                       'Anger','Fear','Surprise','Disgust']);    
            
            dates_list = [];
            for d in frame['Time']:
                dates_list.append(datetime.strptime(d, '%d/%m/%Y %H:%M:%S.%f')) 
                
            #df['Time'] = ut.getTimeElapsed(dates_list); 
            df['Time'] = (dates_list); 
            df['Neutral'] = ((np.array(frame['neutral']).astype(float))) ;
            df['Happiness'] = (np.array(frame['happiness']).astype(float)) ;
            df['Sadness'] = (np.array(frame['sadness']).astype(float)) ;
            df['Anger'] = (np.array(frame['anger']).astype(float)) ;
            df['Fear'] = (np.array(frame['fear']).astype(float)) ;
            df['Surprise'] = (np.array(frame['surprise']).astype(float)) ;
            df['Disgust'] = (np.array(frame['disgust']).astype(float)) ;
            
            return df;
        
        def LoadDataEDA(self,path):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"EDA") 
            print("Load EDA Data")
            index = np.arange(len (e3data.data))
            dataset_array = []    
            for item in e3data.data:
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,e3data.startTime,e3data.getEndTime(),e3data.samplingRate); 
        
        def LoadDataEDASlice(self,path,startTime,endTime):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"EDA") 
            print("Load Data EDA Slice")
            slice = e3data.getSlide(startTime,endTime)
            index = np.arange(len (slice.data))
            dataset_array = []    
            for item in slice.data:
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,slice.samplingRate); 
        
        def LoadDataHR(self,path):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"HR") 
            print("Load HR Data")
            index = np.arange(len (e3data.data))
            dataset_array = []    
            for item in e3data.data:
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,e3data.startTime,e3data.getEndTime(),e3data.samplingRate); 
        
        def LoadDataHRSlice(self,path,startTime,endTime):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"HR") 
            index = np.arange(len (e3data.data))
            print(e3data.data)        
            slice = e3data.getSlide(startTime,endTime)
            print("Load Data HR Slice")
            print(slice.data)
            index = np.arange(len (slice.data))
            dataset_array = []    
            for item in slice.data:
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,slice.samplingRate);   
        
        def LoadDataBVP(self,path):
            print("Load BVP Data")
    
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"BVP") 
            index = np.arange(len (e3data.data))
            dataset_array = []    
            for item in e3data.data:
                dataset_array.append(float(item[0])) 
    
            return (index,dataset_array,e3data.startTime,e3data.samplingRate); 
        
        def LoadDataBVPSlice(self,path,startTime,endTime):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"BVP") 
            print("Load BVP Data Slice")
            slice = e3data.getSlide(startTime,endTime)
            index = np.arange(len (slice.data))
            dataset_array = []    
            for item in slice.data:
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,slice.samplingRate);      
        
        def LoadDataTemp(self,path):
            e3data = self.E3Data.newE3DataFromFilePath(self,path,"TEMP") 
            print("Load Temp Data")
            index = np.arange(len (e3data.data))
            dataset_array = []    
            for item in e3data.data:
                
                dataset_array.append(float(item[0])) 
            
            return (index,dataset_array,e3data.startTime,e3data.getEndTime(),e3data.samplingRate); 
        
        def LoadDataTags(self,path):
            try:
                e3data = self.E3Data.newE3DataFromFilePath(self,path,"TAGS") 
                index = np.arange(len (e3data.data))
                dataset_array = []    
                for item in e3data.data:
                    
                    dataset_array.append((item[0])) 
                return (dataset_array);
            except: 
                print("Oops!", sys.exc_info()[0], "occured.")
                print("Erro in LoadDataTags")
        class E3Data:
            def __init__(self,dataType,startTime,samplingRate,data):
                self.dataType = dataType
                self.startTime = float ( startTime)
                self.samplingRate =  float (samplingRate)
                self.data = data
            
            
        
            def toString(self,unixTime=True):
                if(unixTime):
                    return "Data Type: %s, Start Time:%s, End Time:%s  SamplingRate %s" %(self.dataType,self.startTime,self.getEndTime(),self.samplingRate)
                else:
                    _string = "Data Type: %s, Start Time:%s, End Time:%s  SamplingRate %s" %(
                            self.dataType,datetime.datetime.fromtimestamp(self.startTime)
                            ,datetime.datetime.fromtimestamp( float(self.getEndTime())),self.samplingRate)
                    return _string
            def getData(self):
                return self.data
            def getEndTime(self):
                _startDateTime = datetime.datetime.fromtimestamp(self.startTime)
                _endDateTime = _startDateTime +  datetime.timedelta (seconds=len(self.data) / self.samplingRate )
                return  _endDateTime.strftime("%s")
            def getSlide(self, start,end):
                _slideStartTime = datetime.datetime.fromtimestamp( self.startTime) 
                
                _slideStartTime = _slideStartTime + datetime.timedelta(seconds=start)
                return self.E3Data(self.dataType,
                        _slideStartTime.strftime("%s"),self.samplingRate, 
                        self.data[start * int (self.samplingRate): end * int (self.samplingRate)])
        
            def getNormalTime(self):
                return datetime.datetime.fromtimestamp(self.startTime)
        
            def saveToFile(self,_path):
                with open(_path,"w") as _FILE_OUTPUT:
                    _FILE_OUTPUT.write(str( self.startTime ) + "\n")
                    _FILE_OUTPUT.write(str( self.samplingRate) + "\n")
                    for _line in self.data:
                        _FILE_OUTPUT.writelines(','.join(str(y) for y in _line)+"\n")
        
            @staticmethod
            def newE3DataFromFilePath(self,_FILE_INPUT_PATH,_DATA_TYPE):
               
                with open(_FILE_INPUT_PATH,"r") as _FILE_INPUT:
                    _lineNumber = 0
                    _samplingRate = -1
                    _startTime = ""
                    _data = []
                    for _line in _FILE_INPUT:
                        if (_DATA_TYPE == "TAGS"):
                            _dataLine = _line.replace("\n","").split(",")
                            _data.append(_dataLine)
                            _startTime=0
                            continue;
                        if (_lineNumber == 0):
                            _startTime = _line.replace("\n","").split(",")[0]
                        if (_lineNumber == 1):
                            if not (_DATA_TYPE == "IBI"):
                                _samplingRate = _line.replace("\n","").split(",")[0]
                        if(_lineNumber >1):
                            _dataLine = _line.replace("\n","").split(",")
                            _data.append(_dataLine)
                        _lineNumber += 1
                    
                    return self.E3Data(_DATA_TYPE,_startTime,_samplingRate,_data)   

    


    

    
def scale(val, src, dst):
        return int(((val - src[0]) / float(src[1]-src[0])) * (dst[1]-dst[0]) + dst[0])

class Ui_Form(object):
        def setupUi(self, Form):
            Form.setObjectName("QRangeSlider")
            Form.resize(300, 30)
            Form.setStyleSheet(DEFAULT_CSS)
            self.gridLayout = QtWidgets.QGridLayout(Form)
            self.gridLayout.setContentsMargins(0, 0, 0, 0)
            self.gridLayout.setSpacing(0)
            self.gridLayout.setObjectName("gridLayout")
            self._splitter = QtWidgets.QSplitter(Form)
            self._splitter.setMinimumSize(QtCore.QSize(0, 0))
            self._splitter.setMaximumSize(QtCore.QSize(16777215, 16777215))
            self._splitter.setOrientation(QtCore.Qt.Horizontal)
            self._splitter.setObjectName("splitter")
            self._head = QtWidgets.QGroupBox(self._splitter)
            self._head.setTitle("")
            self._head.setObjectName("Head")
            self._handle = QtWidgets.QGroupBox(self._splitter)
            self._handle.setTitle("")
            self._handle.setObjectName("Span")
            self._tail = QtWidgets.QGroupBox(self._splitter)
            self._tail.setTitle("")
            self._tail.setObjectName("Tail")
            self.gridLayout.addWidget(self._splitter, 0, 0, 1, 1)
            self.retranslateUi(Form)
            QtCore.QMetaObject.connectSlotsByName(Form)
    
        def retranslateUi(self, Form):
            _translate = QtCore.QCoreApplication.translate
            Form.setWindowTitle(_translate("QRangeSlider", "QRangeSlider"))
    
    
class Element(QtWidgets.QGroupBox):
        def __init__(self, parent, main):
            super(Element, self).__init__(parent)
            self.main = main
    
        def setStyleSheet(self, style):
            self.parent().setStyleSheet(style)
    
        def textColor(self):
            return getattr(self, '__textColor', QtGui.QColor(125, 125, 125))
    
        def setTextColor(self, color):
            if type(color) == tuple and len(color) == 3:
                color = QtGui.QColor(color[0], color[1], color[2])
            elif type(color) == int:
                color = QtGui.QColor(color, color, color)
            setattr(self, '__textColor', color)
    
        def paintEvent(self, event):
            qp = QtGui.QPainter()
            qp.begin(self)
            if self.main.drawValues():
                self.drawText(event, qp)
            qp.end()
  
class Head(Element):
        def __init__(self, parent, main):
            super(Head, self).__init__(parent, main)
    
        def drawText(self, event, qp):
            qp.setPen(self.textColor())
            qp.setFont(QtGui.QFont('Arial', 10))
            qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.main.min()))

    
        
        
        
class Handle(Element):
        def __init__(self, parent, main):
            super(Handle, self).__init__(parent, main)
    
        def drawText(self, event, qp):
            qp.setPen(self.textColor())
            qp.setFont(QtGui.QFont('Arial', 10))
            qp.drawText(event.rect(), QtCore.Qt.AlignLeft, str(self.main.start()))
            qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.main.end()))
    
    
            
        def mouseMoveEvent(self, event):
            event.accept()
            mx = event.globalX()
            _mx = getattr(self, '__mx', None)
            if not _mx:
                setattr(self, '__mx', mx)
                dx = 0
            else:
                dx = mx - _mx
            setattr(self, '__mx', mx)
            if dx == 0:
                event.ignore()
                return
            elif dx > 0:
                dx = 1
            elif dx < 0:
                dx = -1
            s = self.main.start() + dx
            e = self.main.end() + dx
            if s >= self.main.min() and e <= self.main.max():
                self.main.setRange(s, e)
class Tail(Element):
            def __init__(self, parent, main):
                super(Tail, self).__init__(parent, main)
        
            def drawText(self, event, qp):
                qp.setPen(self.textColor())
                qp.setFont(QtGui.QFont('Arial', 10))
                qp.drawText(event.rect(), QtCore.Qt.AlignRight, str(self.main.max()))

class QRangeSlider(QtWidgets.QWidget, Ui_Form):
        endValueChanged = QtCore.pyqtSignal(int)
        maxValueChanged = QtCore.pyqtSignal(int)
        minValueChanged = QtCore.pyqtSignal(int)
        startValueChanged = QtCore.pyqtSignal(int)
        minValueChanged = QtCore.pyqtSignal(int)
        maxValueChanged = QtCore.pyqtSignal(int)
        startValueChanged = QtCore.pyqtSignal(int)
        endValueChanged = QtCore.pyqtSignal(int)
        
        _SPLIT_START = 1
        _SPLIT_END = 2
        isMoved = False;
        def __init__(self, parent=None):
            super(QRangeSlider, self).__init__(parent)
            self.setupUi(self)
            self.setMouseTracking(False)
            self._splitter.splitterMoved.connect(self._handleMoveSplitter)
            self._head_layout = QtWidgets.QHBoxLayout()
            self._head_layout.setSpacing(0)
            self._head_layout.setContentsMargins(0, 0, 0, 0)
            self._head.setLayout(self._head_layout)
            self.head = Head(self._head, main=self)
            self._head_layout.addWidget(self.head)
            self._handle_layout = QtWidgets.QHBoxLayout()
            self._handle_layout.setSpacing(0)
            self._handle_layout.setContentsMargins(0, 0, 0, 0)
            self._handle.setLayout(self._handle_layout)
            self.handle = Handle(self._handle, main=self)
            self.handle.setTextColor((150, 255, 150))
            self._handle_layout.addWidget(self.handle)
            self._tail_layout = QtWidgets.QHBoxLayout()
            self._tail_layout.setSpacing(0)
            self._tail_layout.setContentsMargins(0, 0, 0, 0)
            self._tail.setLayout(self._tail_layout)
            self.tail = Tail(self._tail, main=self)
            self._tail_layout.addWidget(self.tail)
            self.setMin(0)
            self.setMax(99)
            self.setStart(0)
            self.setEnd(99)
            self.setDrawValues(False)
            
        
    
        def getMoved(self):
            return self.isMoved;
        def setMoved(self,value):
            self.isMoved = value   
        def min(self):
            return getattr(self, '__min', None)
    
        def max(self):
            return getattr(self, '__max', None)
    
        def setMin(self, value):
            setattr(self, '__min', value)
            self.minValueChanged.emit(value)
    
        def setMax(self, value):
            setattr(self, '__max', value)
            self.maxValueChanged.emit(value)
    
        def start(self):
            return getattr(self, '__start', None)
    
        def end(self):
            return getattr(self, '__end', None)
    
        def _setStart(self, value):
            setattr(self, '__start', value)
            self.startValueChanged.emit(value)
    
        def setStart(self, value):
            v = self._valueToPos(value)
            self._splitter.splitterMoved.disconnect()
            self._splitter.moveSplitter(v, self._SPLIT_START)
            self._splitter.splitterMoved.connect(self._handleMoveSplitter)
            self._setStart(value)
    
        def _setEnd(self, value):
            setattr(self, '__end', value)
            self.endValueChanged.emit(value)
    
        def setEnd(self, value):
            v = self._valueToPos(value)
            self._splitter.splitterMoved.disconnect()
            self._splitter.moveSplitter(v, self._SPLIT_END)
            self._splitter.splitterMoved.connect(self._handleMoveSplitter)
            self._setEnd(value)
    
        def drawValues(self):
            return getattr(self, '__drawValues', None)
    
        def setDrawValues(self, draw):
            setattr(self, '__drawValues', draw)
    
        def getRange(self):
            return (self.start(), self.end())
    
        def setRange(self, start, end):
            self.setStart(start)
            self.setEnd(end)
    
        def keyPressEvent(self, event):
            key = event.key()
            if key == QtCore.Qt.Key_Left:
                s = self.start()-1
                e = self.end()-1
            elif key == QtCore.Qt.Key_Right:
                s = self.start()+1
                e = self.end()+1
            else:
                event.ignore()
                return
            event.accept()
            if s >= self.min() and e <= self.max():
                self.setRange(s, e)
    
        def setBackgroundStyle(self, style):
            self._tail.setStyleSheet(style)
            self._head.setStyleSheet(style)
    
        def setSpanStyle(self, style):
            self._handle.setStyleSheet(style)
    
        def _valueToPos(self, value):
            return scale(value, (self.min(), self.max()), (0, self.width()))
    
        def _posToValue(self, xpos):
            return scale(xpos, (0, self.width()), (self.min(), self.max()))
    
        def _handleMoveSplitter(self, xpos, index):
            self.setMoved(True)
            hw = self._splitter.handleWidth()
            def _lockWidth(widget):
                width = widget.size().width()
                widget.setMinimumWidth(width)
                widget.setMaximumWidth(width)
            def _unlockWidth(widget):
                widget.setMinimumWidth(0)
                widget.setMaximumWidth(16777215)
            v = self._posToValue(xpos)
            if index == self._SPLIT_START:
                _lockWidth(self._tail)
                if v >= self.end():
                    return
                offset = -20
                w = xpos + offset
                self._setStart(v)
            elif index == self._SPLIT_END:
                _lockWidth(self._head)
                if v <= self.start():
                    return
                offset = -40
                w = self.width() - xpos + offset
                self._setEnd(v)
            _unlockWidth(self._tail)
            _unlockWidth(self._head)
            _unlockWidth(self._handle)  






if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    
    flow = FlowChartGame('testeee')
    flow.setGeometry(10, 10, 1000, 800)
    flow.setWindowTitle("Project Game Data Explorer (PGD Ex)")
    flow.show()

    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
    sys.exit(app.exec_())
        
        
        
