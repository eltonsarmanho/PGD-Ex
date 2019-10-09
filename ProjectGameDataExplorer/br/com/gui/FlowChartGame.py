
'''
Created on 15 de abr de 2019

@author: eltonss
'''
import matplotlib

from PyQt5.QtCore import  Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget, QStyle
from datetime import datetime
from pyqtgraph.Qt import QtCore, QtGui
import sys, os
matplotlib.use('Agg')

from br.com.gui.DateAxis import DateAxis
from br.com.gui.TableView import TableView
from br.com.gui.qrangeslider import QRangeSlider
from br.com.util.EDAPeakDetectionScript import EDAPeakDetectionScript
from br.com.util.EmpaticaHRV import EmpaticaHRV
from br.com.util.ProcessingData import ProcessingData
from br.com.util.SourceData import SourceData
from br.com.util.UnixTime import UnixTime
import numpy as np
import pandas as pd
import pyqtgraph as pg

matplotlib.use('Agg')


class FlowChartGame(QtGui.QMainWindow):
    
    timeTagEnd = 0;
    timeTagInitial = 0;
    durantionSession = 0;
    positionInitialSession = 0;
    
    fileBVP = "";
    fileEDA = "" ;
    fileEmotion = "";
    forceDisconnectRangeSlider = False;
    _listEmotion =  ['Happiness','Sadness','Anger',
                      'Fear','Surprise','Disgust']
    
    def __init__(self, buffer_size=0, data_buffer=[], graph_title="", parent=None):
        super(FlowChartGame, self).__init__(parent)               

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        #global  timer;
        #timer = QtCore.QTimer(self)
        
        self.createMediaPlayer();
        self.windowPlots()
        
    def createMediaPlayer(self):
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
        timeIntervals = QtGui.QAction("Time Intervals", self)
        emotionalComponents = QtGui.QAction("Emotional Components", self)
        
        open = QtGui.QAction("Open E4 Data File with Video", self)
        open.setShortcut("Ctrl+O")
        #openSV = QtGui.QAction("Open E4 Data File without Video", self)
        #openSV.setShortcut("Ctrl+A")
        file.addAction(open)
        #file.addAction(openSV)
                    
        quit = QtGui.QAction("Quit", self)
        quit.setShortcut("Ctrl+Q") 
        
        restart = QtGui.QAction("Restart", self)
        restart.setShortcut("Ctrl+R") 
        
        file.addAction(restart)
        file.addAction(quit)
        
        #tools.addAction(resetLG)
        tools.addAction(resetPB)
        tools.addAction(metricEDA)
        tools.addAction(emotionalComponents)
        tools.addAction(timeIntervals)
        file.triggered[QtGui.QAction].connect(self.processtrigger)
        tools.triggered[QtGui.QAction].connect(self.processTools)

        self.setLayout(layout)
    
    def processTools(self,q):
        if(q.text() == "EDA Metrics"):
           
            data ={'Metrics':['Mean','Median','Max','Var','Std_dev','Kurtosis','skewness'],
           'Value': [str(metricsEDA['mean']),str(metricsEDA['median']),str(metricsEDA['max']),
                    str(metricsEDA['var']),str(metricsEDA['std_dev']),
                    str(metricsEDA['kurtosis']),str(metricsEDA['skewness'])]}
          
            self.tv = TableView(data,"EDA Metrics",7,2)
            
            self.tv.show()
        elif(q.text() == "Emotional Components"):
            _list =  ['Happiness','Sadness','Anger',
                      'Fear','Surprise','Disgust']
            data = {'Emotional components': _list}
            
            self.tv = TableView(data,"Time Intervals",6,1)            
            
            self.tv.setModeMultiple()
            self.tv.resizeColumnsToContents()
            self.tv.horizontalHeader().setSectionResizeMode(QtGui.QHeaderView.Stretch)                
            self.win = QWidget()
            
            def getSelectedInterval():
                indexes = self.tv.selectionModel().selectedRows()
                listSelected =[]
                for index in sorted(indexes):
                    print("index %s Emotion %s" % (index.row(),_list[index.row()])) 
                    listSelected.append(_list[index.row()])
                self._listEmotion = listSelected;
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
          
        elif(q.text() == "Time Intervals"):
                
            ut = UnixTime();
            timeLeft = []
            timeRight= []
            for (t1,t2) in zip(arrayTagsInitial,arrayTagsEnd):                    
                timeLeft.append(ut.time_inc(t1, 0).strftime('%H:%M:%S'))
                timeRight.append(ut.time_reduce(t2, 0).strftime('%H:%M:%S'))    
                
            data ={'Initial Time':timeLeft,'End Time': timeRight}
          
            self.tv = TableView(data,"Time Intervals",len(arrayTagsInitial),2)            
            self.tv.resizeColumnsToContents()
            self.tv.horizontalHeader().setSectionResizeMode(QtGui.QHeaderView.Stretch)                
            self.win = QWidget()

            def getSelectedInterval():
                indexes = self.tv.selectionModel().selectedRows()
                for index in sorted(indexes):
                    self.setConfigureTimeInterval(arrayTagsInitial[index.row()], 
                                                  arrayTagsEnd[index.row()])  
                
                self.workloadPlot()
                self.forceDisconnectRangeSlider = True;
                self.updateRangerSlider()
                self.setPositionInPlayer(self.positionInitialSession)
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
            #if(timer.isActive()):
            #    timer.stop()                  
            #    timer.timeout.disconnect(self.eventUpdateTimeLine)
            self.mediaPlayer.pause()
            self.mediaPlayer2.pause()  
            self.durationChanged(0)
            #Update Progress Bar and Spinner        
            self.updateRangerSlider()
            self.updateSpinnerOfTime() 
            self.clearLinearRegion()       
            
        pass;  
   
    def processtrigger(self, q):
        
        if(q.text() == 'Open E4 Data File with Video'):
            self.loadingVisualization()       
        elif (q.text() == "Quit"):
            sys.exit(app.exec_())
        elif (q.text() == "Restart"):
            os.execl(sys.executable, sys.executable, *sys.argv) 
            
    def loadingVisualization(self):
        if(self.workload()):
            self.updateRangerSlider()            
            
    def workload(self ):
               
        dlg = QtGui.QFileDialog()       
        dlg.setFileMode(QtGui.QFileDialog.ExistingFiles)
        filenames = []
        try:
        
            if dlg.exec_():
                filenames = dlg.selectedFiles()               
            else:
                QMessageBox.information(self, "Message", "No appropriate file Located or no file selected"); 
                return False
            
            videoSC = "" 
            videoExtra = ""
            self.fileBVP = ""
            self.fileEDA = ""
            self.fileEmotion = ""
            tagFileVideo = ""
            tagFile = ""
            for file in filenames:
                filename = os.path.basename(file)               
                
                if(not(self.is_video_file(filename) or filename.endswith('.csv'))):
                    QMessageBox.information(self, "Message", "No appropriate file Located");
                    return False;
                
                if(self.is_video_file(filename) and ("SC") in filename):
                    videoSC = file;
                elif(self.is_video_file(filename) and (("WC") in filename or ("HV") in filename)):
                    videoExtra = file;   
                elif(filename == 'timevideo.csv'):                    
                    tagFileVideo = file;
                elif(filename == 'tags.csv'):                    
                    tagFile = file;
                elif filename == 'BVP.csv':
                    self.fileBVP = file;                
                elif filename == 'EDA.csv':
                    self.fileEDA = file; 
                elif ("EMOCAO") in filename:
                    self.fileEmotion = file;            
            
            if tagFileVideo:
                self.setTagVideo(tagFileVideo);
            else: 
                QMessageBox.information(self, "Message", "Tag Video not selected");
                return False;
                    
            if tagFile:                
                self.setTags(tagFile)
            else: 
                QMessageBox.information(self, "Message", "Tag File not selected");
                return False
            
            if videoSC:
                self.openFile(videoSC)
            else: 
                QMessageBox.information(self, "Message", "Error in Loading Media: Screen Capture video not found");
                return False;
            
            if videoExtra:
                self.openFile(videoExtra)
            else: 
                QMessageBox.information(self, "Message", "Error Loading Media: Face or hand's video Not Found");
                return False;
            
            if(self.fileBVP or self.fileEDA or self.fileEmotion):
                QMessageBox.information(self, "Message", "The files were loaded successfully");
                self.workloadPlot()
            else: 
                QMessageBox.information(self, "Message", "The Empatica E4 output or Emotion file not found");
                return False;

            
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
        self.PlotEmotion(self.fileEmotion)
        self.PlotHRFromBVP(self.fileBVP)
        self.PlotEda(self.fileEDA)
        
    def setConfigureTimeInterval(self,initialTime,endTime):
        self.timeTagInitial = initialTime;
        self.timeTagEnd = endTime;
        self.durantionSession = UnixTime().diffTimeStampTags(self.timeTagInitial, self.timeTagEnd)
        self.positionInitialSession = UnixTime().diffTimeStamp(timeVideo,self.timeTagInitial)*1000  
        self.loadingTimeProgressaBar(0,0)

    def loadingTimeProgressaBar(self,shiftLeft,shiftRight):
        ut = UnixTime();
        timeLeft =  ut.time_inc(self.timeTagInitial, shiftLeft)   
        timeRight = ut.time_reduce(self.timeTagEnd, shiftRight)     
        time =  '{} / {}'.format(timeLeft.strftime('%H:%M:%S'),timeRight.strftime('%H:%M:%S'))
        timeProgressBar.setText(time)
        timeLabel.setText(time)
        timeLabel2.setText(time) 
        return time;
    
    def setTags(self,path):
       
        global arrayTagsInitial;
        global arrayTagsEnd;

        try:
            sd = SourceData()
            tags = sd.LoadDataTags(path)
            if(len(tags) == 0):
                QMessageBox.information(self, "Message", "Does not exist tags");
                print("Does not exist tags");
                sys.exit(app.exec_()); 
            elif(len(tags) % 2 == 1  ):
                QMessageBox.information(self, "Message", "No match between Tags");
                print("No match between Tags");
                sys.exit(app.exec_());   
            elif (len(tags) % 2 == 0 ):
                arrayTagsInitial = tags[0::2];
                arrayTagsEnd = tags[1::2];
                self.setConfigureTimeInterval(arrayTagsInitial[0], arrayTagsEnd[0]);
            
        except:
            print("Erro during Loading Tags")
            sys.exit(app.exec_());
            
    def setTagVideo(self,path):
        global timeVideo;
        try:
            f = open(path, "r")
            timeVideo = float(f.read());           
        except:
            QMessageBox.information(self, "Message", "Error during Loading Video Tag");
            print("Erro during Loading Video Tag")
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
        
        
        #global pwTemp;
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
        
        
        #containerbottom = QtGui.QWidget()
        #containerbottom.setLayout(self.uiTimeBar())
        #splitter.addWidget(containerbottom)
        vbox.addWidget(splitter)
       
        
        return vbox
    
    def uiTimeBar(self):
               
        global positionRangeSlider;
        global timeProgressBar;    
        global btPlayer;        
        
        positionRangeSlider = QRangeSlider()
    
        positionRangeSlider.handle.setTextColor(150)
        positionRangeSlider.setFixedHeight(30)
        #positionRangeSlider.setFixedWidth(1000)
        
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
        #editsTimeLayout.addStretch()
        
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
        #bottom.layout().setContentsMargins(0, 0, 150, 150)
        #bottom.layout().setSpacing(1)
        
        splitter = QtGui.QSplitter(Qt.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(bottom)
        splitter.setSizes([100, 100])
        
        vbox = QtGui.QHBoxLayout()        
        
        vbox.addWidget(splitter)
        return vbox;
        
    def openFile(self,filename):        
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
            #print("positionChanged: Position:: %s" % position)
            #if(position>=self.positionInitialSession):
            positionRangeSlider.setStart(position)
            self.addLinearRegionInPlotWidget()

    def durationChanged(self, duration):  
        print("Duration Video in miliseconds: %s" % duration)
        self.setPositionInPlayer(self.positionInitialSession)
     
    def setPositionInPlayer(self, position):
        #print("setPositionInPlayer")
        self.mediaPlayer.setPosition(position)
        self.mediaPlayer2.setPosition(position)

    def handleError(self):
        #selfm.playButton.setEnabled(False)
        timeLabel.setText("Error: " + self.mediaPlayer.errorString())
            
    def updateRangerSlider(self):
        
        if(self.forceDisconnectRangeSlider):
            positionRangeSlider.endValueChanged.disconnect(self.eventChangeRightRangeValue)
            positionRangeSlider.startValueChanged.disconnect(self.eventChangeLeftRangeValue)
        
        global positionEndSession;
        positionEndSession = self.positionInitialSession+self.durantionSession;
        print("Duration session in miliseconds: %s" % (self.durantionSession))
        print("Initial Point: %s" % self.positionInitialSession)
        print("End Point: %s"% (positionEndSession))
        
        positionRangeSlider.setMin(self.positionInitialSession)
        positionRangeSlider.setMax(positionEndSession)
        positionRangeSlider.setRange(self.positionInitialSession,positionEndSession)   
        
        positionRangeSlider.endValueChanged.connect(self.eventChangeRightRangeValue)
        positionRangeSlider.startValueChanged.connect(self.eventChangeLeftRangeValue)
        self.forceDisconnectRangeSlider = False;                
    
    def eventChangeRightRangeValue(self, index):
        
        #print("eventChangeRightRangeValue : %s - %s" % (index,positionEndSession))
        if(index <= positionRangeSlider.start()):
            time = self.loadingTimeProgressaBar(0,0)
        elif ((positionEndSession-index) >= 0):
            time = self.loadingTimeProgressaBar(positionRangeSlider.start()-self.positionInitialSession,
                                                positionEndSession-index)
        if(positionRangeSlider.getMoved()):
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause();
                self.mediaPlayer2.pause()
            
            self.setPositionInPlayer(positionRangeSlider.start())
    
    def eventChangeLeftRangeValue(self, index):
        #print("eventChangeLeftRangeValue: %s - %s = %s" % (index,self.positionInitialSession,(index-self.positionInitialSession)))
        #print(positionRangeSlider.getMoved())
        if(index >= positionRangeSlider.end()):
            positionRangeSlider.setRange(self.positionInitialSession,positionEndSession)
            self.mediaPlayer.pause();
            self.mediaPlayer2.pause();
            self.setPositionInPlayer(self.positionInitialSession)
            time = self.loadingTimeProgressaBar(0,0)
            self.clearLinearRegion()

        elif(index-self.positionInitialSession >= 0):
            time = self.loadingTimeProgressaBar(index-self.positionInitialSession,
                                                positionEndSession-positionRangeSlider.end())
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
             
    def PlotEda(self, path):    
        print("PlotEda")   
       
        if(not path):
            return
        global ts;
        global metricsEDA;
        global plotEDA
            
     
        eda = EDAPeakDetectionScript()
        
        ts,raw_eda,filtered_eda,peaks,amp = eda.processEDA(path,
                                                       UnixTime().run(self.timeTagInitial),
                                                       UnixTime().run(self.timeTagEnd))
        metricsEDA  = ProcessingData().getMetricsEDA(raw_eda)
        normalize_data_eda = ProcessingData().normalize(filtered_eda)
        
        if(self.isCreatedPlotEda):
            plotEDA.clear()
            pwEDA.clear();
            #pwEDA.remove(kind='item') 
        plotEDA = pwEDA.plot(title='EDA Pike', pen='r')
        
        if(not self.isCreatedPlotEda):
            pwEDA.getPlotItem().addLegend(offset=(10,10))        
            pwEDA.addItem(pg.PlotDataItem(pen='r', name='GSR Value', antialias=False))
            pwEDA.addItem(pg.PlotDataItem(pen='b', name='GSR Peak', antialias=False))       
            pwEDA.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            pwEDA.setMouseEnabled(x=False, y=False)
            axis = DateAxis(orientation='bottom')
            axis.attachToPlotItem(pwEDA.getPlotItem())
            
        
        plotEDA.setData(x= ts,y=normalize_data_eda)        
        
        for peak in peaks:
            aux = str(float("{0:.2f}".format(filtered_eda[peak])));            
            l =  pwEDA.addLine(x=ts[peak], y=None, pen='b')
            label = pg.InfLineLabel(l, aux, position=float(aux), rotateAxis=(0,0), anchor=(2, 1))
        
        self.lrEDA = pg.LinearRegionItem([ts[0], ts[len(ts)-1]],bounds=[ts[0], ts[len(ts)-1]])  
        self.lrEDA.setZValue(-10)  
        pwEDA.addItem(self.lrEDA) 
        self.lrEDA.setRegion([ts[0],ts[0]])
        
        self.isCreatedPlotEda = True;
      
    def createHR(self, ts_hr, hr,filteredBVP):
        print("PlotHRFromBVP")    
        global plotHR
        
        timeHR = UnixTime().timeFrom(self.timeTagInitial, ts_hr)
        
        n_array = list(zip(timeHR,hr))        
        df = pd.DataFrame(n_array,columns=['timeHR','hr'])        
        df['timeHR'] = [datetime.fromtimestamp(ts) for ts in df['timeHR']]  
        df = df[(df['timeHR'] >= UnixTime().run(self.timeTagInitial)) & 
             (df['timeHR'] <= UnixTime().run(self.timeTagEnd))]       
        
        hr  = df['hr'].tolist()
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
            axis = DateAxis(orientation='bottom')
            axis.attachToPlotItem(pwHR.getPlotItem())
        
        normalize_data_hr = ProcessingData().normalize(df['hr'])
        plotHR.setData(x=timeHR, y=normalize_data_hr.tolist())
        
        plotHR = pwHR.plot(title="HRV", pen='b')
        if(not self.isCreatedPlotHR):
            pwHR.addItem(pg.PlotDataItem(pen='b', name='HRV Value', antialias=False))
        
        RRI_DF = EmpaticaHRV().getRRI(filteredBVP, self.timeTagInitial, 64)
        HRV_DF = EmpaticaHRV().getHRV(RRI_DF, np.mean(hr))
        timeHR = HRV_DF['Timestamp'].tolist()
        normalize_data_hrv = ProcessingData().normalize(HRV_DF['HRV'])
        plotHR.setData(x=timeHR, y=normalize_data_hrv.tolist())
       
        self.lrHR = pg.LinearRegionItem([timeHR[0], timeHR[len(timeHR)-1]],
                                        bounds=[timeHR[0], timeHR[len(timeHR)-1]])  
        self.lrHR.setZValue(-10)  
        pwHR.addItem(self.lrHR)
        self.lrHR.setRegion([timeHR[0],timeHR[0]])
        
        self.isCreatedPlotHR = True;
    
    def PlotHRFromBVP(self, path):
        if(not path):#If not exist file then dont loading
            return;
       
        sd = SourceData()
        count, data, startTime, endTime, samplingRate = sd.LoadDataBVP(path)       
        
        filteredBVP, ts_hr, hr = ProcessingData().ProcessedBVPDataE4(data) 
       
        ut = UnixTime();        
        endTime, tsBVP = ut.time_array(self.timeTagInitial, count, samplingRate)
                
        n_array = list(zip(tsBVP,filteredBVP))        
        df = pd.DataFrame(n_array,columns=['tsBVP','filteredBVP'])        
        df['tsBVP'] = [datetime.fromtimestamp(ts) for ts in df['tsBVP']]
        #Cut in time
        df = df[(df['tsBVP'] >=  UnixTime().run(self.timeTagInitial)) & 
                (df['tsBVP'] <= UnixTime().run(self.timeTagEnd))]       
        filteredBVP  = df['filteredBVP'].tolist()       
       
        self.createHR(ts_hr, hr,filteredBVP)
   
    def PlotEmotion(self,url):
        print("PlotEmotion")
        if(not url):#If not exist file then dont loading
            return;
            
        try:
            
            js = SourceData()
            df = js.LoadDataFacialExpression(indexSession=None, path=url);
                                 
            d1 = list(zip(df['Time'],df['Happiness'],df['Sadness'],df['Anger'],
                                     df['Surprise'],df['Fear'],df['Disgust']))        
            dataframe = pd.DataFrame(d1,columns=['tsEmotion','Happiness','Sadness',
                                                 "Anger","Surprise","Fear",'Disgust'])   
            
            #Cut in time
            dataframe = dataframe[(dataframe['tsEmotion'] >=  UnixTime().run(self.timeTagInitial)) & 
                                  (dataframe['tsEmotion'] <= UnixTime().run(self.timeTagEnd))]
    
            array1  = dataframe['Happiness'].tolist()
            array2  = dataframe['Sadness'].tolist()
            array3  = dataframe['Anger'].tolist()
            array4  = dataframe['Surprise'].tolist()
            array5  = dataframe['Fear'].tolist()
            array6  = dataframe['Disgust'].tolist()
            
            tsEmotion = [datetime.timestamp(dt) for dt in dataframe['tsEmotion']]
           
            if(self.isCreatedPlotEmotion):                   
                pwEmotion.clear();
                
            if(not self.isCreatedPlotEmotion):
                pwEmotion.addLegend((50,60), offset=(30,30)) 
                pwEmotion.addItem(pg.PlotDataItem(pen='b', name='Happiness', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='c', name='Sadness', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='y', name='Anger', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='g', name='Surprise', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='k', name='Fear', antialias=False))
                pwEmotion.addItem(pg.PlotDataItem(pen='m', name='Disgust', antialias=False))
                pwEmotion.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
                axis = DateAxis(orientation='bottom')
                axis.attachToPlotItem(pwEmotion.getPlotItem()) 
            #b: blue,g: green,r: red,c: cyan,m: magenta,y: yellow,k: black,w: white
            for nameEmotion in self._listEmotion:
                if(nameEmotion == 'Happiness'):
                    plotEmotion = pwEmotion.plot(title='Happiness',pen='b')                           
                    plotEmotion.setData(x=tsEmotion, y=array1)  
                if(nameEmotion == 'Sadness'):  
                    plotEmotion = pwEmotion.plot(title='Sadness',pen='c')
                    plotEmotion.setData(x=tsEmotion, y=array2)
                if(nameEmotion == 'Anger'): 
                    plotEmotion = pwEmotion.plot(title='Anger',pen='y')
                    plotEmotion.setData(x=tsEmotion, y=array3)
                if(nameEmotion == 'Surprise'):
                    plotEmotion = pwEmotion.plot(title='Surprise',pen='g')
                    plotEmotion.setData(x=tsEmotion, y=array4)
                if(nameEmotion == 'Fear'):
                    plotEmotion = pwEmotion.plot(title='Fear',pen='k')
                    plotEmotion.setData(x=tsEmotion, y=array5)
                if(nameEmotion == 'Disgust'):
                    plotEmotion = pwEmotion.plot(title='Disgust',pen='m')
                    plotEmotion.setData(x=tsEmotion, y=array6)           

            #Plot was created
            self.lrEmotion = pg.LinearRegionItem([tsEmotion[0], tsEmotion[len(tsEmotion)-1]],
                                                 bounds=[tsEmotion[0], tsEmotion[len(tsEmotion)-1]])  
            self.lrEmotion.setZValue(-10)  
            pwEmotion.addItem(self.lrEmotion) 
            self.lrEmotion.setRegion([tsEmotion[0],tsEmotion[0]])
            
            self.isCreatedPlotEmotion= True;
            self._listEmotion = ['Happiness','Sadness','Anger',
                      'Fear','Surprise','Disgust']
            return True;
        except: 
            print("Oops!",sys.exc_info()[0],"occured.")
            print("Erro in PlotEmotion")
            return False;
 
    def clearLinearRegion(self):
        ut = UnixTime();
        indexInitial = datetime.timestamp(ut.time_inc(self.timeTagInitial,0))
        self.printRegion(indexInitial,indexInitial)        
            
    def addLinearRegionInPlotWidget(self):
        if self.mediaPlayer.state() != QMediaPlayer.PausedState:
            ut = UnixTime();
            indexInitial = datetime.timestamp(ut.time_inc(self.timeTagInitial, 
                                                          positionRangeSlider.start()-self.positionInitialSession))
            indexEnd =  datetime.timestamp(ut.time_reduce(self.timeTagEnd, 
                                                          positionEndSession-positionRangeSlider.end()))
            
            self.printRegion(indexInitial, indexEnd) 
                
    def printRegion(self,indexInitial,indexEnd):
        try:
            if(self.isCreatedPlotEda):
                self.lrEDA.setRegion([indexInitial,indexEnd])
            if(self.isCreatedPlotHR):
                self.lrHR.setRegion([indexInitial,indexEnd]) 
            if(self.isCreatedPlotEmotion):            
                self.lrEmotion.setRegion([indexInitial,indexEnd])         
        except ValueError:
            print('Non-numeric data found in the file.')           
        except EOFError:
            print('Why did you do an EOF on me?')      
        except:            
            print('Linear Region null') 
    
    def paintEvent(self, event):

        qp = QtGui.QPainter()
        qp.begin(self)
        #print("XXX")
        qp.end()
    
    def is_video_file(self,filename):
        video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.avi', '.dv-avi', 
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2' )

        if filename.endswith(video_file_extensions):
            return True
        return False;
    
    

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    
    flow = FlowChartGame('testeee')
    flow.setGeometry(10, 10, 1000, 800)
    flow.setWindowTitle("Project Game Data Explorer (PGD Ex)")
    flow.show()

    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
    sys.exit(app.exec_())
        
        
        