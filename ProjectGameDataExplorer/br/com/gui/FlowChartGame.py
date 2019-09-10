
'''
Created on 15 de abr de 2019

@author: eltonss
'''

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from pyqtgraph.Qt import QtCore, QtGui, PYSIDE
import pandas as pd
import numpy as np
import sys, os

import matplotlib
from br.com.analytic.reduceFeatures.ReduceEmotion import ReduceEmotion
from br.com.gui.TableView import TableView
from theano.ifelse import ifelse
from numba.tests.test_nested_calls import star
matplotlib.use('Agg')

from br.com.util.SourceData import SourceData
from br.com.util.ProcessingData import ProcessingData

from br.com.util.EmpaticaHRV import EmpaticaHRV
from br.com.util.EDAPeakDetectionScript import EDAPeakDetectionScript
matplotlib.use('Agg')

from PyQt5.QtCore import  pyqtSignal
from PyQt5.QtCore import  QTime
from PyQt5.QtGui import QIcon, QPixmap, QBrush, QColor, QSlider
from PyQt5.QtWidgets import QWidget,  QSplitter,QStyle
from PyQt5.QtCore import  Qt, QUrl
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from br.com.gui.qrangeslider import QRangeSlider
import pyqtgraph as pg
from br.com.util.UnixTime import UnixTime
from br.com.gui.DateAxis import DateAxis
from datetime import datetime
from br.com.gui.TimeWidget import TimeWidget
matplotlib.use('Agg')


class FlowChartGame(QtGui.QMainWindow):
    
    
    def __init__(self, buffer_size=0, data_buffer=[], graph_title="", parent=None):
        super(FlowChartGame, self).__init__(parent)               

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        #global  timer;
        #timer = QtCore.QTimer(self)
        
        self.createMediaPlayer();
        self.windowPlots()
        
    def createMediaPlayer(self):
        self.ShowVideo = True;
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
        self.ShowVideo = False;
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
        
        
        open = QtGui.QAction("Open E4 Data File with Video", self)
        open.setShortcut("Ctrl+O")
        openSV = QtGui.QAction("Open E4 Data File without Video", self)
        openSV.setShortcut("Ctrl+A")
        file.addAction(open)
        file.addAction(openSV)
                    
        quit = QtGui.QAction("Quit", self)
        quit.setShortcut("Ctrl+Q") 
        
        restart = QtGui.QAction("Restart", self)
        restart.setShortcut("Ctrl+R") 
        
        file.addAction(restart)
        file.addAction(quit)
        
        #tools.addAction(resetLG)
        tools.addAction(resetPB)
        tools.addAction(metricEDA)
        file.triggered[QtGui.QAction].connect(self.processtrigger)
        tools.triggered[QtGui.QAction].connect(self.processTools)

        self.setLayout(layout)
    
    def processTools(self,q):
        if(q.text() == "EDA Metrics"):
           
            data ={'Metrics':['Mean','Median','Max','Var','Std_dev','Kurtosis','skewness'],
           'Value': [str(metricsEDA['mean']),str(metricsEDA['median']),str(metricsEDA['max']),
                    str(metricsEDA['var']),str(metricsEDA['std_dev']),
                    str(metricsEDA['kurtosis']),str(metricsEDA['skewness'])]}
          
            self.tv = TableView(data,7,2)
            
            self.tv.show()
           
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
            self.ShowVideo = True;            
            self.loadingVisualization(True)
        elif (q.text() == 'Open E4 Data File without Video'):
            self.destroyMediaPlayer()            
            self.loadingVisualization(False)
        elif (q.text() == "Quit"):
            sys.exit(app.exec_())
        elif (q.text() == "Restart"):
            os.execl(sys.executable, sys.executable, *sys.argv) 
    
    def loadingVisualization(self,enableVideo):
        if(self.plottingGraphics(enableVideo)):
            self.updateRangerSlider()            
            positionRangeSlider.endValueChanged.connect(self.eventChangeRightRangeValue)
            positionRangeSlider.startValueChanged.connect(self.eventChangeLeftRangeValue)

    def plottingGraphics(self, showVideo=None):
               
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
            videoWC = ""
            fileBVP = ""
            fileEDA = ""
            fileEmotion = ""
            tagFileVideo = ""
            tagFile = ""
            for file in filenames:
                filename = os.path.basename(file)               
                
                if(not(self.is_video_file(filename) or filename.endswith('.csv'))):
                    QMessageBox.information(self, "Message", "No appropriate file Located");
                    return False;
                
                if(self.is_video_file(filename) and ("SC") in filename):
                    videoSC = file;
                elif(self.is_video_file(filename) and ("WC") in filename):
                    videoWC = file;   
                elif(filename == 'timevideo.csv'):                    
                    tagFileVideo = file;
                elif(filename == 'tags.csv'):                    
                    tagFile = file;
                elif filename == 'BVP.csv':
                    fileBVP = file;
                
                elif filename == 'EDA.csv':
                    fileEDA = file; 
                elif ("EMOCAO") in filename:
                    fileEmotion = file; 
                               
            
            if tagFile:                
                self.setTags(tagFile)
            else: 
                QMessageBox.information(self, "Message", "Tag File not selected");
                return False
            
            if showVideo: 
                if tagFileVideo:
                    self.setTagVideo(tagFileVideo);
                else: 
                    QMessageBox.information(self, "Message", "Tag Video not selected");
                    return False;
            else:  tagFileVideo = tagFile;#Same Point. Not difference on Time 
                    
            if showVideo:
                if videoSC:
                    self.openFile(videoSC)
                else: 
                    QMessageBox.information(self, "Message", "Error in Loading Media: Screen Capture video not found");
                    return False;
            if showVideo:
                if videoWC:
                    self.openFile(videoWC)
                else: 
                    QMessageBox.information(self, "Message", "Error in Loading Media: video not found");
                    return False;
            
            if(fileBVP or fileEDA or fileEmotion):
                QMessageBox.information(self, "Message", "The files were loaded successfully");
            else: 
                QMessageBox.information(self, "Message", "The Empatica E4 output or Emotion file not found");
                return False;

            if fileEmotion:
                self.PlotEmotion(fileEmotion)
            if fileBVP:
                self.PlotHRFromBVP(fileBVP)
            if fileEDA:
                self.PlotEda(fileEDA)
            
            

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
     
    def setTags(self,path):
       
        global timeTagEnd;
        global timeTagInitial;
        global durantionSession;
        

        try:
            sd = SourceData()
            tags = sd.LoadDataTags(path)
            if(len(tags) % 2 == 1  ):
                QMessageBox.information(self, "Message", "There is only one Tag");
                print("There is only one Tag");
                sys.exit(app.exec_());   
            elif (len(tags) % 2 == 0 ):
                timeTagsInitial = tags[0::2];
                timeTagsEnd = tags[1::2];
                #Popup para escolher Tags
                
                timeTagInitial = timeTagsInitial[0];
                timeTagEnd = timeTagsEnd[0];
                durantionSession = UnixTime().diffTimeStampTags(timeTagInitial, timeTagEnd)    
                self.loadingTimeProgressaBar(0,0)
                
            else: 
                QMessageBox.information(self, "Message", "There should be two Tags");
                print("There should be two Tags");
                sys.exit(app.exec_()); 
        except:
            print("Erro during Loading Tags")
            sys.exit(app.exec_());
    
    def loadingTimeProgressaBar(self,shiftLeft,shiftRight):
        ut = UnixTime();
        timeLeft =  ut.time_inc(timeTagInitial, shiftLeft)   
        timeRight = ut.time_reduce(timeTagEnd, shiftRight)     
        time =  '{} / {}'.format(timeLeft.strftime('%H:%M:%S'),timeRight.strftime('%H:%M:%S'))
        timeProgressBar.setText(time)
        return time;
    
    def setTagVideo(self,path):
        global timeVideo;
        global positionInitialSession
        try:
            f = open(path, "r")
            timeVideo = float(f.read());
            positionInitialSession = UnixTime().diffTimeStamp(timeVideo,timeTagInitial)*1000  
            
        except:
            QMessageBox.information(self, "Message", "Erro during Loading Video Tag");

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
        
        gridl = QtGui.QGridLayout()
        gridl.addWidget(self.splitter)
        return gridl;
                 
    def uiPanelPlot(self):
        global pwEDA;        
        global isCreatedPlotEda;
        global pwHR;
        global isCreatedPlotHR;
        global pwEmotion;
        global isCreatedPlotBVP;
        self.isCreatedPlotEda = False;
        self.isCreatedPlotHR = False;
        self.isCreatedPlotBVP = False;
        
        
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
        
        
        containerbottom = QtGui.QWidget()
        containerbottom.setLayout(self.uiTimeBar())
        splitter.addWidget(containerbottom)
        vbox.addWidget(splitter)
       
        
        return vbox
    
    def uiTimeBar(self):
               
        global positionRangeSlider;
        global timeProgressBar;
    
        global btPlayer;
        
        
        positionRangeSlider = QRangeSlider()
    
        positionRangeSlider.handle.setTextColor(150)
        positionRangeSlider.setFixedHeight(30)
        
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
            if  self.ShowVideo:
                if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                    btPlayer.setIcon(
                            self.style().standardIcon(QStyle.SP_MediaPause))
                else:
                    btPlayer.setIcon(
                            self.style().standardIcon(QStyle.SP_MediaPlay))     
        except: 
            print("No media player");
            
    def changedLabelTime(self, str):  
        if  self.ShowVideo:     
            timeLabel.setText("Time: " + str)
            timeLabel2.setText("Time: " + str)   
           
    def positionChanged(self, position):       
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            #print("positionChanged: Position:: %s" % position)
            #if(position>=positionInitialSession):
            positionRangeSlider.setStart(position)
            self.addLinearRegionInPlotWidget()

    def durationChanged(self, duration):  
        print("Duration Video in miliseconds: %s" % duration)
        self.setPositionInPlayer(positionInitialSession)
     
    def setPositionInPlayer(self, position):
        #print("setPositionInPlayer")
        self.mediaPlayer.setPosition(position)
        self.mediaPlayer2.setPosition(position)

    def handleError(self):
        #selfm.playButton.setEnabled(False)
        timeLabel.setText("Error: " + self.mediaPlayer.errorString())
            
    def updateRangerSlider(self):
        global positionEndSession;
        positionEndSession = positionInitialSession+durantionSession;
        print("Duration session in miliseconds: %s" % (durantionSession))
        print("Initial Point: %s" % positionInitialSession)
        print("End Point: %s"% (positionEndSession))
        positionRangeSlider.setMin(positionInitialSession)
        positionRangeSlider.setMax(positionEndSession)
        positionRangeSlider.setRange(positionInitialSession,positionEndSession)
    
                                              
    def eventChangeRightRangeValue(self, index):
        
        #print("eventChangeRightRangeValue : %s - %s" % (index,positionEndSession))
        if(index <= positionRangeSlider.start()):
            time = self.loadingTimeProgressaBar(0,0)
            self.changedLabelTime(time)
        elif ((positionEndSession-index) >= 0):
            time = self.loadingTimeProgressaBar(positionRangeSlider.start()-positionInitialSession,
                                                positionEndSession-index)
            self.changedLabelTime(time)
        if(positionRangeSlider.getMoved()):
            if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
                self.mediaPlayer.pause();
                self.mediaPlayer2.pause()
            
            self.setPositionInPlayer(positionRangeSlider.start())
         
    
    def eventChangeLeftRangeValue(self, index):
        #print("eventChangeLeftRangeValue: %s - %s = %s" % (index,positionInitialSession,(index-positionInitialSession)))
        #print(positionRangeSlider.getMoved())
        if(index >= positionRangeSlider.end()):
            positionRangeSlider.setRange(positionInitialSession,positionEndSession)
            self.mediaPlayer.pause();
            self.mediaPlayer2.pause();
            self.setPositionInPlayer(positionInitialSession)
            time = self.loadingTimeProgressaBar(0,0)
            self.changedLabelTime(time)
            self.clearLinearRegion()

        elif(index-positionInitialSession >= 0):
            time = self.loadingTimeProgressaBar(index-positionInitialSession,
                                                positionEndSession-positionRangeSlider.end())
            self.changedLabelTime(time)
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
       
        global ts;
        global metricsEDA;
        self.isCreatedPlotEda = True;

     
        eda = EDAPeakDetectionScript()
        
        ts,raw_eda,filtered_eda,peaks,amp = eda.processEDA(path,
                                                       UnixTime().run(timeTagInitial),
                                                       UnixTime().run(timeTagEnd))
        metricsEDA  = ProcessingData().getMetricsEDA(raw_eda)
        
        plot = pwEDA.plot(title='EDA Pike', pen='r')
        pwEDA.getPlotItem().addLegend(offset=(10,10))
        
        pwEDA.addItem(pg.PlotDataItem(pen='r', name='GSR Value', antialias=False))
        pwEDA.addItem(pg.PlotDataItem(pen='b', name='GSR Peak', antialias=False))       
        pwEDA.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)

        axis = DateAxis(orientation='bottom')
        axis.attachToPlotItem(pwEDA.getPlotItem())
        normalize_data_eda = ProcessingData().normalize(filtered_eda)
        plot.setData(x= ts,y=normalize_data_eda)
        
        pwEDA.setMouseEnabled(x=False, y=False)
        for peak in peaks:
            aux = str(float("{0:.2f}".format(filtered_eda[peak])));            
            l =  pwEDA.addLine(x=ts[peak], y=None, pen='b')
            label = pg.InfLineLabel(l, aux, position=float(aux), rotateAxis=(0,0), anchor=(2, 1))
        
        self.lrEDA = pg.LinearRegionItem([ts[0], ts[len(ts)-1]],bounds=[ts[0], ts[len(ts)-1]])  
        self.lrEDA.setZValue(-10)  
        pwEDA.addItem(self.lrEDA) 
        self.lrEDA.setRegion([ts[0],ts[0]])
      
    def createHR(self, ts_hr, hr,filteredBVP):
        print("PlotHRFromBVP")    
        self.isCreatedPlotHR = True;
        
        timeHR = UnixTime().timeFrom(timeTagInitial, ts_hr)
        
        n_array = list(zip(timeHR,hr))        
        df = pd.DataFrame(n_array,columns=['timeHR','hr'])        
        df['timeHR'] = [datetime.fromtimestamp(ts) for ts in df['timeHR']]  
        df = df[(df['timeHR'] >= UnixTime().run(timeTagInitial)) & 
             (df['timeHR'] <= UnixTime().run(timeTagEnd))]       
        
        hr  = df['hr'].tolist()
        timeHR = [datetime.timestamp(dt) for dt in df['timeHR']]
       
        #plot = pwHR.plot(title="HR", pen='r',symbol='o')
        plot = pwHR.plot(title="HR", pen='r')
        pwHR.getPlotItem().addLegend()
        pwHR.addItem(pg.PlotDataItem(pen='r', name='HR Value', antialias=False))
        pwHR.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        
        axis = DateAxis(orientation='bottom')
        axis.attachToPlotItem(pwHR.getPlotItem())
        
        normalize_data_hr = ProcessingData().normalize(df['hr'])
        plot.setData(x=timeHR, y=normalize_data_hr.tolist())
        pwHR.setMouseEnabled(x=False, y=False)
        
        plot = pwHR.plot(title="HRV", pen='b')
        pwHR.addItem(pg.PlotDataItem(pen='b', name='HRV Value', antialias=False))
        
        RRI_DF = EmpaticaHRV().getRRI(filteredBVP, timeTagInitial, 64)
        HRV_DF = EmpaticaHRV().getHRV(RRI_DF, np.mean(hr))
        timeHR = HRV_DF['Timestamp'].tolist()
        normalize_data_hrv = ProcessingData().normalize(HRV_DF['HRV'])
        plot.setData(x=timeHR, y=normalize_data_hrv.tolist())
       
        self.lrHR = pg.LinearRegionItem([timeHR[0], timeHR[len(timeHR)-1]],
                                        bounds=[timeHR[0], timeHR[len(timeHR)-1]])  
        self.lrHR.setZValue(-10)  
        pwHR.addItem(self.lrHR)
        self.lrHR.setRegion([timeHR[0],timeHR[0]])
    
    def PlotHRFromBVP(self, path):
        
       
        sd = SourceData()
        count, data, startTime, endTime, samplingRate = sd.LoadDataBVP(path)       
        
        filteredBVP, ts_hr, hr = ProcessingData().ProcessedBVPDataE4(data) 
       
        ut = UnixTime();        
        endTime, tsBVP = ut.time_array(timeTagInitial, count, samplingRate)
                
        n_array = list(zip(tsBVP,filteredBVP))        
        df = pd.DataFrame(n_array,columns=['tsBVP','filteredBVP'])        
        df['tsBVP'] = [datetime.fromtimestamp(ts) for ts in df['tsBVP']]
        #Cut in time
        df = df[(df['tsBVP'] >=  UnixTime().run(timeTagInitial)) & 
                (df['tsBVP'] <= UnixTime().run(timeTagEnd))]       
        filteredBVP  = df['filteredBVP'].tolist()       
       
        self.createHR(ts_hr, hr,filteredBVP)
   
    def PlotEmotion(self,url):
        
        try:
            js = SourceData()
            df = js.LoadDataFacialExpression(indexSession=None, path=url);
            
            df1 = df[['Happiness','Surprise','Fear']].copy()
            df2 = df[['Sadness','Anger','Disgust']].copy()
           
            var = ReduceEmotion()
            arrayPCA_Positive = var.runPCA(df1)
            arrayPCA_Negative = var.runPCA(df2)  
                     
            d1 = list(zip(df['Time'],arrayPCA_Positive,arrayPCA_Negative))        
            dataframe = pd.DataFrame(d1,columns=['tsEmotion','arrayPCA1','arrayPCA2'])   
            
            #Cut in time
            dataframe = dataframe[(dataframe['tsEmotion'] >=  UnixTime().run(timeTagInitial)) & 
                                  (dataframe['tsEmotion'] <= UnixTime().run(timeTagEnd))]
    
            arrayPCA1  = dataframe['arrayPCA1'].tolist()
            arrayPCA2  = dataframe['arrayPCA2'].tolist()
            tsEmotion = [datetime.timestamp(dt) for dt in dataframe['tsEmotion']]
           
        
            plot = pwEmotion.plot(title="Emotion", pen='b')
            pwEmotion.getPlotItem().addLegend()
            
            pwEmotion.addItem(pg.PlotDataItem(pen='b', name='Positive Value', antialias=False))
            pwEmotion.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
            
            axis = DateAxis(orientation='bottom')
            axis.attachToPlotItem(pwEmotion.getPlotItem())
            
            plot.setData(x=tsEmotion, y=arrayPCA1)
            plot = pwEmotion.plot(title="Emotion", pen='r')
            pwEmotion.addItem(pg.PlotDataItem(pen='r', name='Negative Value', antialias=False))
            plot.setData(x=tsEmotion, y=arrayPCA2)
            
            #Plot was created
            self.isCreatedPlotBVP = True;
            self.lrBVP = pg.LinearRegionItem([tsEmotion[0], tsEmotion[len(tsEmotion)-1]],bounds=[tsEmotion[0], tsEmotion[len(tsEmotion)-1]])  
            self.lrBVP.setZValue(-10)  
            pwEmotion.addItem(self.lrBVP) 
            self.lrBVP.setRegion([tsEmotion[0],tsEmotion[0]])
            return True;
        except: 
            print("Oops!",sys.exc_info()[0],"occured.")
            print("Erro in PlotEmotion")
            return False;
 
    def clearLinearRegion(self):
        ut = UnixTime();

        indexInitial = datetime.timestamp(ut.time_inc(timeTagInitial,0))
             
        if(self.isCreatedPlotEda):          
            #pwEDA.removeItem(self.lrEDA)           
            self.lrEDA.setRegion([indexInitial,indexInitial])
        if(self.isCreatedPlotBVP):
            self.lrBVP.setRegion([indexInitial,indexInitial])  
        if(self.isCreatedPlotHR):
            self.lrHR.setRegion([indexInitial,indexInitial]) 
            
    def addLinearRegionInPlotWidget(self):
        if self.mediaPlayer.state() != QMediaPlayer.PausedState:

            ut = UnixTime();
            indexInitial = datetime.timestamp(ut.time_inc(timeTagInitial, 
                                                          positionRangeSlider.start()-positionInitialSession))
            indexEnd =  datetime.timestamp(ut.time_reduce(timeTagEnd, positionEndSession-positionRangeSlider.end()))
            if(self.isCreatedPlotEda):          
                self.lrEDA.setRegion([indexInitial,indexEnd])
            if(self.isCreatedPlotBVP):
                self.lrBVP.setRegion([indexInitial,indexEnd])            
            if(self.isCreatedPlotHR):
                self.lrHR.setRegion([indexInitial,indexEnd]) 
       
    def is_video_file(self,filename):
        video_file_extensions = (
'.264', '.3g2', '.3gp', '.3gp2', '.3gpp', '.3gpp2', '.3mm', '.3p2', '.60d', '.787', '.avi', '.dv-avi', 
'.mjpg', '.mk3d', '.mkv', '.mmv', '.mnv', '.mob', '.mod', '.modd', '.moff', '.moi', '.moov', '.mov', '.movie', '.mp21',
'.mp21', '.mp2v', '.mp4', '.mp4v', '.mpe', '.mpeg', '.mpeg1', '.mpeg4', '.mpf', '.mpg', '.mpg2' )

        if filename.endswith(video_file_extensions):
            return True
        return False;
    
    def getTimeDetails(self, duration):
        
        hours = duration.hour
        minutes = duration.minute                
        seconds = duration.second 
        return (hours, minutes, seconds) 

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    
    flow = FlowChartGame('testeee')
    flow.setGeometry(10, 10, 1000, 800)
    flow.setWindowTitle("Project Game Data Explorer (PGD Ex)")
    flow.show()

    if sys.flags.interactive != 1 or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(app.exec_())
       
        
        
        