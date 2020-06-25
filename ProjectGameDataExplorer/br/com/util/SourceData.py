from builtins import int
from datetime import datetime, timedelta
import glob

import numpy as np
import pandas as pd
from ProjectGameDataExplorer.br.com.util.E3Data import E3Data


class SourceData:
       
    def __init__(self):
        self.E3Data = E3Data;
        print('Constructor SourceData')
 
    def LoadDataFacialExpression(self, path, indexSession=None):
        
        if(indexSession != None):
            source = 'EMOCAO_*.csv';            
            url = path.format(indexSession, indexSession, source)
            file_ = glob.glob(url)[0]
        else: file_ = path;    
        list_ = [];
        
        df = pd.read_csv(file_, index_col=None, header=0)
        list_.append(df)
            
        frame = pd.concat(list_)
            
        df = pd.DataFrame(columns=['Time', 'Neutral', 'Happiness', 'Sadness',
                                   'Anger', 'Fear', 'Surprise', 'Disgust']);    
        
        dates_list = [];
        for d in frame['Time']:
            dates_list.append(datetime.strptime(d, '%d/%m/%Y %H:%M:%S.%f')) 
            
        # df['Time'] = ut.getTimeElapsed(dates_list); 
        df['Time'] = (dates_list); 
        df['Neutral'] = ((np.array(frame['neutral']).astype(float))) ;
        df['Happiness'] = (np.array(frame['happiness']).astype(float)) ;
        df['Sadness'] = (np.array(frame['sadness']).astype(float)) ;
        df['Anger'] = (np.array(frame['anger']).astype(float)) ;
        df['Fear'] = (np.array(frame['fear']).astype(float)) ;
        df['Surprise'] = (np.array(frame['surprise']).astype(float)) ;
        df['Disgust'] = (np.array(frame['disgust']).astype(float)) ;
        
        return df;
    
    def LoadDataEDA(self, path):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "EDA") 
        print("Load EDA Data")
        index = np.arange(len (e3data.data))
        dataset_array = []    
        for item in e3data.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, e3data.startTime, e3data.getEndTime(), e3data.samplingRate); 
    
    def LoadDataEDASlice(self, path, startTime, endTime):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "EDA") 
        print("Load Data EDA Slice")
        slice = e3data.getSlide(startTime, endTime)
        index = np.arange(len (slice.data))
        dataset_array = []    
        for item in slice.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, slice.samplingRate); 
    
    def LoadDataHR(self, path):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "HR") 
        print("Load HR Data")
        index = np.arange(len (e3data.data))
        dataset_array = []    
        for item in e3data.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, e3data.startTime, e3data.getEndTime(), e3data.samplingRate); 
    
    def LoadDataHRSlice(self, path, startTime, endTime):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "HR") 
        index = np.arange(len (e3data.data))
        print(e3data.data)        
        slice = e3data.getSlide(startTime, endTime)
        print("Load Data HR Slice")
        print(slice.data)
        index = np.arange(len (slice.data))
        dataset_array = []    
        for item in slice.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, slice.samplingRate);   
    
    def LoadDataBVP(self, path):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "BVP") 
        print("Load BVP Data")
        index = np.arange(len (e3data.data))
        dataset_array = []    
        
        for item in e3data.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, e3data.startTime, e3data.getEndTime(), e3data.samplingRate); 
    
    def LoadDataBVPSlice(self, path, startTime, endTime):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "BVP") 
        print("Load BVP Data Slice")
        slice = e3data.getSlide(startTime, endTime)
        index = np.arange(len (slice.data))
        dataset_array = []    
        for item in slice.data:
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, slice.samplingRate);      
    
    def LoadDataTemp(self, path):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "TEMP") 
        print("Load Temp Data")
        index = np.arange(len (e3data.data))
        dataset_array = []    
        for item in e3data.data:
            
            dataset_array.append(float(item[0])) 
        
        return (index, dataset_array, e3data.startTime, e3data.getEndTime(), e3data.samplingRate); 
    
    def LoadDataTags(self, path):
        e3data = self.E3Data.newE3DataFromFilePath(E3Data, path, "TAGS") 
        print("Load TAGS Data")
        index = np.arange(len (e3data.data))
        dataset_array = []    
        for item in e3data.data:
            
            dataset_array.append((item[0])) 
        return (dataset_array); 
    
