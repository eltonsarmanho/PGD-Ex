'''
Created on 20 de dez de 2019

@author: elton
'''
from collections import Counter

from matplotlib.pyplot import plot
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from sklearn.metrics.scorer import SCORERS
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import statsmodels.api as sm
from pandas.plotting._matplotlib.misc import scatter_matrix
from pandas.plotting import parallel_coordinates
class FeaturesBiosignal(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def loadData(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/dataset.csv", sep=',') 
       
        
        self.datasetPX = self.dataset
        self.replaceValue()     
        
        self.datasetPX['Experience'] = 'X'
        self.filter("Tristeza",3,"Tristeza")
        self.filter("Calma",3,"Calma")
        self.filter("Felicidade",3,"Felicidade")
        self.filter("Raiva",2,"Raiva")
        self.filter("Desgosto",3,"Raiva")
        
        self.datasetPX.to_csv("/home/elton/Desktop/datasetUpdate.csv", index=False,sep=',', encoding='utf-8')
  
        
    
    def showPlot(self):
        self.datasetPX = self.datasetPX.drop('Player', 1)
        self.datasetPX = self.datasetPX.drop('Session', 1)
        self.datasetPX = self.datasetPX.drop('Interval Initial', 1)
        self.datasetPX = self.datasetPX.drop('Interval Final', 1)
        self.datasetPX = self.datasetPX.drop('Rater 1', 1)
        self.datasetPX = self.datasetPX.drop('Rater 2', 1)
        self.datasetPX = self.datasetPX.drop('Rater 3', 1)
        self.datasetPX = self.datasetPX.drop('Rater 4', 1)
        self.datasetPX = self.datasetPX.drop('Rater 5', 1)
        self.datasetPX = self.datasetPX.drop('Rater 6', 1)
        self.datasetPX = self.datasetPX.drop('Event', 1)   
        
        self.datasetPX = self.datasetPX.drop('NegativeEmotion_maximum',1)
        self.datasetPX = self.datasetPX.drop('NegativeEmotion_minimum',1)
        self.datasetPX = self.datasetPX.drop('NegativeEmotion_mean',1)
        self.datasetPX = self.datasetPX.drop('NegativeEmotion_range',1)
        self.datasetPX = self.datasetPX.drop('NegativeEmotion_sd',1)
        self.datasetPX = self.datasetPX.drop('PositiveEmotion_maximum',1)
        self.datasetPX = self.datasetPX.drop('PositiveEmotion_minimum',1)
        self.datasetPX = self.datasetPX.drop('PositiveEmotion_mean',1)
        self.datasetPX = self.datasetPX.drop('PositiveEmotion_range',1)
        self.datasetPX = self.datasetPX.drop('PositiveEmotion_sd',1)
        
        #self.datasetPX = self.datasetPX.drop('pha_Mean',1)
        #self.datasetPX = self.datasetPX.drop('pha_StDev',1)
        #self.datasetPX = self.datasetPX.drop('pha_Range',1)
        #self.datasetPX = self.datasetPX.drop('pha_PeaksMax',1)
        #self.datasetPX = self.datasetPX.drop('pha_PeaksMin',1)
        #self.datasetPX = self.datasetPX.drop('pha_PeaksMean',1)
        #self.datasetPX = self.datasetPX.drop('pha_PeaksNum',1)
        #self.datasetPX = self.datasetPX.drop('pha_DurationMean',1)
        #self.datasetPX = self.datasetPX.drop('pha_SlopeMean',1)
        #self.datasetPX = self.datasetPX.drop('pha_AUC',1)
        
        self.datasetPX = self.datasetPX.drop('IBI_RMSSD',1)
        self.datasetPX = self.datasetPX.drop('IBI_SDSD',1)
        self.datasetPX = self.datasetPX.drop('IBI_Mean',1)
        self.datasetPX = self.datasetPX.drop('IBI_RRstd',1)
        self.datasetPX = self.datasetPX.drop('IBI_Median',1)
        self.datasetPX = self.datasetPX.drop('IBI_pnn10',1)
        self.datasetPX = self.datasetPX.drop('IBI_pnn25',1)
        self.datasetPX = self.datasetPX.drop('IBI_pnn50',1)
        self.datasetPX = self.datasetPX.drop('IBI_Min',1)
        self.datasetPX = self.datasetPX.drop('IBI_Max',1)
        self.datasetPX = self.datasetPX.drop('IBI_sd1',1)
        self.datasetPX = self.datasetPX.drop('IBI_sd2',1)
        self.datasetPX = self.datasetPX.drop('IBI_sd12',1)
        self.datasetPX = self.datasetPX.drop('IBI_sdell',1)
       
        #corr = self.datasetPX.corr()
        #sm.graphics.plot_corr(corr, xnames=list(corr.columns))
        scatter_matrix(self.datasetPX,diagonal='kde')
        #parallel_coordinates(self.datasetPX, 'Experience',colormap='winter')
        plt.show()
        
        #self.datasetPX.boxplot()
        #self.datasetPX.hist()
        #
    def filter(self,emotion,n_ocorr,emotionValue):
        df = self.datasetPX.loc[:, 'Event':'Rater 6'];
        print(df.shape)
        for index, row in df.iterrows():
            #print(list(row))
            if(list((row)).count(emotion) >= n_ocorr ):              
                self.datasetPX.loc[index,'Experience'] = emotionValue    
                #print(index)      
  
  
        
        
if __name__ == '__main__':
    run =  FeaturesBiosignal()
    run.loadData()