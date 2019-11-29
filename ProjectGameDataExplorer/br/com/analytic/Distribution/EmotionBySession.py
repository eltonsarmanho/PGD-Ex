'''
Created on 29 de nov de 2019

@author: eltonss
'''
import matplotlib


from datetime import datetime
import sys, os
import math
import numpy

from ProjectGameDataExplorer.br.com.util import UnixTime, SourceData
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from time import mktime
import peakutils
import biosppy
from builtins import int
import glob
import seaborn as sns
matplotlib.use('Agg')
import pyphysio as ph
# import the Signal classes
from pyphysio import EvenlySignal, UnevenlySignal
# import data from included examples
from pyphysio.tests import TestData   
# create a signal
# create a Filter
import pyphysio.filters.Filters as flt
import pyphysio.estimators.Estimators as est

import pyphysio.indicators.TimeDomain as td_ind
import pyphysio.indicators.FrequencyDomain as fd_ind

__all__ = ['QRangeSlider']
class EmotionBySession():
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    def emotiondataDistribution(self,session):
        self.dataset = pd.read_csv("/home/eltonss/Desktop/dataset.csv", sep=',') 
        P = [3,4,5,10,11,12,13,14,15,16,17,21,23,24,26,27,28,29,30]
        
        _UT = UnixTime.UnixTime()
        dataframe = pd.DataFrame(columns=['Group', 'Happiness','Sadness',
                                   'Anger','Fear','Surprise','Disgust']);   
        auxTime = [];
        auxHappiness = [];
        auxFear = [];
        auxSurprise = [];
        auxDisgust = [];
        auxSadness = [];
        auxAnger = [];
        auxGroup = [];
        for participant in P:
                #print("Player: %s" % (participant))
                filter = ((self.dataset['Player'] == participant) & 
                              (self.dataset['Session'] == session));
                auxi = np.min(self.dataset[filter]['Interval Initial'])
                auxf = np.max(self.dataset[filter]['Interval Final'])
                path_default = "/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/EMOCAO.csv".format(participant)

                df = SourceData.SourceData().LoadDataFacialExpression(indexSession=None,
                                                   path=path_default);
            
                df = df.fillna(0)
                df = df.replace(np.nan, 0)

                df['Time'] = [datetime.timestamp(dt) for dt in  df['Time']]
                list_aux = list(zip( df['Happiness'], df['Sadness'], df['Anger'],
                                     df['Surprise'], df['Fear'], df['Disgust']))        
                data = pd.DataFrame(list_aux, columns=['Happiness', 'Sadness',
                                                 "Anger", "Surprise", "Fear", 'Disgust'])   
                Happiness = self.getDataEmotion("Happiness",data, df['Time'][0] , auxi, auxf) 
                Sadness = self.getDataEmotion("Sadness",data,  df['Time'][0], auxi, auxf)
                Anger = self.getDataEmotion("Anger",  data,df['Time'][0], auxi, auxf)
                Surprise = self.getDataEmotion("Surprise", data, df['Time'][0], auxi, auxf)
                Fear = self.getDataEmotion("Fear", data, df['Time'][0], auxi, auxf)
                Disgust = self.getDataEmotion("Disgust", data, df['Time'][0], auxi, auxf) 
                               
                index  = 0
                while (index<len(Happiness.get_values())):
                    auxHappiness.append(Happiness.get_values()[index])
                    auxFear.append(Fear.get_values()[index])
                    auxSurprise.append(Surprise.get_values()[index])
                    auxDisgust.append(Disgust.get_values()[index])
                    auxSadness.append(Sadness.get_values()[index])
                    auxAnger.append(Anger.get_values()[index])
                    auxGroup.append(session);
                    index = index + 1
        
       
        dataframe['Happiness'] = auxHappiness ;       
        dataframe['Fear'] = auxFear;
        dataframe['Surprise'] = auxSurprise ;
        dataframe['Disgust'] = auxDisgust;
        dataframe['Sadness'] =auxSadness ;
        dataframe['Anger'] = auxAnger ;
        dataframe['Group'] = auxGroup;       
        return dataframe;
    def saveEmotiondataDistribution (self):
        collectn_1= self.emotiondataDistribution(1)
        collectn_2= self.emotiondataDistribution(2)
        collectn_3= self.emotiondataDistribution(3)
        collectn_4= self.emotiondataDistribution(4)
       
        data_to_plot = [ collectn_1,collectn_2,collectn_3,collectn_4] 
        
        result = pd.concat(data_to_plot)
        
        fig, axes = plt.subplots(3,2,sharex='col')
        Emotion = 'Anger'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[0, 0]).set_title(TextTitle)
        
        print("------------")
        Emotion = 'Sadness'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[1, 0]).set_title(TextTitle)
        
        print("------------")
        
        Emotion = 'Disgust'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[2, 0]).set_title(TextTitle)
          
        print("------------")
        
        Emotion = 'Fear'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[0, 1]).set_title(TextTitle)
        
        print("------------")
        
        Emotion = 'Surprise'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[1, 1]).set_title(TextTitle)
        
        print("------------")
        
        Emotion = 'Happiness'
        TextTitle = "%s distribution by Session" % (Emotion);
        sns.violinplot(x=result["Group"],  y=result[Emotion],ax=axes[2, 1]).set_title(TextTitle)
        fig.tight_layout() 
        plt.show()
        #sns_plot.figure.savefig("/home/eltonss/Pictures/Resultados/Emotion/output.png")    
        print("Finish saveEmotiondataDistribution")
    def getDataEmotion(self,emotion,data,sTime,auxi, auxf):
        sampleRate = 30
        data = data[emotion].tolist()            
            
        data_emotion = EvenlySignal(values=data,
                   sampling_freq=sampleRate,
                   signal_type='emotion',
                   start_time=sTime)          
        data_emotion = flt.Normalize('maxmin')(data_emotion)
                # Select area of interest
        data_emotion = data_emotion.segment_time(auxi, auxf) 
        return data_emotion;   
    
    
if __name__ == '__main__':
    run = EmotionBySession()
    run.saveEmotiondataDistribution()