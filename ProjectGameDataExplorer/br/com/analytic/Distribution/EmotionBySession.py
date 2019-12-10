'''
Created on 29 de nov de 2019

@author: eltonss
'''
import biosppy
from builtins import int
from datetime import datetime
from datetime import datetime, timedelta
import glob
import math
import matplotlib
import numpy
import peakutils
import sys, os
from time import mktime

from pyphysio import EvenlySignal, UnevenlySignal
from pyphysio.tests import TestData   

from ProjectGameDataExplorer.br.com.util import UnixTime, SourceData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio as ph
import pyphysio.estimators.Estimators as est
import pyphysio.filters.Filters as flt
import pyphysio.indicators.FrequencyDomain as fd_ind
import pyphysio.indicators.TimeDomain as td_ind
import seaborn as sns

import itertools
from matplotlib import pyplot
from scipy import stats
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.data import normalize

from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib.pyplot import plot
matplotlib.use('Agg')
# import the Signal classes
# import data from included examples
# create a signal
# create a Filter
from scipy.stats import rankdata, norm
from scipy.stats import boxcox
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

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
        P = [2,3,4,5,6,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,37,43,45,47]
        #P = [3,4]
        
        _UT = UnixTime.UnixTime()
        dataframe = pd.DataFrame(columns=['Player','Session','ts', 'Happiness','Sadness',
                                   'Anger','Fear','Surprise','Disgust']);   
        auxHappiness = [];
        auxFear = [];
        auxSurprise = [];
        auxDisgust = [];
        auxSadness = [];
        auxAnger = [];
        auxGroup = [];
        auxP = [];
        auxTS = []
        for participant in P:
                print("Player: %s" % (participant))
                filter = ((self.dataset['Player'] == participant) & 
                              (self.dataset['Session'] == session));
                auxi = np.min(self.dataset[filter]['Interval Initial'])
                auxf = np.max(self.dataset[filter]['Interval Final'])
                path_default = "/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/EMOCAO.csv".format(participant)

                if(participant == 2):
                    path_default="/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/Session {}/EMOCAO.csv".format(participant,session)
                
                if(participant == 5 and (session == 1 or session == 3)):
                    path_default="/media/eltonss/9A2494A8249488C1/Users/eltonn/Dropbox/Dados Experimentos/Experimentos/Dados Coletados/Participante {}/Parte 2/EMOCAO.csv".format(participant)


                df = SourceData.SourceData().LoadDataFacialExpression(indexSession=None,
                                                   path=path_default);
             
                df['Time'] = [datetime.timestamp(dt) for dt in  df['Time']]
                
               
                data = df[(df['Time'] >= auxi) & (df['Time'] <= auxf)]
                #data = data[~(data == 0).any(axis=1)]
                data = data.dropna()
                for index, row in data.iterrows():
                    #print(row['c1'], row['c2'])
                    auxHappiness.append(row['Happiness'])
                    auxFear.append(row['Fear'])
                    auxSurprise.append(row['Surprise'])
                    auxDisgust.append( row['Disgust'])
                    auxSadness.append(row['Sadness'])
                    auxAnger.append(row['Anger'])
                    auxGroup.append(session);
                    auxP.append(participant);
                    auxTS.append(row['Time'])
               
        dataframe['Happiness'] = np.log(auxHappiness) ;       
        dataframe['Fear'] =np.log(auxFear)  ;        
        dataframe['Surprise'] =np.log(auxSurprise) ;        
        dataframe['Disgust'] =np.log(auxDisgust);        
        dataframe['Sadness'] =np.log(auxSadness) ;        
        dataframe['Anger'] =  np.log(auxAnger) ;
        dataframe['Session'] = auxGroup; 
        dataframe['Player'] = auxP;  
        dataframe['ts'] = auxTS;  
        dataframe = dataframe[pd.notnull(dataframe)]
        dataframe = dataframe[np.isfinite(dataframe)]
        
        print("Finish Session %s " % session)     
        return dataframe;
    def runFactorAnalyzer(self,cols_to_norm,result):
        fa = FactorAnalyzer(rotation="varimax",n_factors=2)
        df = result[cols_to_norm]      
        result = result.dropna()
        df =  df.dropna()

        fa.fit(df)
        ev= fa.get_eigenvalues();
        kmo_all,kmo_model=calculate_kmo(df)
        print("kmo_model: %s " % kmo_model)
        array = fa.transform(df)
        print("Factors: %s" % (array))
        print("loadings: %s " % fa.loadings_)

        
        print("eigenvalues: %s " % ev[0])
        dataframe = pd.DataFrame(columns=['Player','Session','Time', 'NegativeEmotion','PositiveEmotion']);  
        print("T session: %s " % len(result['Session']))
        dataframe['Session'] = result['Session']; 
        dataframe['Player'] = result['Player'];   
        dataframe['Time'] =result['ts'];
        dataframe['NegativeEmotion'] =array[:,0];
        dataframe['PositiveEmotion'] =array[:,1];
        dataframe.to_csv('/home/eltonss/Desktop/MetricsEmotion.csv',sep=',')
        #print(fa.get_factor_variance())
        #plt.scatter(range(1,df.shape[1]+1),ev[0])
        #plt.plot(range(1,df.shape[1]+1),ev[0])
        #plt.title('Scree Plot')
        #plt.xlabel('Factors')
        #plt.ylabel('Eigenvalue')
        #plt.grid()
        #plt.show()

    def saveEmotiondataDistribution (self):
        cols_to_norm = ['Happiness','Fear','Surprise','Disgust','Sadness','Anger']
        
        collectn_1= self.emotiondataDistribution(1)
        collectn_1[cols_to_norm] = collectn_1[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
       
        collectn_2= self.emotiondataDistribution(2)
        collectn_2[cols_to_norm] = collectn_2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_3= self.emotiondataDistribution(3)
        collectn_3[cols_to_norm] = collectn_3[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_4= self.emotiondataDistribution(4)
        collectn_4[cols_to_norm] = collectn_4[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        data_to_plot = [ collectn_1,collectn_2,collectn_3,collectn_4]
        result = pd.concat(data_to_plot)
                                   
        self.runFactorAnalyzer(cols_to_norm, result)
        
        for Emotion in cols_to_norm:
           
            print(Emotion)
            self.testNormality(collectn_1[Emotion],'{} Session 1'.format(Emotion))           
            self.testNormality(collectn_2[Emotion],'{} Session 2'.format(Emotion))           
            self.testNormality(collectn_3[Emotion],'{} Session 3'.format(Emotion))
            self.testNormality(collectn_4[Emotion],'{} Session 4'.format(Emotion))
            dataset = pd.DataFrame(list(zip((collectn_1[Emotion]),( collectn_2[Emotion]),
                                            (collectn_3[Emotion]),(collectn_4[Emotion]))), 
               columns =['C1', 'C2','C3','C4']) 
            data = [dataset[col].dropna() for col in dataset]
         
            self.calculateANOVA(data[0],data[1],data[2],data[3])
            
            print("------------")
        data_to_plot = [ collectn_1,collectn_2,collectn_3,collectn_4] 
        
        result = pd.concat(data_to_plot)
        fig, axes = plt.subplots(3,2,sharex='col')
        locale = [axes[0,0],axes[1,0],axes[2,0],axes[0,1],axes[1,1],axes[2,1]]
        for Emotion,local in zip(cols_to_norm,locale):
            TextTitle = "%s distribution by Session" % (Emotion);
            sns.violinplot(x=result["Session"],  y=result[Emotion],ax=local).set_title(TextTitle)
        
        fig.tight_layout()
        plt.show()    
        
        #sns_plot.figure.savefig("/home/eltonss/Pictures/Resultados/Emotion/output.png")    
        print("Finish saveEmotiondataDistribution")
    def normalize(self,arr):
        start = 0
        end = 1
        width = end - start
        res = (arr - arr.min())/(arr.max() - arr.min())* width + start;
        return res;    
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
    def calculateANOVA(self,*data):
        fvalue, pvalue = stats.kruskal(*data)
        print("ANOVA - The null hypothesis : No difference between means")
        print(fvalue, pvalue)
        alpha = 1e-3
        if pvalue < 0.05:  # null hypothesis: x comes from a normal distribution
            print("The null hypothesis can be rejected")
        else: print("The null hypothesis cannot be rejected")
        
    def tukeyTest(self,data,labels):
        print("Reject the null hypothesis if the absolute value of the test"+ 
        "statistic is greater than the critical value")
        
        df = pd.DataFrame()
        index = 1;
        for sample,nome in zip(data,labels):
            df[nome] =  pd.Series(sample)
            index = index + 1;
        
        stacked_data = df.stack().reset_index()
        stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})

        MultiComp = MultiComparison(stacked_data['result'],
                            stacked_data['treatment'])

   
        print(MultiComp.tukeyhsd().summary())
        
    def HomogeneityVariances(self,data):
        print("HomogeneityVariances:Null hypothesis: samples from populations have equal variances.")
        w, pvalue = stats.levene(*data)
        print(w, pvalue)
        alpha = 1e-3
        if pvalue < alpha:  # null hypothesis: x comes from a normal distribution
            print("The null hypothesis can be rejected")
        else: print("The null hypothesis cannot be rejected")
    
    def testNormality(self,x,label): 
        print("Normality Test:  Null hypothesis: data is drawn from normal distribution") 
        k2, pvalue = stats.shapiro(x)
        print("For sample ({0}) p = {1}".format(label,pvalue))
        alpha = 1e-3
        if pvalue < 0.05:  # null hypothesis: x comes from a normal distribution
            print("The null hypothesis can be rejected") 
            #"then the null hypothesis is rejected and there is evidence that the data tested are not normally distributed")
        else: print("The null hypothesis cannot be rejected ") 
                  #then the null hypothesis isn't rejected and 
                  
    
    
if __name__ == '__main__':
    run = EmotionBySession()
    run.saveEmotiondataDistribution()