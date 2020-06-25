'''
Created on 29 de nov de 2019

@author: eltonss
'''
from datetime import datetime
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer
from matplotlib.pyplot import plot
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import sys

from pyphysio import EvenlySignal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio.filters.Filters as flt
import seaborn as sns
from ProjectGameDataExplorer.br.com.util.SourceData import SourceData
pd.options.mode.chained_assignment = None  # default='warn'

__all__ = ['QRangeSlider']


class ExtractFactorFace():
    '''
    classdocs
    '''

    def __init__(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/Dataset/dataset.csv", sep=',') 
        Players_train = [3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 37, 43, 45, 47]
        Players_test =  [31,32,33,34,35,36,41,42,44,46,48,49,50,51]
        Players  = Players_test
        #Players = [2]
        #P = [30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51]
        #Players = [3,4]
        for player in Players:
            self.saveEmotiondataDistribution(player)
    
    def emotiondataDistribution(self, participant,session):
        try:
                      
            sd = SourceData();
            print("Player and Session: (%s,%s)" % (participant, session))
            filter = ((self.dataset['Player'] == participant) & (self.dataset['Session'] == session));
            time_initial_session = np.min(self.dataset[filter]['Interval Initial'])
            time_final_session = np.max(self.dataset[filter]['Interval Final'])
                  
            path_default = "/media/elton/Backup/ColetaDados/Validacao/Participante {}/EMOCAO.csv".format(participant)
    
            df = sd.LoadDataFacialExpression(indexSession=None,path=path_default);
                    
            df['Time'] = [datetime.timestamp(dt) for dt in  df['Time']]
            filter = ((df['Time'] >= float(time_initial_session)) & (df['Time'] <= float(time_final_session)))
                    
            data = df[filter]
            # Print shape of original DataFrame
            print("Shape of Original DataFrame without Missing Values: {}".format(data.shape))          
                    
            data['Player'] = participant
            data['Session'] = session                   
            data = data.drop('Neutral', 1)
            data.rename(columns={'Time':'ts'},inplace=True)
                    
            data[data == 0] = np.nan
            data = data.dropna()        
                              
            # Print shape of new DataFrame
            print("Shape of DataFrame After Dropping with Missing Values: {}".format(data.shape))
            
            print("Finish Session %s " % session)     
            return data;
        except: 
            print("Oops!", sys.exc_info()[0], "occured.")
            print("Erro em emotiondataDistribution")

    def runFactorAnalyzer(self, cols_to_norm, result):
        fa = FactorAnalyzer(rotation="varimax", n_factors=2)
        df = result[cols_to_norm]      
        result = result.dropna()
        df = df.dropna()

        fa.fit(df)
        ev = fa.get_eigenvalues();
        kmo_all, kmo_model = calculate_kmo(df)
        
        if(kmo_model < 0.6):
            print("kmo_model: %s " % kmo_model)
        array = fa.transform(df)
        #print("Factors: %s" % (array))
        #print("loadings: %s " % fa.loadings_)
        
        #print("eigenvalues: %s " % ev[0])
        dataframe = pd.DataFrame(columns=['Player', 'Session', 'Time', 'NegativeEmotion', 'PositiveEmotion']);  
        print("T session: %s " % len(result['Session']))
        dataframe['Session'] = result['Session']; 
        dataframe['Player'] = result['Player'];   
        dataframe['Time'] = result['ts'];
        dataframe['NegativeEmotion'] = np.around(array[:, 0],2);
        dataframe['PositiveEmotion'] = np.around(array[:, 1],2);
        dataframe.to_csv('/home/elton/Desktop/Dataset/MetricsEmotion.csv', sep=',',mode='a', header=False)
       
        #print(fa.get_factor_variance())
        
        #plt.scatter(range(1,df.shape[1]+1),ev[0])
        #plt.plot(range(1,df.shape[1]+1),ev[0],'ro-')
      
        #plt.title('Scree Plot')
        #plt.xlabel('Factors')
        #plt.ylabel('Eigenvalue')
        #plt.grid()
        #plt.show()

    def saveEmotiondataDistribution (self,participant):
        cols_to_norm = ['Happiness', 'Fear', 'Surprise', 'Disgust', 'Sadness', 'Anger']
        
        collectn_1 = self.emotiondataDistribution(participant,1)
        collectn_1[cols_to_norm] = collectn_1[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
       
        collectn_2 = self.emotiondataDistribution(participant,2)
        collectn_2[cols_to_norm] = collectn_2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_3 = self.emotiondataDistribution(participant,3)
        collectn_3[cols_to_norm] = collectn_3[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_4 = self.emotiondataDistribution(participant,4)
        collectn_4[cols_to_norm] = collectn_4[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        data_to_plot = [ collectn_1, collectn_2, collectn_3, collectn_4]
        result = pd.concat(data_to_plot)
                                   
        self.runFactorAnalyzer(cols_to_norm, result)
        
         
        print("Finish Extract Factor")
 
if __name__ == '__main__':
    run = ExtractFactorFace()
    
