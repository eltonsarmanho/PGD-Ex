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
from br.com.util.SourceData import SourceData

__all__ = ['QRangeSlider']


class ExtractFactorFace():
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
    
    def emotiondataDistribution(self, session):
        try:
            self.dataset = pd.read_csv("/home/elton/Desktop/datasetValidacao.csv", sep=',') 
            #P = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 37, 43, 45, 47]
            P = [30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51]
            # P = [37]
            dataframe = pd.DataFrame(columns=['Player', 'Session', 'ts', 'Happiness', 'Sadness',
                                       'Anger', 'Fear', 'Surprise', 'Disgust']);   
            auxHappiness = [];
            auxFear = [];
            auxSurprise = [];
            auxDisgust = [];
            auxSadness = [];
            auxAnger = [];
            auxGroup = [];
            auxP = [];
            auxTS = []
            sd = SourceData();
            for participant in P:
                    print("Player and Session: (%s,%s)" % (participant, session))
                    filter = ((self.dataset['Player'] == participant) & 
                                  (self.dataset['Session'] == session));
                    auxi = np.min(self.dataset[filter]['Interval Initial'])
                    auxf = np.max(self.dataset[filter]['Interval Final'])
                  
                    path_default = "/home/elton/Documents/Experimento/Validacao/Participante {}/EMOCAO.csv".format(participant)
    
                    df = sd.LoadDataFacialExpression(indexSession=None,
                                                       path=path_default);
                    
                    df['Time'] = [datetime.timestamp(dt) for dt in  df['Time']]
                    filter = ((df['Time'] >= float(auxi)) & (df['Time'] <= float(auxf)))
                    
                    data = df[filter]
                    

                    # data = data[~(data == 0).any(axis=1)]
                    data = data.dropna()
                    for index, row in data.iterrows():
                        # print(row['c1'], row['c2'])
                        auxHappiness.append(row['Happiness'])
                        auxFear.append(row['Fear'])
                        auxSurprise.append(row['Surprise'])
                        auxDisgust.append(row['Disgust'])
                        auxSadness.append(row['Sadness'])
                        auxAnger.append(row['Anger'])
                        auxGroup.append(session);
                        auxP.append(participant);
                        auxTS.append(row['Time'])
                   
            dataframe['Happiness'] = np.log(auxHappiness) ;       
            dataframe['Fear'] = np.log(auxFear)  ;        
            dataframe['Surprise'] = np.log(auxSurprise) ;        
            dataframe['Disgust'] = np.log(auxDisgust);        
            dataframe['Sadness'] = np.log(auxSadness) ;        
            dataframe['Anger'] = np.log(auxAnger) ;
            dataframe['Session'] = auxGroup; 
            dataframe['Player'] = auxP;  
            dataframe['ts'] = auxTS;  
            dataframe = dataframe[pd.notnull(dataframe)]
            dataframe = dataframe[np.isfinite(dataframe)]
            
            print("Finish Session %s " % session)     
            return dataframe;
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
        print("kmo_model: %s " % kmo_model)
        array = fa.transform(df)
        print("Factors: %s" % (array))
        print("loadings: %s " % fa.loadings_)
        
        print("eigenvalues: %s " % ev[0])
        dataframe = pd.DataFrame(columns=['Player', 'Session', 'Time', 'NegativeEmotion', 'PositiveEmotion']);  
        print("T session: %s " % len(result['Session']))
        dataframe['Session'] = result['Session']; 
        dataframe['Player'] = result['Player'];   
        dataframe['Time'] = result['ts'];
        dataframe['NegativeEmotion'] = np.around(array[:, 0],2);
        dataframe['PositiveEmotion'] = np.around(array[:, 1],2);
        dataframe.to_csv('/home/elton/Desktop/MetricsEmotion.csv', sep=',')
        print(fa.get_factor_variance())
        plt.scatter(range(1,df.shape[1]+1),ev[0])
        plt.plot(range(1,df.shape[1]+1),ev[0],'ro-')
      
        plt.title('Scree Plot')
        plt.xlabel('Factors')
        plt.ylabel('Eigenvalue')
        plt.grid()
        plt.show()

    def saveEmotiondataDistribution (self):
        cols_to_norm = ['Happiness', 'Fear', 'Surprise', 'Disgust', 'Sadness', 'Anger']
        
        collectn_1 = self.emotiondataDistribution(1)
        collectn_1[cols_to_norm] = collectn_1[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
       
        collectn_2 = self.emotiondataDistribution(2)
        collectn_2[cols_to_norm] = collectn_2[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_3 = self.emotiondataDistribution(3)
        collectn_3[cols_to_norm] = collectn_3[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        collectn_4 = self.emotiondataDistribution(4)
        collectn_4[cols_to_norm] = collectn_4[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        data_to_plot = [ collectn_1, collectn_2, collectn_3, collectn_4]
        result = pd.concat(data_to_plot)
                                   
        self.runFactorAnalyzer(cols_to_norm, result)
        
         
        print("Finish Extract Factor")

    def normalize(self, arr):
        start = 0
        end = 1
        width = end - start
        res = (arr - arr.min()) / (arr.max() - arr.min()) * width + start;
        return res;    

    
    
if __name__ == '__main__':
    run = ExtractFactorFace()
    run.saveEmotiondataDistribution()
