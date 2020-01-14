'''
Created on 27 de dez de 2019

@author: elton
'''
import matplotlib

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.stats.stats import spearmanr
from scipy.stats import kendalltau
matplotlib.use('TkAgg')
from sklearn import preprocessing

class ExperienceResult(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/datasetResultado.csv", sep=',') 
        matplotlib.rcParams["savefig.directory"] = os.chdir(os.path.dirname('/home/elton/Pictures/Resultados/Modelo'))

    def groupingByEvent(self):
        
        
        
        
        #_EVENT= "Drifting"
        #_EVENT= "Gear Box"        
        #_EVENT= "Overtake"
        _EVENT= "Collision"
        
        #_EVENT= "Perdendo Posicao"        
        
        #_EVENT= "Car broke down"
        #_EVENT= "Roll over"
        #_EVENT= "Acelerando"
        #_EVENT= "Off Road"

        _SESSION = None
        count_tristeza, count_felicidade, count_raiva, count_calma = self.getOccurrenceEmotions(self.dataset,_EVENT,_SESSION)
        sum =  np.sum([count_tristeza,count_felicidade,count_raiva,count_calma])
        data1 = ([100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum])
        count_tristeza, count_felicidade, count_raiva, count_calma = self.summaryPredition(self.dataset, _EVENT,_SESSION)
        sum =  np.sum([count_tristeza,count_felicidade,count_raiva,count_calma])
        data2 =  ([100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum])
        
     
        coef, p = kendalltau(data1, data2)
        print('kendalltau correlation coefficient: %.3f' % coef)
        # interpret the significance
        alpha = 0.05
        if p > alpha:
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
        else:
            print('Samples are correlated (reject H0) p=%.3f' % p)
            
        coef, p = spearmanr(data1, data2)
        print('spearmanr correlation coefficient: %.3f' % coef)
        # interpret the significance
        alpha = 0.05
        if p > alpha:
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
        else:
            print('Samples are correlated (reject H0) p=%.3f' % p)
        self.plotSampleBar(data1, data2,_EVENT)
        plt.show()
    def geterror(self,count_tristeza, count_felicidade, count_raiva, count_calma):    
        return np.std(count_tristeza),np.std(count_felicidade),np.std(count_raiva),np.std(count_calma)

        
    def plotSampleBar(self,data1,data2,Event):
        labels = ['Calm','Happiness','Anger','Sadness']
     
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, data1, width, label='Reported Experience',edgecolor = 'black',ecolor='black', capsize=3)
        rects2 = ax.bar(x + width/2, data2, width, label='Estimated Experience', edgecolor = 'black',ecolor='black', capsize=3)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores (%)')
        ax.set_title('Scores by Event ({0})'.format(Event))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{0:.2f}%'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()


    def getOccurrenceEmotions(self, df,Event,Session=None):
        
        if(Session != None):
            print("XXXXXXXX")
            filter = ((df['Session'] == Session));
            df = df[filter]
        filter = ((df['Event'] == Event));
        df = df[filter] 
        count = Counter(df['Experience'])
        return count['Tristeza'], count['Felicidade'], count['Raiva'], count['Calma']
    
    def summaryPredition(self,df,Event,Session=None):
        if(Session != None):
            filter = ((df['Session'] == Session));
            df = df[filter]
        filter = ((df['Event'] == Event));
        df = df[filter] 
        count = Counter(df['Predicted'])
        
        meanCalm = np.sum(df['FactorCalma'])
        meanHappiness = np.sum(df['FactorFelicidade'])
        meanAnger = np.sum(df['FactorRaiva'])
        meanSadness = np.sum(df['FactorTristeza']) 
        return meanSadness,meanHappiness,meanAnger,meanCalm

if __name__ == '__main__':
    run = ExperienceResult()
    run.groupingByEvent()

