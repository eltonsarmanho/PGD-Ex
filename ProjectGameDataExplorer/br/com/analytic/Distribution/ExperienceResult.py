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
from pandas.core.frame import DataFrame
from statsmodels.stats.proportion import proportion_confint
matplotlib.use('TkAgg')
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import numpy
from scipy import stats

import seaborn as sns
import numpy as np, statsmodels.stats.api as sms

class ExperienceResult(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/datasetResultado.csv", sep=',') 
        matplotlib.rcParams["savefig.directory"] = os.chdir(os.path.dirname('/home/elton/Pictures/Resultados/Modelo'))
        #USAR SEMPRE CONDICAO SEM ASPAS
        #self.dataset = self.dataset[self.dataset['Session'] == 1]
    
    def calMean(self):
        
        self.dataset = self.dataset[(self.dataset['Event'].isin(["Drifting","Collision","Gear Box"]))]
        
        df = self.dataset[['Session','FactorCalma','FactorFelicidade','FactorRaiva','FactorTristeza']]
        print(df.groupby(['Session']).mean().round(2)*100)
        
        df = self.dataset[['Player','Session','Experience']]
        
        
        df = df.groupby(['Session','Experience']).size().unstack(fill_value=0)
        df = df.reset_index().rename_axis(None, axis=1)
        print(df)
        df = df.set_index('Session')
        res = df.div(df.sum(axis=1)/100, axis=0)
        print(res.reset_index())
        #x = [7,7,5,19]
        #x = [24,11,8,15]
        x = [19,13,7,16]
        #x = [8,5,15,18]
        sum = np.sum(x)
        for value in x:
            ci_low, ci_upp = proportion_confint(value, sum, alpha=0.05, method='wilson')
            print("({0}\%,{1}\%)".format(ci_low.round(2)*100,ci_upp.round(2)*100));
        print("\n")
        for value in x:
            ci_low, ci_upp = proportion_confint(value, sum, alpha=0.05, method='normal')
            print("({0}\%,{1}\%)".format(ci_low.round(2)*100,ci_upp.round(2)*100));    
            #print("(:.2f%,:.2f%)  " % ())
        
    def groupingByEvent(self):
        
        
        
        #_SESSION = 1
        #_EVENT = ["Drifting","Collision","Gear Box"]
        _SESSION = None
        #_EVENT = ["Collision"]
        _EVENT = ["Gear Box"]
        #_EVENT = ['Drifting']
        #_EVENT= "Drifting"
        #_EVENT= "Gear Box"        
        #_EVENT= "Overtake"
        #_EVENT= "Collision"
        #_EVENT= "Off Road"

        
        
        count_tristeza, count_felicidade, count_raiva, count_calma = self.getOccurrenceEmotions(self.dataset,_EVENT,_SESSION)
        sum =  np.sum([count_tristeza,count_felicidade,count_raiva,count_calma])
        data1 = ([100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum])
        str1 = "{0:.2f}\% & {1:.2f}\% & {2:.2f}\% & {3:.2f}\%".format(100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum)
        count_tristeza, count_felicidade, count_raiva, count_calma = self.summaryPredition(self.dataset, _EVENT,_SESSION)
        sum =  np.sum([count_tristeza,count_felicidade,count_raiva,count_calma])
        data2 =  ([100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum])
        str2 = "{0:.2f}\% & {1:.2f}\% & {2:.2f}\% & {3:.2f}\%".format(100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum)
        print(str2+" & "+str1)
        coef, p = kendalltau(data1, data2)
        print('kendalltau correlation coefficient: %.3f' % coef)
        
        # interpret the significance
        alpha = 0.05
        print(p > alpha)
        if np.around(p,2) > alpha:
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % np.around(p,2) )
        else:
            print('Samples are correlated (reject H0) p=%.3f' % np.around(p,2) )
            
        coef, p = spearmanr(data1, data2)
        print('spearmanr correlation coefficient: %.3f' % coef)
        # interpret the significance
        alpha = 0.05
        if np.around(p,2) > alpha:
            #H0
            print('Samples are uncorrelated (fail to reject H0) p=%.3f' % np.around(p,2) )
        else:
            print('Samples are correlated (reject H0) p=%.3f' % np.around(p,2) )
        
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
        ax.legend(fontsize=14)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
        
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{0:.2f}%'.format(height),
                            size =14,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        
        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()


    def getOccurrenceEmotions(self, df,Event = None,Session=None):
        
        if(Session != None):
            df = df[df['Session'] == Session]
        if(Event != None):
            df = df[(df['Event'].isin(Event))] 
        count = Counter(df['Experience'])
        return count['Tristeza'], count['Felicidade'], count['Raiva'], count['Calma']
    
    def summaryPredition(self,df,Event= None,Session=None):
        if(Session != None):
            df = df[df['Session'] == Session]
        if(Event != None):
            df = df[(df['Event'].isin(Event))]
              
        count = Counter(df['Predicted'])
        
        meanCalm = np.sum(df['FactorCalma'])
        meanHappiness = np.sum(df['FactorFelicidade'])
        meanAnger = np.sum(df['FactorRaiva'])
        meanSadness = np.sum(df['FactorTristeza']) 
        return meanSadness,meanHappiness,meanAnger,meanCalm
    def calcConfidenceInterval(self):
        

        x1 = np.divide([26.63,22.27,41.38, 39.06],100)
        x2 = np.divide([18.42,41.38,34.55, 17.39],100)
        
        #x1 = [22.66, 30.84, 40.70, 2.26]
        #x2 = [18.42, 18.97, 23.64, 10.87]
        
        #x1 = [14.97, 14.20, 19.97, 19.49]
        #x2 = [13.16, 13.79, 12.73, 32.61]
        
        #x1 = [37.74, 27.50, 17.17, 39.19]
        #x2 = [50, 25.86, 29.09, 39.13]       
        
        cm = sms.CompareMeans(sms.DescrStatsW(x1), sms.DescrStatsW(x2))
        a, b = cm.tconfint_diff(usevar='unequal')
        print (a.round(2),b.round(2))
        
    def dataframeQuestionario(self):
        url = "/home/elton/Documents/Experimento/dataEscala.csv"
        data = pd.read_csv(url)
                        

        #df = data[['Challenge_1','Competence_1', 'Immersion_1','Flow_1', 'PositiveAffect_1','NegativeAffect_1','Calm_1','Happy_1','Anger_1','Sadness_1']]
        #df = data[['Challenge_2','Competence_2_1','Competence_2_2', 'Immersion_2', 'Flow_2', 'PositiveAffect_2','NegativeAffect_2','Calm_2','Happy_2','Anger_2','Sadness_2']]
        #df = data[['Challenge_3','Competence_3_1','Competence_3_2',  'Immersion_3','Flow_3', 'PositiveAffect_3','NegativeAffect_3','Calm_3','Happy_3','Anger_3','Sadness_3']]
        df = data[['Challenge_4','Competence_4', 'Immersion_4','Flow_4', 'PositiveAffect_4','NegativeAffect_4','Calm_4','Happy_4','Anger_4','Sadness_4']]

        df  = df.dropna()

        #self.dataset = pd.read_csv("/home/elton/Desktop/datasetResultado.csv", sep=',')
        #30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51

        #self.dataset = self.dataset[self.dataset['Player'] == 51]
        
        #count_tristeza, count_felicidade, count_raiva, count_calma = self.getOccurrenceEmotions(self.dataset,None,4)
        #sum =  np.sum([count_tristeza,count_felicidade,count_raiva,count_calma])
        #data = ([100*count_calma/sum, 100*count_felicidade/sum,100*count_raiva/sum,100*count_tristeza/sum])
        #print(data)
        
        corr_matrix = df.corr(method='spearman').round(decimals=2)
        print(corr_matrix)
        fig, ax = plt.subplots()
        plt.title("Session 1")
        im = ax.imshow(corr_matrix)
        im.set_clim(-1, 1)
       
        sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), annot= True,cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
        plt.show()
if __name__ == '__main__':
    run = ExperienceResult()
    #run.calMean()
    #run.dataframeQuestionario()
    run.groupingByEvent()

