'''
Created on 24 de ago de 2019

@author: eltonss
'''

import csv
from itertools import chain, repeat, islice, count
import itertools
from nltk import agreement
import os
import os, fnmatch
import re
from sklearn.metrics import cohen_kappa_score
import numpy as np

import pandas as pd
import collections 


class KappaStatistic():
    
    def __init__(self):
        # ratingtask = agreement.AnnotationTask(data=ListVector)
        # print("kappa (Mean): %.2f" % (ratingtask.kappa()))
        # print("fleiss " + str(ratingtask.multi_kappa()))
        # print("alpha " +str(ratingtask.alpha()))
        # print("scotts " + str(ratingtask.pi()))
        pass;

    def getKappa(self, taskdata):
        ratingtask = agreement.AnnotationTask(data=taskdata)
        # if(float(ratingtask.kappa()) >= 0.4):
            # print("kappa: %.2f" % (ratingtask.kappa()))
        print("fleiss: %.2f " % (ratingtask.multi_kappa()))
        print("alpha: %.2f " % (ratingtask.alpha()))
        print("scotts: %.2f " % (ratingtask.pi()))
            # return True;
        return float(ratingtask.kappa());
    
    def CountFrequency(self, arr): 
        return collections.Counter(arr)

    
if __name__ == '__main__':

    group_emotion = dict(
    Asco='Nenhum', Nothing='Nenhum', Nenhum='Nenhum', none='Nenhum',
    Irritado='Raiva', Raiva='Raiva', Loucura='Raiva', Furia='Raiva', Stress='Raiva',
    Nojo='Desgosto', Repulsa='Desgosto', Maldisposicao='Desgosto', Nausea='Desgosto',
    Horror='Medo', Assustado='Medo', Medo='Medo', Panico='Medo',
    
    Surpresa='Ansiedade', Preocupado='Ansiedade', Preocupacao='Ansiedade', Ansiedade='Ansiedade',
    Concentracao='Ansiedade', Pavor='Ansiedade', Nervosismo='Ansiedade', Nervoso='Ansiedade',
    
    Solidao='Tristeza', Pesar='Tristeza', Tristeza='Tristeza', Frustracao='Tristeza', Vazio='Tristeza', Desanimo='Tristeza',
    
    Determinacao='Desejo', Insuficiencia='Desejo', Desejo='Desejo', Saudade='Desejo',
    
    Calmo='Calma', Calma='Calma', Tranquilidade='Calma', Descontracao='Calma', Suavidade='Calma', Relaxado='Calma',
    
    Felicidade='Felicidade', Divertido='Felicidade', Diversao='Felicidade', Satisfacao='Felicidade', Simpatia='Felicidade')
    
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result

    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i * length // wanted_parts: (i + 1) * length // wanted_parts] 
                 for i in range(wanted_parts) ]

    def setJudge(index_rater, data):
        index_response = 0;
        for row in data:
            if(row == 'X'): return 0;
            taskdata.append([str(index_rater), index_response, row])
            index_response = index_response + 1;
        return 1;
   
    array_participant = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 20, 21, 22, 23, 24, 25, 26, 27,
                         28, 29, 30, 37, 43, 45, 47];
    
    array_participant = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 20, 21, 23, 24, 25, 26, 27,
                         28, 29, 30, 37, 43, 45, 47];
                         
    array_participant = [2, 4, 5, 10, 12, 13, 14, 15, 16,
                         17, 20, 21, 23, 24, 25, 26, 27,
                         28, 29, 30, 37, 43, 45, 47];
    array_participant = [2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 20, 21, 23, 24, 25, 26, 27,
                         28, 29, 30, 37, 43, 45, 47];
    
    dataset = pd.read_csv("/home/elton/Desktop/dataset.csv", sep=',') 
    #dataset = pd.read_csv("/home/elton/Desktop/datasetResultado.csv", sep=',') 
    session = 4;
    lst = []
    for participant in array_participant:
        try:
            taskdata = []
            
            filter = ((dataset['Player'] == participant) & (dataset['Session'] == session));
            #filter = ((dataset['Player'] == 51) & (dataset['Event'].isin(['Cambio'])));
            #filter = ((dataset['Player'] == 51) & (dataset['Event'].isin(['Colisao'])));
            #filter = ((dataset['Player'] == 51) & (dataset['Event'].isin(['Roll over'])));
            #filter = ((dataset['Player'] == 51) & (dataset['Event'].isin(['Off Road Over'])));
            _rater_1 = (dataset[filter]['Rater 1'])
            _rater_2 = (dataset[filter]['Rater 2'])
            _rater_3 = (dataset[filter]['Rater 3'])
            _rater_4 = (dataset[filter]['Rater 4'])        
            _rater_5 = (dataset[filter]['Rater 5'])
            _rater_6 = (dataset[filter]['Rater 6'])
            index_rater = 0
            index_rater = index_rater + setJudge(index_rater, _rater_1);
            index_rater = index_rater + setJudge(index_rater, _rater_2);
            index_rater = index_rater + setJudge(index_rater, _rater_3);
            index_rater = index_rater + setJudge(index_rater, _rater_4);
            index_rater = index_rater + setJudge(index_rater, _rater_5);
            index_rater = index_rater + setJudge(index_rater, _rater_6);
            rater = []
            kappa = KappaStatistic();
            #print("Player and Session: (%s,%s)" % (participant, session))
            for i in range(2, index_rater + 1):
                    c = list(itertools.combinations(range(0, index_rater), i))
                    unq = set(c) 
                    for tuple in (unq):  
                                  
                        rater = []
                        names = [];
                        for index in tuple:
                            array = [ [x, y, z] for x, y, z in taskdata if x == str(index)]                                                 
                            for element in array:                    
                                rater.append(element)
                        print(rater)
                        resp = kappa.getKappa(rater)
                        #print("Raters (Kappa = %.2f): %s " % (resp, str(tuple))) 
                        if(resp >= 0.6):   
                            lst.append(resp)                         
                            print("Raters (Kappa = %.2f): %s " % (resp, str(tuple))) 
                            
        except: print("ERRO")
    print("Mean %s " % (np.mean(lst)))

