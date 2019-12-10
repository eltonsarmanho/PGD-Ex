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


class KappaStatistic():
    
    def __init__(self):
        #ratingtask = agreement.AnnotationTask(data=ListVector)
        #print("kappa (Mean): %.2f" % (ratingtask.kappa()))
        #print("fleiss " + str(ratingtask.multi_kappa()))
        #print("alpha " +str(ratingtask.alpha()))
        #print("scotts " + str(ratingtask.pi()))
        pass;
    def getKappa(self,taskdata):
        ratingtask = agreement.AnnotationTask(data=taskdata)
        return ratingtask.kappa();
    def responseKappa(self,n_result,taskdata,rater_dict):
        try:
            t = []
            txt = []
            for element in range(0,n_result):
                t.append(element)
            for i in range(2,n_result+1):
                c = list(itertools.combinations(t, i))
                unq = set(c)            
                for tuple in (unq):               
                    rater =[]
                    names = [];
                    for index in tuple:
                        names.append(rater_dict[index]); 
                        array = [ [x,y,z] for x, y, z in taskdata if x == str(index)]                                                 
                        for element in array:                    
                            rater.append(element)
                    kappa = self.getKappa(rater)
                
                    responde = "Kappa{0}:{1:.2f}".format(names,kappa)
                    #print(responde)
                    txt.append(responde)
            return txt;
        except:
            print("Erro Segment {%s}");
            txt = []
            return txt;
    
    
if __name__ == '__main__':
    
    def find(pattern, path):
        result = []
        for root, dirs, files in os.walk(path):
            for name in files:
                if fnmatch.fnmatch(name, pattern):
                    result.append(os.path.join(root, name))
        return result
    def split_list(alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                 for i in range(wanted_parts) ]
    
    
    group_emotion = dict(
    Asco = 'Nenhum',Nothing ='Nenhum',Nenhum='Nenhum', none='Nenhum',
    Irritado = 'Raiva',Raiva='Raiva',Loucura='Raiva',Furia='Raiva', Stress='Raiva',
    Nojo='Desgosto', Repulsa='Desgosto', Maldisposicao='Desgosto', Nausea='Desgosto',
    Horror='Medo', Assustado='Medo', Medo='Medo',Panico='Medo',
    Surpresa='Ansiedade',Preocupado = 'Ansiedade',Preocupacao='Ansiedade', Ansiedade='Ansiedade',
    Concentracao='Ansiedade',Pavor='Ansiedade',Nervosismo='Ansiedade',Nervoso='Ansiedade',
    Solidao='Tristeza', Pesar='Tristeza',Tristeza='Tristeza',Frustracao='Tristeza',Vazio='Tristeza', Desanimo='Tristeza',
    Determinacao='Desejo',Insuficiencia='Desejo',Desejo='Desejo', Saudade='Desejo',
    Calmo = 'Calma',Calma='Calma', Tranquilidade='Calma',Descontracao='Calma', Suavidade='Calma',Relaxado='Calma',
    Felicidade='Felicidade', Divertido='Felicidade',Diversao='Felicidade', Satisfacao='Felicidade',Simpatia='Felicidade')
    
    array_participant = [2,3,4,5,6,7,8,10,11,12,13,14,15,16,
                         17,18,19,20,21,22,23,24,25,26,27,28,29,30];
    for index_participant in array_participant:
        participante = '/home/eltonss/Documents/Julgamentos/P{}'.format(index_participant)
        for i in range (1,5):#1.,2,3,4
            rater_dict = {}
    
            file_ = 'S{}_*.csv'.format(i)
            session = 'Session {}'.format(i)
            print("Participante {%s} - %s" % (index_participant,session))
            result = find(file_, participante)    
            if(len(result) <= 1):
                continue
            index_rater = 0    
            taskdata =[]
            for files in result:
                matches = re.findall(r'S{}_(.+?)_P{}'.format(i,index_participant),files)
                index_response = 0;
                with open(files) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=';')
                    line_count = 0
                    for row in csv_reader:
                        taskdata.append([str(index_rater),index_response,group_emotion[str(row[2])]])
                        index_response = index_response + 1
                
                rater_dict[index_rater]  = matches[0]
                index_rater = index_rater + 1;
            kappa = KappaStatistic();
            array1 = kappa.responseKappa(len(result), taskdata,rater_dict);
            
            c = [array1]      
                    
            with open(participante+".txt", "a") as file:
                file.write("{0}\n".format(session))
                for x in zip(*c):
                    file.write("{0:<8}\n".format(*x))   
                
        


