'''
Created on 24 de ago de 2019

@author: eltonss
'''

from nltk import agreement
import os
import csv
import os, fnmatch
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
    Asco = 'Nenhum',Nothing ='Nenhum',Nenhum='Nenhum', none='Nenhum',Irritado = 'Raiva',Raiva='Raiva',Loucura='Raiva',Furia='Raiva', Stress='Raiva',
    Nojo='Desgosto', Repulsa='Desgosto', Maldisposicao='Desgosto', Nausea='Desgosto',
    Horror='Medo', Assustado='Medo', Medo='Medo',Panico='Medo',
    Surpresa='Ansiedade',Preocupado = 'Ansiedade',Preocupacao='Ansiedade', Ansiedade='Ansiedade',
    Concentracao='Ansiedade',Pavor='Ansiedade',Nervosismo='Ansiedade',Nervoso='Ansiedade',
    Solidao='Tristeza', Pesar='Tristeza',Tristeza='Tristeza',Frustracao='Tristeza',Vazio='Tristeza', Desanimo='Tristeza',
    Determinacao='Desejo',Insuficiencia='Desejo',Desejo='Desejo', Saudade='Desejo',
    Calmo = 'Calma',Calma='Calma', Tranquilidade='Calma',Descontracao='Calma', Suavidade='Calma',Relaxado='Calma',
    Felicidade='Felicidade', Divertido='Felicidade',Diversao='Felicidade', Satisfacao='Felicidade',Simpatia='Felicidade')
    #print(group_emotion)
    for i in range (1,5):
        file_ = 'S{}_*.csv'.format(i)
        print("Session %s" % (i))
        result = find(file_, '/home/eltonss/Documents/Julgamentos/P12')    
        if(len(result) <= 1):
            continue
        index_rater = 0    
        taskdata =[]
        for files in result:
            index_response = 0;
            with open(files) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                line_count = 0
                for row in csv_reader:
                    taskdata.append([str(index_rater),index_response,group_emotion[str(row[2])]])
                    index_response = index_response + 1
            index_rater = index_rater + 1;
        rater =[]
        half =0
        print("Half (%s)" % half)
        for i in range(len(result)):
            array = [(z) for x, y, z in taskdata if x ==str(i)]
            if(half in (1,0)):
                array = split_list(array,2)[half]
            rater.append(array)
    
        #for i in range(len(result)):
            #print("Resposta do J(%s): %s "%(i,rater[i]))
        list_value= []
        for i in range(0,len(result)):
            for j in range(i+1,len(result)):
                kappa = cohen_kappa_score(rater[i], rater[j])
                list_value.append(kappa)
                #print("Kappa(J%s,J%s) :%.2f" %(i,j,kappa))
        list_value.sort(reverse=True)
        print(list_value)
        kappa = KappaStatistic().getKappa(taskdata)
        print("[%.2f] %.2f" % (list_value[0],kappa))
        #print("\n")


