'''
Created on 24 de ago de 2019

@author: eltonss
'''

from nltk import agreement
rater1 = ['N','S','S','N','S','S']
rater2 = ['S','S','S','N','S','N']
rater3 = ['S','N','S','N','S','N']
taskdata =[]
for i in range(0, len(rater1)):
    taskdata.append(['0',(i),str(rater1[i])])
    taskdata.append(['1',(i),str(rater2[i])])
    #taskdata.append(['2',(i),str(rater3[i])])
print(taskdata)



#taskdata=[['0',(i),str(rater1[i])] for i in range(0,len(rater1))]+[['1',(i),str(rater2[i])] for i in range(0,len(rater2))]+[['2',(i),str(rater3[i])] for i in range(0,len(rater3))]
# taskdata = [['1', 5723, 'ORG'],
#             ['2', 5723, 'ORG'],
#             ['1', 55829, 'LOC'],
#             ['2', 55829, 'LOC'],
#             ['1', 259742, 'LOC'],
#             ['2', 259742, 'LOC'],
#             ['1', 269340, 'LOC'],
#             ['2', 269340, 'LOC']]


ratingtask = agreement.AnnotationTask(data=taskdata)
print("kappa " +str(ratingtask.kappa()))
print("fleiss " + str(ratingtask.multi_kappa()))
print("alpha " +str(ratingtask.alpha()))
print("scotts " + str(ratingtask.pi()))