'''
Created on 18 de dez de 2019

@author: eltonss
'''
from datetime import datetime
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer
from matplotlib.pyplot import plot
from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import sys

from pyphysio import EvenlySignal

from ProjectGameDataExplorer.br.com.util import UnixTime, SourceData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyphysio.filters.Filters as flt
import seaborn as sns
from datetime import timezone
import itertools
from sklearn import preprocessing

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
class ANNPX(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor     
       '''
    def loadData(self):
        self.dataset = pd.read_csv("/home/eltonss/Desktop/dataset.csv", sep=',') 
        self.dataset = self.dataset[pd.notnull(self.dataset)]
        self.dataset = self.dataset.fillna(0)
        filter = self.dataset['Experience'] != 'X';
        self.datasetPX = self.dataset[filter]
        self.datasetPX = self.datasetPX.drop('Player', 1)
        self.datasetPX = self.datasetPX.drop('Session', 1)
        self.datasetPX = self.datasetPX.drop('Interval Initial', 1)
        self.datasetPX = self.datasetPX.drop('Interval Final', 1)
        self.datasetPX = self.datasetPX.drop('Rater 1', 1)
        self.datasetPX = self.datasetPX.drop('Rater 2', 1)
        self.datasetPX = self.datasetPX.drop('Rater 3', 1)
        self.datasetPX = self.datasetPX.drop('Rater 4', 1)
        self.datasetPX = self.datasetPX.drop('Rater 5', 1)
        self.datasetPX = self.datasetPX.drop('Rater 6', 1)
        self.datasetPX = self.datasetPX.drop('Event', 1)
        
        #print(self.datasetPX)
        
    def run(self):
        X,y, X_train, X_test, y_train, y_test = self.preprocessing()
        #To run the classification process
        y_predict = self.runClassification(X,y, X_train, X_test, y_train, y_test)
        #Using  Confusion Matrix for a prediction test
        self.evaluationMetric(y_test, y_predict); 
        
    def preprocessing(self):
        #I'm codifying categorical value to numeric value
        le_sex = preprocessing.LabelEncoder()
        new = self.datasetPX.copy()
        le_sex.fit(['Raiva','Desgosto','Medo','Ansiedade','Tristeza','Desejo','Calma','Felicidade'])
        print(Counter(new['Experience']))
        new['Experience'] = le_sex.transform(new['Experience']) 
        #print(Counter(new['Experience']))
        #To convert for array        
        y = np.asarray(new['Experience'])
        X = np.asarray(self.datasetPX.copy().drop('Experience', 1))#Remove the predict variable
        
        #Split dataset
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=4)
        print ('Train set:', X_train.shape,  y_train.shape)
        print ('Test set:', X_test.shape,  y_test.shape)
        
        return X,y, X_train, X_test, y_train, y_test;
     
    def modelEvaluation(self,clf, X, y):
        scores = cross_val_score(clf, X, y, scoring='f1_macro' ,cv=5)
        #The mean score and the 95% confidence interval of the score estimate are hence given by:
        #print("The mean score and the 95% confidence interval of the score estimate are hence given by:")
        print("The mean score and the 95%% confidence interval of the score estimate are hence given by Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    def evaluationMetric(self,y_test, yhat):   
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1,2,3,4,5,6,7])
        np.set_printoptions(precision=2)
        print (classification_report(y_test, yhat))
        
        # Plot non-normalized confusion matrix
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=['Raiva(0)','Desgosto(1)','Medo(2)','Ansiedade(3)','Tristeza(4)',
                                                        'Desejo(5)','Calma(6)','Felicidade(7)'],
                              normalize= False,  title='Confusion matrix')
        
    def runClassification(self,X,y,X_train, X_test, y_train, y_test):
        #Creating classifier
        clf = svm.SVC(kernel='poly')
        
        #Training and fitting the classifier 
        clf.fit(X_train, y_train) 
        print("Finish Fitting")
        #Using KFold
        self.modelEvaluation(clf, X, y)
        
        #A prediction test
        y_predict = clf.predict(X_test)
        print("Our y_predict is %s " % (y_predict))
        
        return y_predict;    
     
    def plot_confusion_matrix(self,cm, classes,normalize=False,title='Confusion matrix',cmap='Blues'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
if __name__ == '__main__':
    projeto = ANNPX();
    projeto.loadData()
    projeto.run()
        