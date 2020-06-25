'''
Created on 18 de dez de 2019

@author: eltonss
'''

from collections import Counter

from matplotlib.pyplot import plot
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings

from sklearn.metrics.scorer import SCORERS
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from builtins import len
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
import matplotlib
import os
matplotlib.use('TkAgg')
from sklearn.metrics import roc_auc_score

class ANNPX(object):
    # different learning rate schedules and momentum parameters
    global params;
    #identity,relu,tanh,logistic
   

    def __init__(self):
        '''
        Constructor     
       '''
        matplotlib.rcParams["savefig.directory"] = os.chdir(os.path.dirname('/home/elton/Desktop'))


    def loadData(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/datasetUpdatecopy.csv", sep=',') 
        self.dataset = self.dataset.replace('Felicidade', 'Happiness')
        self.dataset = self.dataset.replace('Raiva', 'Anger')
        self.dataset = self.dataset.replace('Tristeza', 'Sadness')
        self.dataset = self.dataset.replace('Calma', 'Calm')
        self.dataset = self.dataset.dropna()        
        
        filter = ((self.dataset['Experience'] != 'X'));
        self.datasetPX = self.dataset[filter]
        columns = ['Player','Session','Interval Initial','Interval Final',
                   'Rater 1','Rater 2','Rater 3','Rater 4','Rater 5','Rater 6','Event']
        for col in columns:
            self.datasetPX = self.datasetPX.drop(col, 1)
        
        #Habilida analise da VALIDACAO
        self.initDataSetValidacao()
        
    def initDataSetValidacao(self):
        self.datasetValidacao = pd.read_csv("/home/elton/Desktop/datasetResultado_coc_1.csv", sep=',') 
        self.datasetValidacao = self.datasetValidacao.dropna()
        lst = ['Felicidade','Tristeza','Calma','Raiva']

        filter = ((self.datasetValidacao['Experience'].isin(lst)));
        self.datasetValidacao =  self.datasetValidacao[filter]
        self.datasetValidacaoPX = self.datasetValidacao
        
        #self.datasetValidacaoPX = self.datasetValidacaoPX[self.datasetValidacaoPX['Session'] == 1]      
        #print(self.datasetValidacaoPX['Event'])  
        self.datasetValidacaoPX = self.datasetValidacaoPX[self.datasetValidacaoPX['Event'].isin(["Gear Box",])] 
        #self.datasetValidacaoPX = self.datasetValidacaoPX[self.datasetValidacaoPX['Experience'] == 'Raiva'] 
        columns = ['Player','Session','Interval Initial','Interval Final','Event','Predicted','FactorCalma','FactorFelicidade','FactorRaiva','FactorTristeza']
        for col in columns:
            self.datasetValidacaoPX = self.datasetValidacaoPX.drop(col, 1)
           
    def run(self):
        X, y, X_train, X_test, y_train, y_test = self.preprocessing()
        X_test, y_test = self.preprocessingSample(self.datasetValidacaoPX);
        
        # To run the classification process
        #self.runPerformClassificationANN(X, y, X_train, X_test, y_train, y_test)
        
        y_predict, instances = self.runClassificationANN(X, y, X_train, X_test, y_train, y_test)
        
        #self.processingResult(y_predict, instances)        
        # Using  Confusion Matrix for a prediction test
        #self.evaluationMetric(y_test, y_predict); 
        
    def processingResult(self,y_predict,instances):
        count = 0
        instances = instances.tolist()
       
        y_predict_labels = self.le_sex.inverse_transform(y_predict)
        for index, row in self.datasetValidacao.iterrows():
            self.datasetValidacao.loc[index,'Predicted'] = y_predict_labels[count]
            self.datasetValidacao.loc[index,'FactorCalma'] = instances[count][0]
            self.datasetValidacao.loc[index,'FactorFelicidade'] = instances[count][1]
            self.datasetValidacao.loc[index,'FactorRaiva'] = instances[count][2]
            self.datasetValidacao.loc[index,'FactorTristeza'] = instances[count][3]
            count = count+1 
        self.datasetValidacao.to_csv("/home/elton/Desktop/datasetResultado_coc_1.csv", index=False,sep=',', encoding='utf-8')   
    
    def preprocessingSample(self, dataset):
        self.le_sex = preprocessing.LabelEncoder()
        new = dataset.copy()
        self.le_sex.fit(list(set(dataset['Experience'])))
        print(Counter(new['Experience']))  
        new['Experience'] = self.le_sex.transform(new['Experience']) 
        print(Counter(new['Experience']))      

        # To convert for array        
        y_test = np.asarray(new['Experience'])
        df = dataset.copy().drop('Experience', 1)# Remove the predict variable
              
        for name in self.index:
            df = df.drop(df.columns[name], 1)
        X = np.asarray(df)  
        X_test = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
       
        return X_test, y_test;
        
    def analysePCA(self):
        df = self.datasetPX.copy().drop('Experience', 1)
        X = np.asarray(df)
        #print(df.shape)
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        pca = PCA()
        pca.fit(X)
        self.X_pca = pca.transform(X)
       
        #loadings = pca.components_
        self.index = []
        for loadings in pca.components_:
            #print(loadings)
            self.index = np.where(loadings < 0)
            break;
    
    
    def preprocessing(self):
        # I'm codifying categorical value to numeric value
        le_sex = preprocessing.LabelEncoder()
        new = self.datasetPX.copy()
        le_sex.fit(list(set(self.datasetPX['Experience'])))
        print(Counter(new['Experience']))      
        new['Experience'] = le_sex.transform(new['Experience']) 
        print(Counter(new['Experience'])) 
        # To convert for array        
        y = np.asarray(new['Experience'])
        print(y)
        df = self.datasetPX.copy().drop('Experience', 1)# Remove the predict variable
        #Remove features irrelevant
        #print("original shape:   ", df.shape)        
        for name in self.index:
            df = df.drop(df.columns[name], 1)
        #print("transformed shape:", df.shape)
        X = np.asarray(df)  
       
        
        #Data Standardization give data zero mean and unit variance, 
        X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
        print ('Train set:', X_train.shape, y_train.shape)
        print ('Test set:', X_test.shape, y_test.shape)
        return X, y, X_train, X_test, y_train, y_test;
     
    def modelEvaluation(self, clf, X, y):
        #https://scikit-learn.org/stable/modules/model_evaluation.html
        #print(SCORERS.keys())
        scores = cross_val_score(clf, X, y, scoring='accuracy' , cv=5)
        # The mean score and the 95% confidence interval of the score estimate are hence given by:
        # print("The mean score and the 95% confidence interval of the score estimate are hence given by:")
        print("The mean score and the 95%% confidence interval of the score estimate are hence given by Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return (scores.mean(), scores.std() * 2)
        
    def evaluationMetric(self, y_test, yhat):   
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=range(0,len(set(self.datasetPX['Experience']))))
        np.set_printoptions(precision=2)
        
        print (classification_report(y_test, yhat))
        
        # Plot non-normalized confusion matrix
        plt.figure()
        #Counter({'Happiness': 42, 'Calm': 36, 'Sadness': 34, 'Anger': 20})
        #Counter({2: 42, 1: 36, 3: 34, 0: 20})
        #print(cnf_matrix)
        #print(sorted(Counter(self.datasetPX['Experience']).keys()))
        self.plot_confusion_matrix(cnf_matrix, classes=sorted(Counter(self.datasetPX['Experience']).keys()),
                              normalize=True, title='Confusion matrix')
        
    def runClassificationSVM(self, X, y, X_train, X_test, y_train, y_test):
        # Creating classifier
        clf = svm.SVC(kernel='linear',decision_function_shape='ovo')
        #t must be one of 'linear', 'poly', 'rbf', 'sigmoid'
        # Training and fitting the classifier 
        clf.fit(X_train, y_train) 
        print("Finish Fitting")
        # Using KFold
        self.modelEvaluation(clf, X, y)
        
        # A prediction test
        y_predict = clf.predict(X_test)
      
       
        #print("Our y_predict is %s " % (y_predict))
        
        return y_predict;
  
    def svc_param_selection(self,X, y, nfolds):
        Cs =     [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.00001,0.0001,0.001, 0.01, 0.1, 1]
        decision_function_shape = ['ovo', 'ovr']
        param_grid = {'C': Cs, 'gamma' : gammas, 'degree':[1,2,3,4,5,6],
                      'kernel' : ['linear','poly','sigmoid','rbf'],'decision_function_shape': decision_function_shape}
        grid_search = GridSearchCV(svm.SVC(probability=True), param_grid, cv=nfolds)
        grid_search.fit(X, y)
        grid_search.best_params_
        print(grid_search.best_params_)
        print(grid_search.best_score_)
        return grid_search.best_params_

    def runClassificationANN(self, X, y, X_train, X_test, y_train, y_test):
        # Creating classifier
        #clf = MLPClassifier(solver='adam',learning_rate_init=0.001,hidden_layer_sizes=(75,75),
        #                    max_iter=1000,activation ='tanh',momentum=0.9,nesterovs_momentum = False,warm_start = True,verbose=False)
        clf = MLPClassifier(solver='sgd',learning_rate_init=0.01,
                            max_iter=300,activation ='tanh',momentum=0.15)
        #59
        #self.svc_param_selection(X,y, 5)
        clf = svm.SVC(C= 1, decision_function_shape= 'ovo', degree= 1, gamma= 0.001, kernel= 'linear',probability=True)
       
        #clf = MLPClassifier(solver='sgd',learning_rate='constant',momentum= 0.9,learning_rate_init=0.001,hidden_layer_sizes=(30,30),
        #                    max_iter=800,activation ='tanh',nesterovs_momentum = False,warm_start = True,verbose=False)
        
        #activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu' 
        #adam or sgd or lbfgs
        #constant,invscaling
        # Training and fitting the classifier 
        clf.fit(X_train, y_train) 
        #plt.plot(clf.loss_curve_,label="Loss Curve")
        #plt.show()
        print("Finish Fitting")
        # Using KFold
        self.modelEvaluation(clf, X_train, y_train)
      
        # A prediction test
        y_predict = clf.predict(X_test)
        instances =  clf.predict_proba(X_test).round(2);
      
        
        return y_predict,instances;    
     
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm*100
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45,fontsize=14)
        plt.yticks(tick_marks, classes,fontsize=14)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt)+"%",fontsize=14,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label',fontsize=14)
        plt.xlabel('Predicted label',fontsize=14)
        plt.show()
    
    

if __name__ == '__main__':
    projeto = ANNPX();
    projeto.loadData()
    projeto.analysePCA()
    projeto.run()
        
