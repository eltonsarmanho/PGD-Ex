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
from sklearn.metrics.scorer import SCORERS
from factor_analyzer.factor_analyzer import calculate_kmo, FactorAnalyzer

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

class ANNPX(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor     
       '''

    def loadData(self):
        self.dataset = pd.read_csv("/home/elton/Desktop/datasetUpdatecopy.csv", sep=',') 
        self.dataset = self.dataset[pd.notnull(self.dataset)]
        self.dataset = self.dataset.fillna(0)
        filter = ((self.dataset['Experience'] != 'X'));
        #filter = ((self.dataset['Experience'] != 'X') );
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
        
      
   
        #----- self.datasetPX = self.datasetPX.drop('NegativeEmotion_maximum',1)
        #----- self.datasetPX = self.datasetPX.drop('NegativeEmotion_minimum',1)
        #-------- self.datasetPX = self.datasetPX.drop('NegativeEmotion_mean',1)
        #------- self.datasetPX = self.datasetPX.drop('NegativeEmotion_range',1)
        #---------- self.datasetPX = self.datasetPX.drop('NegativeEmotion_sd',1)
        #----- self.datasetPX = self.datasetPX.drop('PositiveEmotion_maximum',1)
        #----- self.datasetPX = self.datasetPX.drop('PositiveEmotion_minimum',1)
        #-------- self.datasetPX = self.datasetPX.drop('PositiveEmotion_mean',1)
        #------- self.datasetPX = self.datasetPX.drop('PositiveEmotion_range',1)
        #---------- self.datasetPX = self.datasetPX.drop('PositiveEmotion_sd',1)
       
#         self.datasetPX = self.datasetPX.drop('pha_Mean',1)
#         self.datasetPX = self.datasetPX.drop('pha_StDev',1)
#         self.datasetPX = self.datasetPX.drop('pha_Range',1)
#         self.datasetPX = self.datasetPX.drop('pha_PeaksMax',1)
#         self.datasetPX = self.datasetPX.drop('pha_PeaksMin',1)
#         self.datasetPX = self.datasetPX.drop('pha_PeaksMean',1)
#         self.datasetPX = self.datasetPX.drop('pha_PeaksNum',1)
#         self.datasetPX = self.datasetPX.drop('pha_DurationMean',1)
#         self.datasetPX = self.datasetPX.drop('pha_SlopeMean',1)
#         self.datasetPX = self.datasetPX.drop('pha_AUC',1)
        
#         self.datasetPX = self.datasetPX.drop('IBI_RMSSD',1)
#         self.datasetPX = self.datasetPX.drop('IBI_SDSD',1)
#         self.datasetPX = self.datasetPX.drop('IBI_Mean',1)
#         self.datasetPX = self.datasetPX.drop('IBI_RRstd',1)
#         self.datasetPX = self.datasetPX.drop('IBI_Median',1)
#         self.datasetPX = self.datasetPX.drop('IBI_pnn10',1)
#         self.datasetPX = self.datasetPX.drop('IBI_pnn25',1)
#         self.datasetPX = self.datasetPX.drop('IBI_pnn50',1)
#         self.datasetPX = self.datasetPX.drop('IBI_Min',1)
#         self.datasetPX = self.datasetPX.drop('IBI_Max',1)
#         self.datasetPX = self.datasetPX.drop('IBI_sd1',1)
#         self.datasetPX = self.datasetPX.drop('IBI_sd2',1)
#         self.datasetPX = self.datasetPX.drop('IBI_sd12',1)
#         self.datasetPX = self.datasetPX.drop('IBI_sdell',1)
       
        # print(self.datasetPX)
        
    def run(self):
        X, y, X_train, X_test, y_train, y_test = self.preprocessing()
        # To run the classification process
        y_predict = self.runClassificationANN(X, y, X_train, X_test, y_train, y_test)
        # Using  Confusion Matrix for a prediction test
        self.evaluationMetric(y_test, y_predict); 
        
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
        print(df.columns[self.index])
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.xlabel('number of components')
        #plt.ylabel('cumulative explained variance');
        #plt.show()
    
    def preprocessing(self):
        # I'm codifying categorical value to numeric value
        le_sex = preprocessing.LabelEncoder()
        new = self.datasetPX.copy()
        le_sex.fit(list(set(self.datasetPX['Experience'])))
        print(Counter(new['Experience']))      
        new['Experience'] = le_sex.transform(new['Experience']) 
        
        # To convert for array        
        y = np.asarray(new['Experience'])
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
        scores = cross_val_score(clf, X, y, scoring='accuracy' , cv=10)
        # The mean score and the 95% confidence interval of the score estimate are hence given by:
        # print("The mean score and the 95% confidence interval of the score estimate are hence given by:")
        print("The mean score and the 95%% confidence interval of the score estimate are hence given by Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
    def evaluationMetric(self, y_test, yhat):   
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, yhat, labels=range(0,len(set(self.datasetPX['Experience']))))
        np.set_printoptions(precision=2)
        
        print (classification_report(y_test, yhat))
        
        # Plot non-normalized confusion matrix
        plt.figure()
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
    
    def runClassificationANN(self, X, y, X_train, X_test, y_train, y_test):
        # Creating classifier
        clf = MLPClassifier(solver='adam',learning_rate_init=0.001,hidden_layer_sizes=(25,25),
                            max_iter=1000,activation ='identity',warm_start = True)
        #activation : {'identity', 'logistic', 'tanh', 'relu'}, default 'relu'
        #adam or sgd or lbfgs
        # Training and fitting the classifier 
        clf.fit(X_train, y_train) 
        print("Finish Fitting")
        # Using KFold
        self.modelEvaluation(clf, X, y)
        
        # A prediction test
        y_predict = clf.predict(X_test)
        #for instances in clf.predict_proba(X_test):
        #    print(instances*100)
        
        #print("Our y_predict is %s " % (y_predict))
        
        return y_predict;    
     
    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
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
    projeto.analysePCA()
    projeto.run()
        
