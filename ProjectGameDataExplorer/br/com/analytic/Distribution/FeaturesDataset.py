from builtins import len
from collections import Counter
from collections import defaultdict
from numpy import set_printoptions
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition.kernel_pca import KernelPCA
from sklearn.decomposition.nmf import NMF
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics.scorer import SCORERS
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import RepeatedStratifiedKFold
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def showPlot(df):
        #self.dataset.boxplot(column=['EspessuraFibras', 'EspessuraLumem', 'EspessuraParede'])
    df.plot(kind='kde',subplots= True,layout = (11,11))
        #self.checkFA()
    plt.grid()
    plt.show()
    # The PCA model
    # pca = PCA(n_components=2) # estimate only 2 PCs
    # X_new = pca.fit_transform(X) # project the original data into the PCA space
    # fig, axes = plt.subplots(1,2)
    # print(pca.explained_variance_ratio_)
    # axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
    # axes[1].set_xlabel('PC1')
    # axes[1].set_ylabel('PC2')
    # axes[1].set_title('After PCA')
    # plt.show()
    # define dataset
    # define the pipeline
    sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
    
def runUnivariateSelection(X,y,score_func=f_classif,n_features=5):
       
        test = SelectKBest( score_func, k=n_features)
        fit = test.fit(X,y)
        # summarize scores
        set_printoptions(precision=8)
        #print(fit.scores_)
        features = fit.transform(X)
        # summarize selected features
        cols = fit.get_support(indices=True)
        print("Select columns: %s" % (cols))
        return cols;

def executePipelineMain(X,y):
    #Todas features = Face+ EDA+ BVP
    #Best knn : 0.593407 using {'classify__n_neighbors': 4, 'classify__weights': 'uniform', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=None, kernel='cosine',
    # kernel_params=None, max_iter=None, n_components=15, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__kernel': 'cosine', 'reduce_dim__n_components': 15}
    #Best SVM : 0.615385 using {'classify__C': 100, 'classify__kernel': 'rbf', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=0.1, kernel='linear',
    #kernel_params=None, max_iter=None, n_components=15, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__n_components': 15}
    #Best xgb : 0.593407 using {'classify__gamma': 0.01, 'classify__learning_rate': 0.1, 'classify__n_estimators': 200, 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
    #n_components=15, random_state=None, shuffle=False, solver='cd',
    #tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.0, 'reduce_dim__n_components': 15}
    
    ##Todas features = Face+ EDA
    #Best SVM : 0.593407 using {'classify__C': 10, 'classify__kernel': 'rbf', 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.1, max_iter=200,
    # 'reduce_dim__l1_ratio': 0.1, 'reduce_dim__n_components': 10}
    #Todas features = Face+ BVP
    #Best knn : 0.593407 using {'classify__n_neighbors': 7, 'classify__weights': 'distance', 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.5172413793103449,
    #max_iter=200, n_components=20, random_state=None, shuffle=False,
    #solver='cd', tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.5172413793103449, 'reduce_dim__n_components': 20}
    #Best xgb : 0.582418 using {'classify__gamma': 0.0001, 'classify__learning_rate': 0.1, 'classify__n_estimators': 300, 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.9310344827586207,
    #max_iter=200, n_components=20, random_state=None, shuffle=False,
    #solver='cd', tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.9310344827586207, 'reduce_dim__n_components': 20}
    
    #Todas features = Face+ BVP
    #Best SVM : 0.556777 using {'classify__C': 10, 'classify__kernel': 'rbf', 'reduce_dim': SelectKBest(k=20, score_func=<function f_classif at 0x7fa0bd4e1730>), 'reduce_dim__k': 20}
    #Best knn : 0.567766 using {'classify__n_neighbors': 5, 'classify__weights': 'uniform', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=None, kernel='sigmoid',
    #kernel_params=None, max_iter=None, n_components=10, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__kernel': 'sigmoid', 'reduce_dim__n_components': 10}
    #Best xgb : 0.582418 using {'classify__gamma': 0.0001, 'classify__learning_rate': 0.1, 'classify__n_estimators': 300, 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.9310344827586207,
    #max_iter=200, n_components=20, random_state=None, shuffle=False,
    #solver='cd', tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.9310344827586207, 'reduce_dim__n_components': 20}
    
    #Todas features = EDA+ BVP
    #Best SVM : 0.560440 using {'classify__C': 1, 'classify__kernel': 'rbf', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    # fit_inverse_transform=False, gamma=0.1, kernel='linear',
    # kernel_params=None, max_iter=None, n_components=10, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__n_components': 10}
    
    #Best knn : 0.575092 using {'classify__n_neighbors': 14, 'classify__weights': 'distance', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=None, kernel='poly',
    #kernel_params=None, max_iter=None, n_components=15, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__kernel': 'poly', 'reduce_dim__n_components': 15}
    
    #Best xgb : 0.531136 using {'classify__gamma': 0.001, 'classify__learning_rate': 0.1, 'classify__n_estimators': 50, 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=None, kernel='poly',
    #kernel_params=None, max_iter=None, n_components=10, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__kernel': 'poly', 'reduce_dim__n_components': 10}
    
    #EDA ## 11 var
    #Best xgb : 0.560440 using {'classify__gamma': 0.1, 'classify__learning_rate': 0.1, 'classify__n_estimators': 100, 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
    #n_components=8, random_state=None, shuffle=False, solver='cd',
    #tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.0, 'reduce_dim__n_components': 8}
    #Best knn : 0.582418 using {'classify__n_neighbors': 7, 'classify__weights': 'distance', 'reduce_dim': NMF(alpha=0.0, beta_loss='frobenius', init=None, l1_ratio=0.0, max_iter=200,
    #n_components=8, random_state=None, shuffle=False, solver='cd',
    #tol=0.0001, verbose=0), 'reduce_dim__l1_ratio': 0.0, 'reduce_dim__n_components': 8}
    #Best SVM : 0.582418 using {'classify__C': 10, 'classify__kernel': 'rbf', 'reduce_dim': KernelPCA(alpha=1.0, coef0=1, copy_X=True, degree=3, eigen_solver='auto',
    #fit_inverse_transform=False, gamma=0.1, kernel='linear',
    #kernel_params=None, max_iter=None, n_components=8, n_jobs=None,
    #random_state=None, remove_zero_eig=False, tol=0), 'reduce_dim__n_components': 8}
    
    steps = [('kpca', KernelPCA(n_components=2,fit_inverse_transform=True, n_jobs=-1)), ('SVM', svm.SVC())]
    model = Pipeline(steps);
    X_train, X_test, y_train, y_test  = preprocessingData(X, y)
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)
    # Compute the accuracy: accuracy
    accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
    print("svm_classification accuracy: %f" % (accuracy));
    print(preds_prob)
    
def runPipeline(X,y):
    
    N_FEATURES_OPTIONS = [10,15,20]#Sempre 3 var
    N_FEATURES_OPTIONS = [2,5,8]#Sempre 3 var
    
    kernelsPCA = ["linear" , "poly" , "rbf" , "sigmoid" , "cosine" ]
    #xgb
    learning_rate =[0.1,0.01,0.001,0.0001]#Sempre 4 var
    n_estimators=[50,100,200,300]#Sempre 4 var                      
    gamma = [0.1,0.01,0.001,0.0001]#Sempre 4 var    
    #SVM
    kernels= ['linear','rbf', 'poly', 'sigmoid']
    C_OPTIONS = [1, 10, 100,1000]#Sempre 4 var
    #KNN
    k_range = list(range(3, 30))
    weight_options = ['uniform', 'distance']

    classify = 'SVM'
    classify = 'xgb'
    classify = 'knn'

    if(classify == 'SVM'):
        pipe = Pipeline([
                        
                ('reduce_dim',None),
                ('classify',svm.SVC(gamma='scale'))        
    ])
        param_grid = [
        {
            'reduce_dim': [SelectKBest(f_classif)],'reduce_dim__k':N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,'classify__kernel': kernels           
        },
        {            
            'reduce_dim': [KernelPCA(gamma=0.1)],'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,'classify__kernel': kernels               
        },
        {
            'reduce_dim': [NMF()],'reduce_dim__l1_ratio':[0.1],'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__C': C_OPTIONS,'classify__kernel': kernels
        }
    ]
    elif(classify == 'knn'):
        pipe = Pipeline([        
            ('reduce_dim',None),('classify',KNeighborsClassifier())
        ])
        param_grid = [
        {            
           'reduce_dim': [SelectKBest(f_classif)],'reduce_dim__k':N_FEATURES_OPTIONS,
           'classify__n_neighbors':k_range ,'classify__weights':weight_options        
        },
        {            
            'reduce_dim': [KernelPCA()],'reduce_dim__n_components': N_FEATURES_OPTIONS,'reduce_dim__kernel': kernelsPCA,           
            'classify__n_neighbors':k_range ,'classify__weights':weight_options 
        },
        {
            'reduce_dim': [NMF()],'reduce_dim__l1_ratio':np.linspace(0,1,30),'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__n_neighbors':k_range ,'classify__weights':weight_options 
        }
    ]
    else:    
        pipe = Pipeline([        
            ('reduce_dim',None),('classify',XGBClassifier(99))
        ])
        param_grid = [
        {            
           'reduce_dim': [SelectKBest(f_classif)],'reduce_dim__k':N_FEATURES_OPTIONS,
           'classify__gamma':gamma ,'classify__learning_rate': learning_rate,'classify__n_estimators': n_estimators          
        },
        {            
            'reduce_dim': [KernelPCA()],'reduce_dim__n_components': N_FEATURES_OPTIONS,'reduce_dim__kernel': kernelsPCA,           
            'classify__gamma':gamma ,'classify__learning_rate': learning_rate,'classify__n_estimators': n_estimators
        },
        {
            'reduce_dim': [NMF()],'reduce_dim__l1_ratio':np.linspace(0,1,30),'reduce_dim__n_components': N_FEATURES_OPTIONS,
            'classify__gamma':gamma ,'classify__learning_rate': learning_rate,'classify__n_estimators': n_estimators
        }
    ]

    X_train, X_test, y_train, y_test  = preprocessingData(X, y)
    reducer_labels = ['SelectKBest','KernelPCA','NMF']#Equal instance of class
    grid = GridSearchCV(pipe, n_jobs=1, param_grid=param_grid)   
    grid.fit(X_train, y_train)
    
    print("Best %s : %f using %s" % (classify,grid.best_score_, grid.best_params_))
  
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    
    dict_reduce = defaultdict(dict)
    dict_reduce_std = defaultdict(dict)
    for reduce in reducer_labels:
        for features in N_FEATURES_OPTIONS:
            dict_reduce[reduce][str(features)] = 0
            dict_reduce_std[reduce][str(features)] = 0
    for mean, stdev, param in zip(means, stds, params):
       
        if('reduce_dim' in param):
            #print("%f (%f) with: %r" % (mean, stdev, param))
            if (isinstance(param['reduce_dim'], SelectKBest)):               
                if(dict_reduce['SelectKBest'][str(param['reduce_dim__k'])] < mean):                
                    dict_reduce['SelectKBest'][str(param['reduce_dim__k'])] = mean
                    dict_reduce_std['SelectKBest'][str(param['reduce_dim__k'])] = stdev
            elif (isinstance(param['reduce_dim'], KernelPCA)):
                if(dict_reduce['KernelPCA'][str(param['reduce_dim__n_components'])] < mean):                
                    dict_reduce['KernelPCA'][str(param['reduce_dim__n_components'])] = mean
                    dict_reduce_std['KernelPCA'][str(param['reduce_dim__n_components'])] = stdev
            elif (isinstance(param['reduce_dim'], NMF)):
                if(dict_reduce['NMF'][str(param['reduce_dim__n_components'])] < mean):                
                    dict_reduce['NMF'][str(param['reduce_dim__n_components'])] = mean
                    dict_reduce_std['NMF'][str(param['reduce_dim__n_components'])] = stdev
       
    bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
                   (len(reducer_labels) + 1) + .5)
    
    plt.figure()
    COLORS = 'bgrcmyk'
    for i, (label) in enumerate(reducer_labels):
        plt.bar(bar_offsets+i, dict_reduce[label].values(), yerr=dict_reduce_std[label].values() ,label=label, color=COLORS[i])
        
    
    plt.title("Comparing feature reduction techniques with "+classify)
    plt.xlabel('Reduced number of features')
    plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
    plt.ylabel('classification accuracy')
    plt.ylim((0, 1))
    plt.legend(loc='upper left')
    
    plt.show()




def svc_param_selection(X, y, nfolds = 3):
        #72% {'C': 0.0001, 'gamma': 1, 'kernel': 'poly'}
        #Best SVM: 0.503856 using {'C': 1, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 1, 'kernel': 'rbf'}

        Cs =     [0.001, 0.01, 0.1, 1, 10,100]
        gammas = [0.0001,0.001, 0.01, 0.1, 1]
        #decision_function_shape = ['ovo', 'ovr']
        param_grid = {'C': Cs,                      
                      'gamma': gammas,
                      'kernel': ['rbf', 'poly', 'sigmoid']}
        #param_grid = {'C': Cs, 'gamma' : gammas, 'degree':[1,2,3,4,5,6],
        #              'kernel' : ['linear','poly','sigmoid','rbf'],'decision_function_shape': decision_function_shape}
        grid_search = GridSearchCV(svm.SVC(probability=True),param_grid=param_grid,refit=True, cv=nfolds)
        grid_result = grid_search.fit(X, y)
        print("Best SVM: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
def svm_classification(X,y):

    model = svm.SVC(probability=True,)
    X_train, X_test, y_train, y_test  = preprocessingData(X, y)
    model.fit(X_train,y_train)
    # Predict the labels of the test set: preds
    #modelEvaluation(model,X_train,y_train)
    preds = model.predict(X_test)
    
    # Compute the accuracy: accuracy
    accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
    print("svm_classification accuracy: %f" % (accuracy));
    
def xgbc_param_selection(X,y):
    #Best: 0.732099 using {'gamma': 0.001, 'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'scale_pos_weight': 1}

    model =  XGBClassifier()
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    # define grid
    weights = [1, 99]
    param_grid = dict(
                      learning_rate =[0.1,0.01,0.001],
                      n_estimators=[100,200,300],                      
                      gamma = [0.001,0.01,0.1,1])
  
    # define grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    # execute the grid search
    grid_result = grid.fit(X, y)
    # report the best configuration
    print("Best XGBC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def xgbc_classification(X,y):
    #Best XGBC: 0.590598 using {'gamma': 0.001, 'learning_rate': 0.1, 'n_estimators': 300} = 3 classes dados EDA e Face - 19 features
    #Best SVM: 0.517949 using {'C': 0.1, 'gamma': 1, 'kernel': 'poly'}= 3 classes dados EDA e Face - 19 features

    xg_cl =  XGBClassifier(scale_pos_weight=1,gamma = 0.001,learning_rate= 0.01,max_depth = 5,n_estimators=100 )
    X_train, X_test, y_train, y_test  = preprocessingData(X, y)
    xg_cl.fit(X_train,y_train)
    # Predict the labels of the test set: preds
    preds = xg_cl.predict(X_test)
    instances =  xg_cl.predict_proba(X_test).round(2);
    #print(instances)
    # Compute the accuracy: accuracy
    accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
    print("xgbc_classification accuracy: %f" % (accuracy))
    return preds, xg_cl, y_test




def preprocessingData(X,y):
        
        #X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
        X = preprocessing.MinMaxScaler().fit(X).transform(X.astype(float))
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print ('Train set:', X_train.shape, y_train.shape)
        print ('Test set:', X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test; 


    
dataset = pd.read_csv("/home/elton/Desktop/Dataset/dataset.csv", sep=',') 
#dataset_validation = [30,31,32,33,34,35,36,37,41,42,43,44,45,46,47,48,49,50,51]
dataset_train = [2,3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 37, 43, 45, 47]
Players_test =  [31,32,33,34,35,36,41,42,44,48,49,50,51]

dataset = dataset[dataset['Player'].isin(dataset_train)]
filter = dataset['Experience'].isin(['Negative','Neutral','Positive'])
dataset = dataset[filter]


features_remove =['Player','Session','Interval Initial','Interval Final','Event','Rater 1','Rater 2','Rater 3','Rater 4','Rater 5','Rater 6',
                 'NegativeEmotion_maximum','NegativeEmotion_minimum','NegativeEmotion_sd','PositiveEmotion_maximum','PositiveEmotion_minimum','PositiveEmotion_sd','NegativeEmotion_mean','NegativeEmotion_range','PositiveEmotion_mean','PositiveEmotion_range',                  
                 #'pha_StDev','pha_PeaksMax','pha_DurationMean','pha_PeaksMin','pha_SlopeMean','pha_AUC','pha_Mean','pha_Range','pha_PeaksMean','pha_PeaksNum',
                  'IBI_Min','IBI_Max','IBI_pnn10','IBI_Median','IBI_pnn25','IBI_pnn50','IBI_sd12','IBI_sdell','IBI_sd1','IBI_sd2','IBI_RMSSD', 'IBI_Mean','IBI_RRstd','IBI_SDSD'
                  ]
  
df = dataset.copy().drop(features_remove, axis = 1)
print(df.info())
Y = dataset['Experience']        
le_sex = preprocessing.LabelEncoder()
le_sex.fit(list(set(Y)))
print(Counter(Y))  
Y = le_sex.transform(Y) 
print(Counter(Y))  
   
y = np.asarray(Y)  
X = np.asarray(df.copy().drop('Experience', 1))

runPipeline(X,y)   