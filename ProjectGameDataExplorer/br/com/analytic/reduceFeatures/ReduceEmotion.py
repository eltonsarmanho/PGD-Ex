
from sklearn.decomposition import  PCA

import numpy as np


class ReduceEmotion:

    def __init__(self):
        pass;
     
    def runPCA(self,X):
        
       
        pca = PCA(n_components=0.95)
           
        principalComponents = pca.fit_transform(X)
        arrayPCA = principalComponents;
        if(len(pca.explained_variance_ratio_) == 2):
            EVR_1, EVR_2 = pca.explained_variance_ratio_
            EVR_3 = 0;
        elif (len(pca.explained_variance_ratio_) == 3): 
            EVR_1, EVR_2, EVR_3 = pca.explained_variance_ratio_ 
            if(EVR_3 > EVR_2):
                EVR_2 = EVR_3;
                principalComponents[:,1] = principalComponents[:,2]   
        else:           
            EVR_1 = pca.explained_variance_ratio_ 
            EVR_2 = 0;
            EVR_3 = 0;
        
        #Cumulative Variance explains
        #var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
        #CVE_1,CVE_2 = var1;
        
        if(EVR_1 > EVR_2):
            EVR = EVR_1;
            arrayPCA = principalComponents[:,0];#Array of the Component1 
        else: 
            EVR = EVR_2;
            arrayPCA = principalComponents[:,1];#Array of the Component2
       
        min = np.min(arrayPCA)
        if(min < 0 ):
            min = np.abs(min)
            arrayPCA = [x+min for x in arrayPCA];
      
        return arrayPCA;
        
  

    