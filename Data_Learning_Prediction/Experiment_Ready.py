
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split


# In[ ]:


import os
import sys


# In[ ]:


from sklearn import metrics
import statsmodels
from math import sqrt
from math import log
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

import random
from random import gauss
from random import seed
datetime.now().strftime('%m-%d %H:%M')


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


# In[ ]:


def Xscaler(X,y,scalertype):

    if scalertype=="Normalizer":
        X=pd.DataFrame(Normalizer().fit_transform(X,y))
        print("normalize")
    elif scalertype=="MinMaxScaler":
        X=pd.DataFrame(MinMaxScaler().fit_transform(X,y))
        print("minmax")
    elif scalertype=="MaxAbsScaler":
        X=pd.DataFrame(MaxAbsScaler().fit_transform(X,y))
        print("maxabs")
    elif scalertype=="RobustScaler":
        X=pd.DataFrame(RobustScaler().fit_transform(X,y))
    elif scalertype=="StandardScaler":
        X=pd.DataFrame(StandardScaler().fit_transform(X,y))
    return X


# In[ ]:


def X_Y_scaler_train_test_Split(X,y,Z,random=42):

    values = X.values
    values = values.astype('float32')
    X_Column_X_Column_Names=X.columns
    
    scaler_X= MinMaxScaler(feature_range=(0, 1))

    scaled_value_X = scaler_X.fit_transform(values)
    scaled_value_X = pd.DataFrame(data=scaled_value_X[:,:])
    scaled_value_X.columns=X_Column_X_Column_Names
    
    values = y.values
    values = values.astype('float32')
    values = values.reshape(-1, 1)
    
    scaler_y= MinMaxScaler(feature_range=(0, 1))

    scaled_value_y = scaler_y.fit_transform(values)
    scaled_value_y = pd.DataFrame(data=scaled_value_y)

    X_train, X_test, y_train, y_test = train_test_split(scaled_value_X,scaled_value_y,test_size=0.2,random_state=42,stratify=Z['Month'])

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y, scaled_value_X, scaled_value_y


# In[ ]:


# For Random Forest with variable tuning 

def randomforest(X_train, X_test, y_train, y_test,scaler_y,rand=50,is_random_fixed='TRUE',est=10,min_leaf=1,feat='auto',max_leaf=None,min_weight=0.0,min_impurity=1e-07):
    from sklearn.model_selection import cross_val_score   
    from sklearn.model_selection import cross_val_predict
    
    if is_random_fixed == 'TRUE': 
        rs=rand
    else :
        rs=random.randint(1,100)
    #print('randomforest rs=',rs)
    rfc=RandomForestRegressor(n_estimators=est,
                              min_samples_leaf=min_leaf,
                              random_state =rs,
                              max_features=feat,
                              max_leaf_nodes=max_leaf,
                              min_weight_fraction_leaf =min_weight,
                              min_impurity_decrease=min_impurity
                            )
   
    RandomForestRegressor.fit(rfc,X_train,y_train)
    
    y_predict_test = rfc.predict(X_test)
    y_predict_train = rfc.predict(X_train)
    
    result_test=inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,'NO')
    result_train=inverse_scale_and_graph_Y_predict_and_test(y_predict_train,y_train,scaler_y,'NO')
    
    return result_test, result_train



# In[ ]:


def inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,plot_on):

    y_predict_test=y_predict_test.reshape(-1, 1)
    inv_y_predict_test = scaler_y.inverse_transform(y_predict_test)
    predictions=inv_y_predict_test

  
    inv_y_test = scaler_y.inverse_transform(y_test)
    inv_y_test = inv_y_test[:,0]

    MAE=int(metrics.mean_absolute_error(inv_y_test, predictions))
    MSE=int(sqrt(metrics.mean_squared_error(inv_y_test, predictions)))
    flatten=predictions.flatten()
    R2=int(1000*pearsonr(inv_y_test,flatten )[0]**2)/1000
#    R2=int(1000*(metrics.r2_score(inv_y_test, predictions)))/1000
    
    
    if plot_on =='YES':
        plt.scatter(inv_y_test,predictions)
    
    return MAE,MSE,R2


# In[ ]:


def experiment_RandomForest(repeats,
                  X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',
                  est=10,min_leaf=1,feat='auto',max_leaf=None,min_weight=0.0,min_impurity=1e-07):
    
    error_rmse = list()
    error_R2 = list()
    
    for r in range(repeats):

        result=randomforest(X_train, X_test, y_train, y_test,scaler_y,
                            rand=rand,is_random_fixed=is_random_fixed,
                            est=est,min_leaf=min_leaf,feat=feat,max_leaf=max_leaf,
                            min_weight=min_weight,min_impurity=min_impurity)

    
      
        rmse_test=result[0][1]
        R2_test=result[0][2]
        
        rmse_train=result[1][0]
        R2_train=result[1][1]
        
        error_rmse.append(rmse_test)
        error_R2.append(R2_test)
    
    return error_rmse, error_R2


# In[ ]:


from sklearn.neural_network import MLPRegressor


# In[ ]:


def NeuralNetwork(X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',
                  activ='relu', alph=0.0001, slv='adam', max_iteration=200,  hidden_layer=(30,30)):
    
    
    if is_random_fixed == 'TRUE': 
        rs=rand
    else :
        rs=random.randint(1,100)
    #print('neuralnetwork rs=',rs)   

    MLP = MLPRegressor(
                            activation=activ,
                            random_state =rs,                      
                            alpha = alph,
                            solver=slv ,
                            max_iter=max_iteration,  
                            hidden_layer_sizes=hidden_layer
                        )


    MLPRegressor.fit(MLP,X_train,y_train)
    
    y_predict_test = MLP.predict(X_test)
    y_predict_train = MLP.predict(X_train)
    
    result_test=inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,'NO')
    result_train=inverse_scale_and_graph_Y_predict_and_test(y_predict_train,y_train,scaler_y,'NO')
   
    return result_test, result_train


# In[ ]:


def experiment_NN(repeats,
                  X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',
                  activ='relu',alph=0.0001, max_iteration=200, slv='adam',  hidden_layer=(30,30)):


    error_rmse = list()
    error_R2 = list()
    
    for r in range(repeats):
            
        result = NeuralNetwork(X_train, X_test, y_train, y_test,scaler_y,
                               rand=rand,is_random_fixed=is_random_fixed,
                               activ=activ,alph=alph,max_iteration=max_iteration, slv=slv, hidden_layer=hidden_layer)
        
        
        rmse_test=result[0][1]
        R2_test=result[0][2]
        
        rmse_train=result[1][0]
        R2_train=result[1][1]
        
        error_rmse.append(rmse_test)
        error_R2.append(R2_test)
    
    return error_rmse, error_R2

