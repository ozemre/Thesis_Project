
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split


# In[17]:


import os
import sys


# In[18]:


from sklearn import metrics
#import statsmodels
from math import sqrt
from math import log
from math import exp

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

import random
from random import gauss
from random import seed
datetime.now().strftime('%m-%d %H:%M')


# In[19]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


# In[20]:


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


# In[21]:


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


# In[22]:


# For Random Forest with variable tuning 

def randomforest(X_train, X_test, y_train, y_test,scaler_y,
                 rand=50,is_random_fixed='TRUE',dependenttype='same',
                 est=10,min_leaf=1,feat='auto',max_leaf=None,min_weight=0.0,min_impurity=1e-07):
    
    from sklearn.model_selection import cross_val_score   
    from sklearn.model_selection import cross_val_predict
    
    if is_random_fixed == 'TRUE': 
        rs=rand
    else :
        rs=random.randint(1,100)
    print('randomforest rs=',rs)
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
    
    result_test=inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,'NO',dependenttype)
    result_train=inverse_scale_and_graph_Y_predict_and_test(y_predict_train,y_train,scaler_y,'NO',dependenttype)
    
    
  
    return result_test, result_train



# In[1]:


def inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,plot_on,dependenttype):

    y_predict_test=y_predict_test.reshape(-1, 1)
    predictions = scaler_y.inverse_transform(y_predict_test)
    inv_y_predict_test=predictions.flatten()
  

    inv_y_test = scaler_y.inverse_transform(y_test)
    inv_y_test = inv_y_test[:,0]

    MAE=int(metrics.mean_absolute_error(inv_y_test, inv_y_predict_test))
    MSE=int(sqrt(metrics.mean_squared_error(inv_y_test, inv_y_predict_test)))
    
    trained_orginal_R2=int(1000*pearsonr(inv_y_test,inv_y_predict_test )[0]**2)/1000
#    R2=int(1000*(metrics.r2_score(inv_y_test, predictions)))/1000
    
    if dependenttype=='same':
        con_y_test=inv_y_test
        con_y_predict_test=inv_y_predict_test
    elif dependenttype=='log':
        con_y_test=[exp(num) for num in inv_y_test]
        con_y_predict_test=[exp(num) for num in inv_y_predict_test]
    elif dependenttype=='sqrt':
        con_y_test=[num**2 for num in inv_y_test]
        con_y_predict_test=[num**2 for num in inv_y_predict_test]
    
    converted_R2=int(1000*pearsonr(con_y_test,con_y_predict_test )[0]**2)/1000
    
    if plot_on =='YES':
        plt.scatter(con_y_test,con_y_predict_test)
    
    return MAE,MSE,converted_R2,trained_orginal_R2,con_y_test,con_y_predict_test


# In[24]:


def experiment_RandomForest(repeats,
                  X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',dependenttype='same',
                  est=10,min_leaf=1,feat='auto',max_leaf=None,min_weight=0.0,min_impurity=1e-07):
    
    error_R2_original = list()
    error_R2_converted = list()
    
    for r in range(repeats):

        result=randomforest(X_train, X_test, y_train, y_test,scaler_y,
                            rand=rand,is_random_fixed=is_random_fixed,dependenttype=dependenttype,
                            est=est,min_leaf=min_leaf,feat=feat,max_leaf=max_leaf,
                            min_weight=min_weight,min_impurity=min_impurity)

#        rmse_train=result[1][0]
#        R2_train=result[1][1]
      
        R2_test_org=result[0][3]
        R2_test_con=result[0][2]
        
        error_R2_original.append(R2_test_org)
        error_R2_converted.append(R2_test_con)
    
    return error_R2_original, error_R2_converted


# In[25]:


from sklearn.neural_network import MLPRegressor


# In[26]:


def NeuralNetwork(X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',dependenttype='same',
                  activ='relu', alph=0.0001, slv='adam', max_iteration=200,  hidden_layer=(30,30)):
    
    
    if is_random_fixed == 'TRUE': 
        rs=rand
    else :
        rs=random.randint(1,100)
    print('neuralnetwork rs=',rs)   

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
    
    result_test=inverse_scale_and_graph_Y_predict_and_test(y_predict_test,y_test,scaler_y,'NO',dependenttype)
    result_train=inverse_scale_and_graph_Y_predict_and_test(y_predict_train,y_train,scaler_y,'NO',dependenttype)
   
    return result_test, result_train


# In[27]:


def experiment_NN(repeats,
                  X_train, X_test, y_train, y_test,scaler_y,
                  rand=50,is_random_fixed='TRUE',dependenttype='same',
                  activ='relu',alph=0.0001, max_iteration=200, slv='adam',  hidden_layer=(30,30)):


    error_R2_original = list()
    error_R2_converted = list()
    
    for r in range(repeats):
            
        result = NeuralNetwork(X_train, X_test, y_train, y_test,scaler_y,
                               rand=rand,is_random_fixed=is_random_fixed,dependenttype=dependenttype,
                               activ=activ,alph=alph,max_iteration=max_iteration, slv=slv, hidden_layer=hidden_layer)
        
        
#        rmse_train=result[1][0]
#        R2_train=result[1][1]
      
        R2_test_org=result[0][3]
        R2_test_con=result[0][2]
        
        error_R2_original.append(R2_test_org)
        error_R2_converted.append(R2_test_con)
    
    return error_R2_original, error_R2_converted


# # Functions for Feature Selection for Ver#3 End

# In[28]:


# Function #1 for #3 Version for Train Test Split and Feature selection 
# Calculates Importances and R2 scores  only for each iteration
def get_feature_importance_and_R2 (X,y,Z,n_feature,split=5):
    from collections import defaultdict
    
    rf = RandomForestRegressor()
    number_of_split=split
    random_state_options=np.random.randint(1,100,size=number_of_split)

    feature_indices = np.ones((n_feature, number_of_split))
    feature_importances=np.ones((n_feature, number_of_split))

    scores = defaultdict(list)
    feature_std = np.ones((n_feature, number_of_split))
    feature_score=np.zeros((n_feature))

    R2=np.ones(number_of_split)

    for turn in range(number_of_split):
        random=random_state_options[turn]
    
        X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.2,random_state=random,stratify=Z['Month'])

        RandomForestRegressor.fit(rf,X_train, Y_train)    
   
        flatten=rf.predict(X_test).flatten()
        R2[turn]=int(1000*pearsonr(Y_test,flatten )[0]**2)/1000
 
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        feature_importances[:,turn]=importances
        feature_indices[:,turn]=indices

    return R2,feature_importances,feature_indices


# In[29]:


# Function #2 for #3 Version for Train Test Split and Feature selection 
# For each Iteration combinesimportances and R2 scores get 1 end result
def combine_feature_importance_and_R2 (score_coefficient,feature_importances,n_feature,number_of_split):
    feature_score=np.zeros((n_feature))
    for i in range(n_feature):
        feature_score[i]=0
    
        for j in range(number_of_split):
                
            importances_coeff=int((feature_importances[i,j]*10000))/10000
        
            score_coeff=int((score_coefficient[j]+1)*10)/10

            score=score_coeff*(importances_coeff)

            feature_score[i]=feature_score[i]+score
    
    scored_feature_indices = (np.argsort(feature_score)[::-1])

    return scored_feature_indices,feature_score
    


# In[30]:


# Function #3 for #3 Version for Train Test Split and Feature selection 
# This Function first calls Function #1 and then Function #2 
def get_feature_importance_result (X,y,Z,n_feature,number_of_split):
    
    result=get_feature_importance_and_R2 (X,y,Z,n_feature,number_of_split)
    R2=result[0]
    feature_importances=result[1]
    feature_indices=result[2]

    R2_Adj=1-R2
    score_coefficient=n_feature*(R2_Adj - np.max(R2_Adj))/-np.ptp(R2_Adj)

    result=combine_feature_importance_and_R2(score_coefficient,feature_importances,n_feature,number_of_split)
    scored_feature_indices=result[0]
    feature_score=result[1]/number_of_split

    return scored_feature_indices,feature_score


# # Functions for Feature Selection for Ver#3 Start
# 3 Different Functions

# In[ ]:


def format_func(value, tick_number):
    # find number of multiples of pi/2
    N = tick_number # int(np.round(2 * value / np.pi))
    
    if N == 1:
        return "200604"
    elif N == 2:
        return "200712"
    elif N == 3:
        return "200908"
    elif N ==4: 
        return "201104"
    elif N == 5:
        return "201212"
    elif N == 6:
        return "201408"
    elif N == 7: 
        return "201604"
    elif N == 8: 
        return "201712"
    else:
        return ""

