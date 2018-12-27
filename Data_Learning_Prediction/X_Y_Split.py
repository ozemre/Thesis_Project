
# coding: utf-8

# In[1]:


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




