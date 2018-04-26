# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the dataset
train_dataset=pd.read_csv('sales_train_v2.csv')
test_dataset=pd.read_csv('test.csv')

#describing
train_dataset.describe()
test_dataset.describe()

#describing
train_dataset.info()
test_dataset.info()

#checking if the dataset has any NAN values
train_dataset.isnull().sum()
test_dataset.isnull().sum()



#splitting the dataset into dependent and independent variables
x=train_dataset.iloc[:,[2,3]].values
y=train_dataset.iloc[:,5].values


#feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc_x= MinMaxScaler()
x=sc_x.fit_transform(x)

#spliting dataset for 30timesteps
x_train=[]
y_train=[]
for i in range(30,len(x)):
    x_train.append(x[i-30:i,[0,1]])    
    y_train.append(y[i])
x_train=np.array(x_train)
y_train=np.array(y_train)


#reshaping
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],2))

#building lstm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

#Intializing
regressor=Sequential()

#LSTM
regressor.add(LSTM(units=50,activation='sigmoid',input_shape=(None,2),return_sequences=True))

#Another Lstm
regressor.add(LSTM(units=50,activation='sigmoid'))

#output layer
regressor.add(Dense(units=1))

#compilation
regressor.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

regressor.fit(x_train,y_train)

#prediction
test_set=test_dataset.iloc[:,[1,2]].values
test_set=sc_x.fit_transform(test_set)
test_set=np.concatenate((x[0:len(x)],test_set),axis=0)
x_test=[]
for j in range(len(x),len(test_set)):
    x_test.append(test_set[i-30:i,0:2])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],2))
y_pred=regressor.predict(x_test)
prediction=pd.DataFrame(y_pred)
prediction.to_csv('/output/predicted.csv')
