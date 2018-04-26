# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the dataset
train_dataset=pd.read_csv('sales_train_v2.csv')
test_dataset=pd.read_csv('test.csv')

#describing
#train_dataset.describe()
#test_dataset.describe()


#splitting the dataset into dependent and independent variables
x=train_dataset.iloc[:,[2,3]].values
y=train_dataset.iloc[:,5:].values

##Visualzing the dataset using seaborn Pairplot
df_x=pd.DataFrame(train_dataset)
sample=df_x.sample(frac=0.0020,random_state=0)
sns.pairplot(sample)

#visualizing the corelation between the variables
corr=sample.corr()
sns.heatmap(data=corr)


##visualizing in linear plot
#sns.lmplot(x='item_id',y='shop_id',hue='item_cnt_day',data=sample)
#
##visualzing the item id to check how they are distributed
#sns.distplot(sample['item_id'])
#
##visualizing the shopid and item_id to find variety of items in shops 
#sns.jointplot(x='shop_id',y='item_id',data=sample)
#
##visualizing how many items sold by each shop
#sns.barplot(x='shop_id',y='item_cnt_day',data=sample)

##splitting the sample dataet
#x_sample=sample.iloc[:,[2,3]].values
#y_sample=sample.iloc[:,5:].values

#visualizing the item_count_day for any outliers
sns.distplot(train_dataset['item_cnt_day'])


##splitting the dataset in to train and test dataset for sample
#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x_sample,y_sample,test_size=0.24,random_state=0)

###splitting the dataset in to train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.01,random_state=0)


##feature Scaling for sample dataset
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_train=sc_x.fit_transform(x_train)
#x_test=sc_x.transform(x_test)
#sc_y=StandardScaler()
#y_train=sc_y.fit_transform(y_train)
#y_test=sc_y.transform(y_test)

##feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_train=sc_x.fit_transform(x_train)
#x_test=sc_x.transform(x_test)
#sc_y=StandardScaler()
#y_train=sc_y.fit_transform(y_train)
#y_test=sc_y.transform(y_test)

#building multiple regeression model
#from sklearn.linear_model import LinearRegression
#regressor=LinearRegression()                         #Accuracy=0.55
#regressor.fit(x_train,y_train)

#Buidling Xgboost model
from xgboost import XGBRegressor
regressor=XGBRegressor()                           #Accuracy=94
regressor.fit(x_train,y_train)


#building Svm model
#from sklearn.svm import SVR
#regressor=SVR(kernel='rbf')                   #Accuracy=0.071
#regressor.fit(x_train,y_train)


#building Svm model
#from sklearn.svm import SVR
#regressor=SVR(kernel='poly',degree=3)                   #Accuracy=0.098
#regressor.fit(x_train,y_train)

#building Svm model
#from sklearn.svm import SVR
#regressor=SVR(kernel='sigmoid')                   #No very poor model
#regressor.fit(x_train,y_train)

#building Svm model
#from sklearn.svm import SVR
#regressor=SVR(kernel='linear')                   #Acuuracy=0.78
#regressor.fit(x_train,y_train)


#building random forest model
#from sklearn.ensemble import RandomForestRegressor
#regressor=RandomForestRegressor(n_estimators=20,random_state=0)    #Accuracy=6.194
#regressor.fit(x_train,y_train)

#building the decisiontree model
#from sklearn.tree import DecisionTreeRegressor
#regressor=DecisionTreeRegressor(random_state=0)
#regressor.fit(x_train,y_train)


#Test prediction
#y_pred=regressor.predict(x_test)

#unknown prediction
x_testset=test_dataset.iloc[:,[1,2]].values
#x_testset=sc_x.transform(x_testset)
y_pred=regressor.predict(x_testset)


#acuuracy
accuracy=regressor.score(x_test,y_test)

##kfold Cross validation
from sklearn.model_selection import cross_val_score
kfold=cross_val_score(estimator=regressor,X=x_train,y=y_train,cv=10)    
kfold.mean()
kfold.std()




#Evaluating the model
import statsmodels.formula.api as sm
x_sample=np.append(arr=np.ones((411,1)).astype(int),values=x_sample,axis=1)
x_opt=x_sample[:,[0,1,2]]
regressor_ols=sm.OLS(endog=y_sample,exog=x_opt).fit()
regressor_ols.summary()

x_opt=x_sample[:,[0,2]]
regressor_ols=sm.OLS(endog=y_sample,exog=x_opt).fit()
regressor_ols.summary()

#Gridsearch for svm poly
#from sklearn.grid_search import GridSearchCV
#parameters=[{'C':[1,2,3,4,5,6,7],
#           'kernel' :['poly'],   #CAnt use grid search because continoue value
#           'degree' : [1,2,3,4,5,6],
#           'gamma':[0.25],
#           'coef0':[0.1,0.2,0.3,0.4,0.5]
#           }]
#grid_search=GridSearchCV(estimator=regressor,param_grid=parameters,scoring='accuracy',
#                         n_jobs=-1,cv=10)
#grid_search=grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#grid_accuracy=grid_search.best_params_









