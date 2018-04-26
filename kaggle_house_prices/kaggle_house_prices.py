# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#loading the dataset
train_dataset=pd.read_csv('train.csv')
test_dataset=pd.read_csv('test.csv')

#info
train_dataset.info()
test_dataset.info()

#Describing
train_dataset.describe()
test_dataset.describe()

#dropping the most missing values
train_dataset=train_dataset.drop(['Id','Alley','FireplaceQu','PoolQC',
                                  'Fence','MiscFeature','Exterior1st','RoofMatl',
                                 'Condition2','GarageQual',
                                 'Heating','Exterior2nd','Electrical',
                                 'HouseStyle','Utilities'],axis=1)
test_dataset=test_dataset.drop(['Id','Alley','FireplaceQu','PoolQC',
                                  'Fence','MiscFeature','Exterior1st','RoofMatl',
                                 'Condition2','GarageQual',
                                 'Heating','Exterior2nd','Electrical',
                                 'HouseStyle','Utilities'],axis=1)




#handling Missing Values
train_dataset=train_dataset.fillna(train_dataset.mean())
test_dataset=test_dataset.fillna(test_dataset.mean())

#handling Missing Values for MasVnrType
train_dataset.isnull().sum()
train_dataset['MasVnrType'].value_counts()
train_dataset['MasVnrType'].fillna('BrkCmn',inplace=True)
test_dataset.isnull().sum()
test_dataset['MasVnrType'].value_counts()
test_dataset['MasVnrType'].fillna('BrkCmn',inplace=True)

#handling Missing Values for BsmtQual
train_dataset.isnull().sum()
train_dataset['BsmtQual'].value_counts()
train_dataset['BsmtQual'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['BsmtQual'].value_counts()
test_dataset['BsmtQual'].fillna('NA',inplace=True)

#handling Missing Values for BsmtCond
train_dataset.isnull().sum()
train_dataset['BsmtCond'].value_counts()
train_dataset['BsmtCond'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['BsmtCond'].value_counts()
test_dataset['BsmtCond'].fillna('NA',inplace=True)

#handling Missing Values for BsmtExposure
train_dataset.isnull().sum()
train_dataset['BsmtExposure'].value_counts()
train_dataset['BsmtExposure'].fillna('No',inplace=True)
test_dataset.isnull().sum()
test_dataset['BsmtExposure'].value_counts()
test_dataset['BsmtExposure'].fillna('No',inplace=True)

#handling Missing Values for BsmtFinType1
train_dataset.isnull().sum()
train_dataset['BsmtFinType1'].value_counts()
train_dataset['BsmtFinType1'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['BsmtFinType1'].value_counts()
test_dataset['BsmtFinType1'].fillna('NA',inplace=True)

#handling Missing Values for BsmtFinType2
train_dataset.isnull().sum()
train_dataset['BsmtFinType2'].value_counts()
train_dataset['BsmtFinType2'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['BsmtFinType2'].value_counts()
test_dataset['BsmtFinType2'].fillna('NA',inplace=True)



#handling Missing Values for GarageType
train_dataset.isnull().sum()
train_dataset['GarageType'].value_counts()
train_dataset['GarageType'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['GarageType'].value_counts()
test_dataset['GarageType'].fillna('NA',inplace=True)


#handling Missing Values for GarageFinish
train_dataset.isnull().sum()
train_dataset['GarageFinish'].value_counts()
train_dataset['GarageFinish'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['GarageFinish'].value_counts()
test_dataset['GarageFinish'].fillna('NA',inplace=True)


#handling Missing Values for GarageCond
train_dataset.isnull().sum()
train_dataset['GarageCond'].value_counts()
train_dataset['GarageCond'].fillna('NA',inplace=True)
test_dataset.isnull().sum()
test_dataset['GarageCond'].value_counts()
test_dataset['GarageCond'].fillna('NA',inplace=True)

#handling Missing Values for test MSZoning
test_dataset.isnull().sum()
test_dataset['MSZoning'].value_counts()
test_dataset['MSZoning'].fillna('RL',inplace=True)



#handling Missing Values for test KitchenQual
test_dataset.isnull().sum()
test_dataset['KitchenQual'].value_counts()
test_dataset['KitchenQual'].fillna('TA',inplace=True)


#handling Missing Values for test Functional
test_dataset.isnull().sum()
test_dataset['Functional'].value_counts()
test_dataset['Functional'].fillna('Typ',inplace=True)

#handling Missing Values for test SaleType
test_dataset.isnull().sum()
test_dataset['SaleType'].value_counts()
test_dataset['SaleType'].fillna('WD',inplace=True)

#Finding the difference
set_test=[]
for x in test_dataset:
    set_test.append(x)

    
set_train=[]
for x in train_dataset:
    set_train.append(x)    

set1=set(set_test)
set2=set(set_train)

print(set2.difference(set1))


#finding the corealtion betwwen the varibales
corr=train_dataset.corr()
sns.heatmap(data=corr)

##label encoding
train_dataset=pd.get_dummies(train_dataset)
test_dataset=pd.get_dummies(test_dataset)

#splitting in dependent and independent variable
x=train_dataset.drop('SalePrice',axis=1).values
y=train_dataset.loc[:,'SalePrice'].values


##splitting the test data
#x_test=test_dataset.iloc[:,:].values


##Splitting into training and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.21,random_state=0)

##feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

##feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x=sc_x.fit_transform(x)
#x_test=sc_x.transform(x_test)


#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=34)
x_train=pca.fit_transform(x_train)
x_test=pca.transform(x_test)
variance_ratio=pca.explained_variance_ratio_



##building the model
from xgboost import XGBRegressor
regressor=XGBRegressor()
regressor.fit(x_train,y_train)

#building the model
#from xgboost import XGBRegressor
#regressor=XGBRegressor()
#regressor.fit(x,y)

#Building Random forest
#from sklearn.ensemble import RandomForestRegressor
#regressor=RandomForestRegressor(n_estimators=100,random_state=0)
#regressor.fit(x,y)

#prediction
y_pred=regressor.predict(x_test)

#RMSE
from sklearn.metrics import mean_squared_error
rmse=mean_squared_error(y_test,y_pred)

#log RMS
from sklearn.metrics import mean_squared_log_error
lrmse=mean_squared_log_error(y_test,y_pred)



#Kfold validation
from sklearn.model_selection import cross_val_score
kfold=cross_val_score(estimator=regressor,X=x_train,y=y_train,cv=10)           
kfold.mean()
kfold.std()

##Kfold validation for orginal training set
#from sklearn.model_selection import cross_val_score
#kfold=cross_val_score(estimator=regressor,X=x,y=y,cv=10)           
#kfold.mean()
#kfold.std()


#Grid Search
from sklearn.grid_search import GridSearchCV
parameters=[{'learning_rate':np.arange(.1,1,.1),
        'reg_alpha':np.arange(0,10,1),
        'reg_lambda':np.arange(0,10,1),
        'objective':['reg:linear'],
        'max_depth':np.arange(1,10,1),
        'min_child_weight':np.arange(1,10,1),
        'gamma':np.arange(.1,1,.1)
        }]
grid=GridSearchCV(estimator=regressor,param_grid=parameters,scoring='accuracy',
                  n_jobs=-1,cv=10)
grid=grid.fit(x_train,y_train)
best_param=grid.best_params_
best_score=grid.best_score_



