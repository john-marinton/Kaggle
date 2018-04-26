# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

style.use('ggplot')


#loading the dataset
train_dataset=pd.read_csv('train.csv')
test_dataset=pd.read_csv('test.csv')

#Desribing
train_dataset.describe()
test_dataset.describe()

#handling Missing Values in Age test and train dataset
train_dataset['Age']=train_dataset['Age'].fillna(train_dataset['Age'].mean())
test_dataset['Age']=test_dataset['Age'].fillna(test_dataset['Age'].mean())

#Handling missing values in Embarked in train and test dataset
train_dataset['Embarked']=train_dataset['Embarked'].fillna('S')
test_dataset['Embarked']=test_dataset['Embarked'].fillna('S')


#visualzing the corrleation between varibales
corr=train_dataset.corr()
sns.heatmap(data=corr)


#visualize the train dataset using pairplot
sns.pairplot(train_dataset)

#visualize Pclass, sex,Age
sns.factorplot(x='Pclass',y='Age',hue='Sex',data=train_dataset,kind='bar')

#visulaize how many male and female survived on basis of class
sns.factorplot(x='Pclass',y='Survived',hue='Sex',data=train_dataset,kind='bar')

#Visualizing the corelation betwwen embarked and survived
sns.factorplot(x='Embarked',y='Survived',data=train_dataset,kind='bar')

#visualizing whether if they had family they survived or not
train_dataset['family']=train_dataset['SibSp']+train_dataset['Parch']
test_dataset['family']=test_dataset['SibSp']+test_dataset['Parch']

#spliting training dataset into test and train
x=train_dataset.iloc[:,[2,4,5,11,12]].values
y=train_dataset.iloc[:,1].values

##labelEncoding for Embarked,gender
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])
x[:,1]=labelencoder.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1,3])
x=onehotencoder.fit_transform(x).toarray()


#Avoiding Dummy Variable Trap
#x=x[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.23,random_state=0)

#feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_train=sc_x.fit_transform(x_train)
#x_test=sc_x.transform(x_test)

#Building the model

#logistic Regression
#from sklearn.linear_model import LogisticRegression
#classifier=LogisticRegression(random_state=0)      #Accuracy=79
#classifier.fit(x_train,y_train)

#Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()                             #Accuracy=78
#classifier.fit(x_train,y_train)


##Knn Neighbors classifier
#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(algorithm='kd_tree',n_neighbors=6,p=1,metric='minkowski',weights='uniform')   #Accuracy=81
#classifier.fit(x_train,y_train)

#Random Forest Classifier
#from sklearn.ensemble import RandomForestClassifier
#classifier=RandomForestClassifier(n_estimators=10,bootstrap=False,criterion='gini',
#                                  random_state=0,max_depth=3,min_samples_leaf=1,
#                                  min_samples_split=2)   #Accuracy=0.8341
#classifier.fit(x_train,y_train)

#SVM
#from sklearn.svm import SVC
#classifier=SVC(kernel='rbf',random_state=0)            #Accuracy=0.8243
#classifier.fit(x_train,y_train)


#Xgboost Model
from xgboost import XGBClassifier
classifier=XGBClassifier(n_estimaors=200)                          #acuuracy=84
classifier.fit(x_train,y_train)



#prediction
y_pred=classifier.predict(x_test)




#Accuracy
accuracy=classifier.score(x_test,y_test)

#Evaluating the accuracy of the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Evaluating model performance using k-fold cross validation
from sklearn.model_selection import cross_val_score
kfoldaccuracy= cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
kfoldaccuracy.mean()
kfoldaccuracy.std()

##Applyiing the grid search for knn to find best hyper parameters
#from sklearn.grid_search import GridSearchCV
#parameters=[{'n_neighbors':[1,2,3,4,5,6,7,8,9],'weights':['uniform','distance'],
#                            'algorithm':['ball_tree','kd_tree','brute'],
#                            'p':[1,2]}]
#grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',
#                         n_jobs=-1,cv=10)
#grid_search=grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#best_param=grid_search.best_params_

##Applyiing the grid search for logistic to find best hyper parameters
#from sklearn.grid_search import GridSearchCV
#parameters=[{'penalty':['l1'],
#                            'solver':['liblinear'],
#                            'intercept_scaling':[1,2,3,4,5,6]},
#            {'penalty':['l1'],
#                            'solver':['saga'],
#                            'intercept_scaling':[1,2,3,4,5,6]},
#            {'penalty':['l2'],
#                   'solver':['newton-cg','lbfgs','sag'],
#                          'max_iter':[10,20,30,40]}]
#grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',
#                         n_jobs=-1,cv=10)
#grid_search=grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#best_param=grid_search.best_params_

#Applying grid search for random forest
#from sklearn.grid_search import GridSearchCV
#parameters=[{"max_depth": [3, None],
#              "min_samples_split": [2, 3, 10],
#              "min_samples_leaf": [1, 3, 10],
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}]
#grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,n_jobs=-1,scoring='accuracy')
#grid_search=grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#best_param=grid_search.best_params_

#Applying grid search for Svm
#from sklearn.grid_search import GridSearchCV
#parameters=[{'C':[10,100,1.0,2.0], 'kernel':['rbf'],
#             'gamma': [0.16,0.14,0.15],'decision_function_shape':['ovo','ovr'],
#             },
#               {'C':[10,100,1.0,2.0], 'kernel':['sigmoid'],
#                'coef0':[0.0,0.1,0.2],'gamma': [0.16,0.14,0.15],
#                'decision_function_shape':['ovo','ovr']}]
#grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,
#                         n_jobs=-1,cv=10)
#grid_search=grid_search.fit(x_train,y_train)
#best_accuracy=grid_search.best_score_
#best_parameters=grid_search.best_params_




#getting the test data
x_test_data=test_dataset.iloc[:,[1,3,4,10,11]].values

##labelEncoding for Embarked,gender
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x_test_data[:,3]=labelencoder.fit_transform(x_test_data[:,3])
x_test_data[:,1]=labelencoder.fit_transform(x_test_data[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1,3])
x_test_data=onehotencoder.fit_transform(x_test_data).toarray()

#Avoiding Dummy Variable Trap
#x_test_data=x_test_data[:,1:]


#feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_test_data=sc_x.fit_transform(x_test_data)

#predicting the test data
test_predict=classifier.predict(x_test_data)

