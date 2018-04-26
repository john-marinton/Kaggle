# -*- coding: utf-8 -*-

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
labelencoder1=LabelEncoder()
x[:,1]=labelencoder1.fit_transform(x[:,1])
labelencoder3=LabelEncoder()
x[:,3]=labelencoder3.fit_transform(x[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()


#Avoiding Dummy Variable Trap
x=x[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.23,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Building ANN

#importing libraries
from keras.models import Sequential
from keras.layers import Dense

#Intializing the ANN
classifier=Sequential()

#Adding Input and Hidden Layer
classifier.add(Dense(units=4,activation='relu',input_dim=6,
                     kernel_initializer='uniform'))

#Adding Hidden Layer
classifier.add(Dense(units=2,activation='relu',
                     kernel_initializer='uniform'))


#output layer
classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

#Compiling
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ann with training set
classifier.fit(x_train,y_train,epochs=1000,batch_size=10)


#prediction
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#Evaluating the accuracy of the model
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Evalutaing model Accuracy
TP=cm[1][1]
TN=cm[0][0]
FP=cm[0][1]
FN=cm[1][0]

Accuracy=(TP+TN)/(TP+TN+FP+FN)
precision=TP/(TP+FP)
recall=TP/(TP+FN)
f1_score=2*precision*recall/(precision+recall)


#Evaluating model performance using k-fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():

    from keras.models import Sequential
    from keras.layers import Dense

    #Intializing the ANN
    classifier=Sequential()

    #Adding Input and Hidden Layer
    classifier.add(Dense(units=4,activation='relu',input_dim=6,
                         kernel_initializer='uniform'))

    #output layer
    classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

    #Compiling
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
       
    
    
    
    
    
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=200)
kfoldaccuracy= cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,
                               scoring='accuracy')
kfoldaccuracy.mean()
kfoldaccuracy.std()


#getting the test data
x_test_data=test_dataset.iloc[:,[1,3,4,10,11]].values

##labelEncoding for Embarked,gender
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
test_labelencoder1=LabelEncoder()
x_test_data[:,1]=test_labelencoder1.fit_transform(x_test_data[:,1])
test_labelencoder3=LabelEncoder()
x_test_data[:,3]=test_labelencoder3.fit_transform(x_test_data[:,3])
test_onehotencoder=OneHotEncoder(categorical_features=[3])
x_test_data=test_onehotencoder.fit_transform(x_test_data).toarray()

#Avoiding Dummy Variable Trap
x_test_data=x_test_data[:,1:]


#feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_test_data=sc_x.fit_transform(x_test_data)

#predicting the test data
test_predict=classifier.predict(x_test_data)
test_predict=(test_predict>0.5)


