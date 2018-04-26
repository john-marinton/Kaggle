# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

style.use('ggplot')


#loading the dataset
dataset=pd.read_csv('train.csv')
x=dataset.iloc[:,[2,4,5]].values
y=dataset.iloc[:,1].values



#Desribing
dataset.describe()


#Dealing with missing values in age
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(dataset.iloc[:,5:6])
dataset.iloc[:,5:6]=imputer.transform(dataset.iloc[:,5:6])

#Visualizing the dataset with sns pairplot
sns.pairplot(dataset)

#visualizing the dataset on basis of survived with kind in regression type
sns.pairplot(dataset,kind='reg')

#Visualizing the dataset with sns pairplot wiht diag)kind in kde
sns.pairplot(dataset,diag_kind='kde')

#visualizing the dataset on basis of survived 
sns.pairplot(dataset,hue='Survived')


#visualizing the dataset on basis of how many survive by thier age and class
sns.pairplot(dataset,vars=['Pclass','Age'],hue='Survived')

#visulaizng the dataset on basis of family memebers and survival
sns.pairplot(dataset,vars=['SibSp','Parch'],hue='Survived')


#Visulazing the dataset to know how well the Pclass are distributed
sns.distplot(dataset['Pclass'],rug=True)
plt.show()

#Visulazing the dataset to know how well the Age are distributed using distplot
sns.distplot(dataset['Age'],norm_hist=False)
plt.show()

#Visulazing the dataset to know how well the age are 
#distributed on the basis of Pclass using swarmplots
#Swarm plot shows good visualizing rahter than dist plot 
sns.swarmplot(x='Pclass',y='Age',data=dataset)
plt.show()

#visualizing how many men and women survived on basis of age
sns.swarmplot(x='Sex',y='Age',hue='Survived',data=dataset)

#visualizing how many men and women survived on basis of age
sns.violinplot(x='Sex',y='Age',hue='Survived',data=dataset)



#visualing the dataset
dataset['Age'].value_counts().plot(kind='bar')


#Visualzing how many survived and not survived
dataset['Survived'].value_counts().plot(kind='bar')
plt.title('Survival Rate 0=Not Survived 1=Survived')

#visulaizing the dataset on basis of pclass,sex,survived 
sns.factorplot('Pclass','Survived','Sex',data=dataset,kind='bar',palette='muted')
plt.show()

#Visualizing dataset whether if they had family and survived or not
df_family=dataset['SibSp']+dataset['Parch']
sns.factorplot(x=df_family,y='Survived',kind='bar',data=dataset)
plt.show()


#visualizing how many survived on basis of their age
dataset.iloc[:,5:6].astype(int)
dataset.loc[dataset['Age']<=16,'Age']=0
dataset.loc[(dataset['Age']>16) &(dataset['Age']<=32) ,'Age']=1
dataset.loc[(dataset['Age']>32) &(dataset['Age']<=48) ,'Age']=2
dataset.loc[(dataset['Age']>48) &(dataset['Age']<=64) ,'Age']=3
dataset.loc[dataset['Age']>64,'Age']=4

#dataset['Age'].hist(bins=70)

sns.factorplot('Sex','Survived',data=dataset,kind='bar',palette='muted',hue='Age')
plt.show()





#labelEncoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x[:,1]=labelencoder.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()



#Avoiding Dummy Variable Trap
#x=x[:,1:]

#splitting the dataset
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.23,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Building the model

#logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)      #Accuracy=79
classifier.fit(x_train,y_train)

#Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#classifier=GaussianNB()                             #Accuracy=78
#classifier.fit(x_train,y_train)


##Knn Neighbors classifier
#from sklearn.neighbors import KNeighborsClassifier
#classifier=KNeighborsClassifier(algorithm='brute',n_neighbors=2,metric='minkowski',p=2,weights='distance')   #Accuracy=81
#classifier.fit(x_train,y_train)

#Random Forest Classifier
#from sklearn.ensemble import RandomForestClassifier
#classifier=RandomForestClassifier(n_estimators=1000,random_state=0)   #Accuracy=78
#classifier.fit(x_train,y_train)

#SVM
#from sklearn.svm import SVC
#classifier=SVC(kernel='rbf',random_state=0)            #Accuracy=78
#classifier.fit(x_train,y_train)


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
from sklearn.grid_search import GridSearchCV
parameters=[{'penalty':['l1'],
                            'solver':['liblinear'],
                            'intercept_scaling':[1,2,3,4,5,6]},
            {'penalty':['l1'],
                            'solver':['saga'],
                            'intercept_scaling':[1,2,3,4,5,6]},
            {'penalty':['l2'],
                   'solver':['newton-cg','lbfgs','sag'],
                          'max_iter':[10,20,30,40]}]
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',
                         n_jobs=-1,cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_accuracy=grid_search.best_score_
best_param=grid_search.best_params_



#visualizing the trainig set classifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set,y_set=x_train,y_train
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
                  )
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),
                label=j)
plt.title("Titanic training set")
plt.xlabel('Passengerclass')
plt.ylabel('Age')
plt.show()

#visualizing the testing set classifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),
                  np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01)
                  )
plt.contourf(x1,x2,classifier.predict(np.array(([x1.ravel(),x2.ravel()])).T).reshape(x1.shape),
             alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.xlim(x2.min(),x2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),
                label=j)
plt.title("Titanic testing set")
plt.xlabel('Passengerclass')
plt.ylabel('Age')
plt.show()
