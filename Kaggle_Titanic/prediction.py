# -*- coding: utf-8 -*-

import ml_titanic_training_model
import pandas as pd
import numpy as np

#loading the test datset
test_dataset=pd.read_csv('test.csv')
x_test=test_dataset.iloc[:,[1,3]].values

#labelEncoding for test set
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
x_test[:,1]=labelencoder.fit_transform(x_test[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x_test=onehotencoder.fit_transform(x_test).toarray()


#prediction
y_pred_test=classifier.predict(x_test)

