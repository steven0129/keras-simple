# 訓練一個網絡來區分從金屬圓柱體反彈的聲納信號和從大致圓柱形的岩石彈起的聲納信號

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import pandas
from urllib.request import urlopen
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# download dataset
response = urlopen(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data')
data = response.read()
fileName = './data/sonar.csv'

if(os.path.isfile(fileName)):
    os.remove(fileName)

file = open(fileName, 'wb')
file.write(data)
file.close()

# load dataset
mydata = pandas.read_csv('./data/sonar.csv', header=None)
dataset = mydata.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# encode class value as integers
encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)

# Define function to build keras model
def create_baseline():
    model = Sequential()
    model.add(Dense(60, input_dim=60, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# create the model using the KerasClassifier function
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))