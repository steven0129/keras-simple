import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy
import pandas
from urllib.request import urlopen
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# download dataset
response = urlopen(
    'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
data = response.read()
fileName = './data/iris.csv'

if(os.path.isfile(fileName)):
    os.remove(fileName)

file = open(fileName, 'wb')
file.write(data)
file.close()

# load dataset
dataframe = pandas.read_csv('./data/iris.csv', header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)
dummyY = np_utils.to_categorical(encodedY)  # one-hot encoding

# define baseline model
def baseline_model():
	# create model
    # 4 inputs --> [4 hidden nodes] --> 3 outputs
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=100000, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(estimator, X, dummyY, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))