import numpy
import pandas

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# For reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset (rivedere!)
# load dataset using pandas; split the column into 60 input variables and 1 output variable (this is due to the sonar.csv -> adapt to our dataset!)
# pandas -> used because it easily handles strings, while numpy does not
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# use scikit-learn to evaluate the model using stratified k-fold cross validation
# what is stratified k-fold cross validation? it is a resampling technique that will provide an estimate of the 
"""
performance of the model. it does this by splitting the data inot k parts, training the model on all parts except one which is held out as a test set to evaluate the performance of the model.



"""


def create_baseline():
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


estimator = KerasClassifier(build_fn=create_baseline,
	nb_epoch=100,
	batch_size=5,
	verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

# evaluate baseline model with standardized dataset
numpy.random,seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)