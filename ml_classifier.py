#  Was build with the help of: https://medium.com/intuitive-deep-learning/build-your-first-neural-network-to-predict-house-prices-with-keras-eb5db60232c

#### README #########################################################################################################

	# ToDo:
	# normalize data in a way that new data is scalable as well for the actual usage

#####################################################################################################################



import numpy as np														# super important for arrays
import pandas as pd 													# pandas is a data analysis tool, uses numpy!
from sklearn import preprocessing										# normalize data
from sklearn.model_selection import train_test_split					# split data in test training

import keras 															# Machine Learning Framework
from keras.models import Sequential										
from keras.layers import Dense
from keras.models import load_model

import matplotlib.pyplot as plt											# Plotter



#### Hyperparameters ################################################################################################
NAME = 'test_model_X=8,-Y=gender-withNormalization'
INPUT_SHAPE = 8
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 64

OPTIMIZER = 'adam'														# probabaly right
LOSS = 'binary_crossentropy'											# important, more info needed
ACTIVATION_LAST_LAYER = 'sigmoid'										# important, sigmoid doesnt work when y has more than 2 possible outcomes 

BATCH_SIZE = 32															# number of samples to process before updating parameters, hier mini-batch gradient descent
EPOCHS = 600															# number of runs through the entire dataset


# Data:
# X = patient 2, age 4, intensity 7, painlocation 8, paintype 9, painorigin 10, dayslost 12, shiftwork 13
# Y = gender


def main():
#### Step 1: Data Preprocessing #####################################################################################

																		# df is a pandas dataframe, dtype for mixed values					
	df = pd.read_csv('data/M_anfaelle_mit_patientendaten_kurz.csv', dtype={'Medname6': str, 'Medname7': str})
	dataset = df.values													# remove first row and col: 2D Array => is numpy.ndarray

	label_encoder = preprocessing.LabelEncoder()						# Label Encoder: turn things into numbers

	patient = dataset[:,2]
	gender = label_encoding(label_encoder, dataset[:,3])				# turn strings into number => is numpy.ndarray
	age = dataset[:,4]
	intensity = dataset[:,7]
	pain_location = label_encoding(label_encoder, dataset[:,8])			
	pain_type = label_encoding(label_encoder, dataset[:,9])
	pain_origin = label_encoding(label_encoder, dataset[:,10])
	dayslost = dataset[:,12]
	shiftwork = label_encoding(label_encoder, dataset[:,13])

	X = np.column_stack((patient, age))									# extract feature cols, inside () because not iterable otherwise
	X = np.column_stack((X, intensity))							
	X = np.column_stack((X, pain_location))							
	X = np.column_stack((X, pain_type))							
	X = np.column_stack((X, pain_origin))							
	X = np.column_stack((X, dayslost))							
	X = np.column_stack((X, shiftwork))							

	Y = gender															# extract predicton col

	min_max_scaler = preprocessing.MinMaxScaler()						# Data Normalizer
	X_scale = min_max_scaler.fit_transform(X)							
																		# Split data into training, test, validation
	X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)	
	X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

	print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape) # nice to debug

#### Step 2: Building the Neural Network ############################################################################

	# Add Dropout if overfitting occurs
	model = Sequential([
	    Dense(HIDDEN_LAYER_1, activation='relu', input_shape=(INPUT_SHAPE,)),
	    Dense(HIDDEN_LAYER_2, activation='relu'),
	    Dense(1, activation=ACTIVATION_LAST_LAYER),
	])

	model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

#### Step 3: Training & Testing #####################################################################################

	hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val))
	
	model.save('models/ml_classifier/{}.h5'.format(NAME))

	model.evaluate(X_test, Y_test)[1]									# [0] = loss, [1] = accuracy

#### Step 4: Making Predictions #####################################################################################

	pre_1 = X_scale[0,0:9].reshape(-1,8)
	pre_2 = X_scale[1,0:9].reshape(-1,8)
	pre_3 = X_scale[2,0:9].reshape(-1,8)
	pre_4 = X_scale[3,0:9].reshape(-1,8)

	print('[ === Final Prediction === ]')
	print('Should be: 0, 0, 1, 0')
	print(model.predict(pre_1))
	print(model.predict(pre_2))
	print(model.predict(pre_3))
	print(model.predict(pre_4))

	plot_acc(hist).show

def use_model():

	df = pd.read_csv('data/M_anfaelle_mit_patientendaten_kurz.csv', dtype={'Medname6': str, 'Medname7': str})
	dataset = df.values													# remove first row and col: 2D Array => is numpy.ndarray

	label_encoder = preprocessing.LabelEncoder()						# Label Encoder: turn things into numbers

	patient = dataset[:,2]
	gender = label_encoding(label_encoder, dataset[:,3])				# turn strings into number => is numpy.ndarray
	age = dataset[:,4]
	intensity = dataset[:,7]
	pain_location = label_encoding(label_encoder, dataset[:,8])			
	pain_type = label_encoding(label_encoder, dataset[:,9])
	pain_origin = label_encoding(label_encoder, dataset[:,10])
	dayslost = dataset[:,12]
	shiftwork = label_encoding(label_encoder, dataset[:,13])

	X = np.column_stack((patient, age))									# extract feature cols, inside () because not iterable otherwise
	X = np.column_stack((X, intensity))							
	X = np.column_stack((X, pain_location))							
	X = np.column_stack((X, pain_type))							
	X = np.column_stack((X, pain_origin))							
	X = np.column_stack((X, dayslost))							
	X = np.column_stack((X, shiftwork))		

	model = keras.models.load_model('models/ml_classifier/{}.h5'.format(NAME))

	min_max_scaler = preprocessing.MinMaxScaler()						# Data Normalizer
	X_scale = min_max_scaler.fit_transform(X)							

	pre_1 = X_scale[0,0:9].reshape(-1,8)
	pre_2 = X_scale[1,0:9].reshape(-1,8)
	pre_3 = X_scale[2,0:9].reshape(-1,8)
	pre_4 = X_scale[3,0:9].reshape(-1,8)

	print('[ === Final Prediction  Loaded === ]')
	print('Should be: 0, 0, 1, 0')
	print(model.predict(pre_1))
	print(model.predict(pre_2))
	print(model.predict(pre_3))
	print(model.predict(pre_4))

#### HELPERS ########################################################################################################

def plot_loss(hist):
	plt.plot(hist.history['loss'])									
	plt.plot(hist.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper right')
	return plt

def plot_acc(hist):
	plt.plot(hist.history['acc'])										
	plt.plot(hist.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='lower right')
	return plt

def label_encoding(label_encoder, data):
	label_encoder.fit(data)
	return label_encoder.transform(data)

#####################################################################################################################

if __name__ == '__main__':
    # main() 
    use_model()
