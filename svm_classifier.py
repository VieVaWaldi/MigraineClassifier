#### SVM 
#### README #########################################################################################################

	# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

	# ToDo:
	# normalize data in a way that new data is scalable as well for the actual usage

#####################################################################################################################



import numpy as np														# super important for arrays
import pandas as pd 													# pandas is a data analysis tool, uses numpy!
from sklearn import preprocessing										# normalize data
from sklearn.model_selection import train_test_split					# split data in test training

from sklearn.svm import SVC  

import matplotlib.pyplot as plt											# Plotter
from mlxtend.plotting import plot_decision_regions



#### Hyperparameters ################################################################################################
NAME = ''
KERNEL = 'poly'															# linear, rfb, poly
GAMMA = 0.01
PROBABILITY = False

# Data:
# X = patient 2, age 4, intensity 7, painlocation 8, paintype 9, painorigin 10, dayslost 12, shiftwork 13
# Y = gender

# Data:
# X = patient 2, age 4
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
	# intensity = dataset[:,7]
	# pain_location = label_encoding(label_encoder, dataset[:,8])			
	# pain_type = label_encoding(label_encoder, dataset[:,9])
	# pain_origin = label_encoding(label_encoder, dataset[:,10])
	# dayslost = dataset[:,12]
	# shiftwork = label_encoding(label_encoder, dataset[:,13])

	X = np.column_stack((patient, age))									# extract feature cols, inside () because not iterable otherwise
	# X = np.column_stack((X, intensity))							
	# X = np.column_stack((X, pain_location))							
	# X = np.column_stack((X, pain_type))							
	# X = np.column_stack((X, pain_origin))							
	# X = np.column_stack((X, dayslost))							
	# X = np.column_stack((X, shiftwork))							

	Y = gender															# extract predicton col

	min_max_scaler = preprocessing.MinMaxScaler()						# Data Normalizer
	X_scale = min_max_scaler.fit_transform(X)							
																		# Split data into training, test, validation
	X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)	
	X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

	# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape) # nice to debug

#### Step 2: Building the SVM  ######################################################################################

	svclassifier = SVC(kernel=KERNEL, gamma=GAMMA, probability=PROBABILITY, verbose=True)  

	# obj 				is the optimal objective value of the dual SVM problem
	# rho 				is the bias term in the decision function sgn(w^Tx - rho)
	# nSV and nBSV 		are number of support vectors and bounded support vectors (i.e., alpha_i = C)
	# nu-svm 			is a somewhat equivalent form of C-SVM where C is replaced by nu
	# nu 				simply shows the corresponding parameter. More details are in libsvm document					

#### Step 3: Training & Testing #####################################################################################

	training = svclassifier.fit(X_train, Y_train)  
	print('[=== Trained Model ===]')
	print(training)

	plot_decision_regions(X=X_train, y=Y_train, clf=svclassifier, legend=2)

	# Update plot object with X/Y axis labels and Figure Title
	plt.xlabel(X.columns[0], size=14)
	plt.ylabel(X.columns[1], size=14)
	plt.title('SVM Decision Region Boundary', size=16)
	plt.show()

#### Step 4: Making Predictions #####################################################################################

	pre_1 = X_scale[0,0:9].reshape(-1,8)
	pre_2 = X_scale[1,0:9].reshape(-1,8)
	pre_3 = X_scale[2,0:9].reshape(-1,8)
	pre_4 = X_scale[3,0:9].reshape(-1,8)

	print('[ === Final Prediction === ]')
	print('Should be: 0, 0, 1, 0')
	print(svclassifier.predict(float(pre_1)))
	print(svclassifier.predict(float(pre_2)))
	print(svclassifier.predict(float(pre_3)))
	print(svclassifier.predict(float(pre_4)))

def use_model():
	pass

#### HELPERS ########################################################################################################

def label_encoding(label_encoder, data):
	label_encoder.fit(data)
	return label_encoder.transform(data)

#####################################################################################################################

if __name__ == '__main__':
    main() 
    # use_model()
