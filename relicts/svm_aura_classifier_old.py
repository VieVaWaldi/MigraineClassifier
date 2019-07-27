#### SVM 
#### README #########################################################################################################

	# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

	# ToDo:
	# normalize data in a way that new data is scalable as well for the actual usage

#####################################################################################################################



from datetime import datetime
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

	gender = label_encoding(label_encoder, dataset[:,3])
	age = dataset[:,4]		

	### Stupid Time #####################			

	begin_time = dataset[:,5]			#		
	end_time = dataset[:,6]				#		
	duration = []
	time_start = []

	i = 0
	for date in begin_time:
		begin_time[i] = datetime.strptime(date[:16], '%Y-%m-%d %H:%M')
		time_start.append(datetime.strptime(date[11:16], '%H:%M'))
		i+=1

	i = 0
	for date in end_time:
		end_time[i] = datetime.strptime(date[:16], '%Y-%m-%d %H:%M')
		i+=1

	for i in range(len(begin_time)):
		dur = end_time[i] - begin_time[i]
		duration.append(int(dur.total_seconds() / 60))

		dur2 = time_start[i] - datetime(1900,1,1,0,0,0)
		time_start[i] = int(dur2.total_seconds() / 60)

	######################

	duration = np.array(duration)
	time_start = np.array(time_start)
	intensity = dataset[:,7]					
	pain_location = label_encoding(label_encoder, dataset[:,8])
	pain_type = label_encoding(label_encoder, dataset[:,9])
	pain_origin = label_encoding(label_encoder, dataset[:,10])

	###################### SYMPTOMAMIMOMAMAMAMA

	symptoms = dataset[:,11]
	uebelkeit = [0] * len(dataset[:,11])
	erbrechen = [0] * len(dataset[:,11])
	lichtempfindlichkeit = [0] * len(dataset[:,11])
	laermempfindlichkeit = [0] * len(dataset[:,11])
	geruchsempfindlichkeit = [0] * len(dataset[:,11])
	ruhebeduerfnis = [0] * len(dataset[:,11])
	bewegungsdrang = [0] * len(dataset[:,11])
	schwindel = [0] * len(dataset[:,11])
	sonstiges = [0] * len(dataset[:,11])

	#10010100 -> 00101001 -> 101001 -> 101001 000
	idx = -1
	for ele in symptoms:
		idx += 1

		ele = str(int(ele))
		ele = ele[::-1]

		if ele == '0':
			continue
		
		ele = ele[2:len(ele)]
		
		col = -1
		for char in ele:
			col += 1
			if char == '1':
				if col == 0:
					uebelkeit[idx] = 1
				if col == 1:
					erbrechen[idx] = 1
				if col == 2:
					lichtempfindlichkeit[idx] = 1
				if col == 3:
					laermempfindlichkeit[idx] = 1
				if col == 4:
					geruchsempfindlichkeit[idx] = 1
				if col == 5:
					ruhebeduerfnis[idx] = 1
				if col == 6:
					bewegungsdrang[idx] = 1
				if col == 7:
					schwindel[idx] = 1
				if col == 8:
					sonstiges[idx] = 1

	######################

	week_day = label_encoding(label_encoder, dataset[:,17])
	med_1 = dataset[:,23]
	med_effect = dataset[:,32]

	i = 0
	for ele in med_1:
		if type(ele) != str:			
			med_1[i] = 'keins'
			# print(ele)
		i+= 1

	i = 0
	for ele in med_effect:
		if type(ele) != str:			
			med_effect[i] = 'keins'
			# print(ele)
		i+= 1

	med_1 = label_encoding(label_encoder, med_1)
	# med_2 = label_encoding(label_encoder, dataset[:,24])
	med_effect = label_encoding(label_encoder, med_effect)

	k_type = label_encoding(label_encoder, dataset[:,34])

	X = np.column_stack((gender, age))							
	X = np.column_stack((X, duration))							
	X = np.column_stack((X, time_start))							
	X = np.column_stack((X, intensity))							
	X = np.column_stack((X, pain_location))							
	X = np.column_stack((X, pain_type))							
	X = np.column_stack((X, pain_origin))		

	X = np.column_stack((X, uebelkeit))		
	X = np.column_stack((X, erbrechen))		
	X = np.column_stack((X, lichtempfindlichkeit))		
	X = np.column_stack((X, laermempfindlichkeit))		
	X = np.column_stack((X, geruchsempfindlichkeit))		
	X = np.column_stack((X, ruhebeduerfnis))		
	X = np.column_stack((X, bewegungsdrang))		
	X = np.column_stack((X, schwindel))		
	X = np.column_stack((X, sonstiges))		

	X = np.column_stack((X, week_day))							
	X = np.column_stack((X, med_1))							
	# X = np.column_stack((X, med_2))							
	X = np.column_stack((X, med_effect))							
	# X = np.column_stack((X, k_type))							
	
	Y = k_type															# extract predicton col

	# np.savetxt('processed_data.csv', X, fmt='%d', delimiter=',')
	# numpy.savetxt('processed_data.csv', X, fmt='%d', delimiter=',')

	############## VERTEILUNG MIT OHNE AURA ###################################### 	
	# mit = 0
	# ohne = 0

	# for data in k_type:
	# 	print(data)
	# 	if data == 0:
	# 		mit += 1
	# 	else:
	# 		ohne += 1

	# print('mit', mit)
	# print('ohne', ohne)

	# min_max_scaler = preprocessing.MinMaxScaler()						# Data Normalizer
	# X_scale = min_max_scaler.fit_transform(X)							
	X_scale = X
																		# Split data into training, test, validation
	X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)	
	# X_val_and_test, X_train, Y_val_and_test, Y_train = train_test_split(X_scale, Y, test_size=0.7)	
	X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

	print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape) # nice to debug

#### Step 2: Building the SVM  ######################################################################################

	svclassifier = SVC(kernel=KERNEL, gamma=GAMMA, probability=PROBABILITY, verbose=True)  

	# obj 				is the optimal objective value of the dual SVM problem
	# rho 				is the bias term in the decision function sgn(w^Tx - rho)
	# nSV and nBSV 		are number of support vectors and bounded support vectors (i.e., alpha_i = C)
	# nu-svm 			is a somewhat equivalent form of C-SVM where C is replaced by nu
	# nu 				simply shows the corresponding parameter. More details are in libsvm document					

#### Step 3: Training & Testing #####################################################################################

	training = svclassifier.fit(X_train, Y_train)  
	# print('[=== Trained Model ===]')
	# print(training)

#### Step 4: Making Predictions #####################################################################################

	pre_1 = X_scale[0,0:9].reshape(-1,2)
	pre_2 = X_scale[1,0:9].reshape(-1,2)
	pre_3 = X_scale[2,0:9].reshape(-1,2)
	pre_4 = X_scale[3,0:9].reshape(-1,2)

	print('[ === Final Prediction === ]')
	print('Should be: 0, 0, 1, 0')
	print(svclassifier.predict(np.ndarray([101,8]).reshape(-1, 1)))
	# print(svclassifier.predict(float(pre_2)))
	# print(svclassifier.predict(float(pre_3)))
	# print(svclassifier.predict(float(pre_4)))

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
