#  Was build with the help of: https://medium.com/intuitive-deep-learning/build-your-first-neural-network-to-predict-house-prices-with-keras-eb5db60232c

import pandas as pd 												# pandas is a data analysis tool
from sklearn import preprocessing									# normalize data
from sklearn.model_selection import train_test_split				# split data in test training

from keras.models import Sequential									# Machine Learning Framework
from keras.layers import Dense

import matplotlib.pyplot as plt										# Plotter



#### Hyperparameters ################################################################################################
HIDDEN_LAYER_1 = 64
HIDDEN_LAYER_2 = 64
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
BATCH_SIZE = 32
EPOCHS = 100


#### Step 1: Data Preprocessing #####################################################################################

# df = pd.read_csv('data/remove_me.csv')								# df is dataframe
df = pd.read_csv('data/M_anfaelle_mit_patientendaten_kurz.csv')								
dataset = df.values													# ohne zeilennummer und spalten header: 2D Array

X = dataset[:,0:10]													# Feature Spalten extrahieren
Y = dataset[:,10]													# Prediction Spalte extrahieren

min_max_scaler = preprocessing.MinMaxScaler()						# Data Normalizer
X_scale = min_max_scaler.fit_transform(X)							# normalize Input data

																	# Split data into training, test, validation
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)	
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

# print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape) # nice to debug

#### Step 2: Building the Neural Network ############################################################################

# Add Dropout if overfitting occurs
model = Sequential([
    Dense(HIDDEN_LAYER_1, activation='relu', input_shape=(10,)),
    Dense(HIDDEN_LAYER_2, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=['accuracy'])

#### Step 3: Training & Testing #####################################################################################

hist = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)[1]									# [0] = loss, [1] = accuracy

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
