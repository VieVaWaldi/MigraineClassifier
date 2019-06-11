# https://www.kaggle.com/hammadzahidali/classification-tutorial-machine-learning

import pandas as pd 				# Data analysis tool
import matplotlib.pyplot as plt 	# Plotter
import seaborn as sns				# Also a plotter
from sklearn import preprocessing	# Simple data learning tool, i d rather use pytorch
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

print('')

# 1. Understanding the data #####################################################

data = pd.read_csv('Iris.csv')

# print(data.head(5))						
# print(data.tail(5))

# Species is the column that should be learned
# print('All target classes in column "Species": {}'.format(data['Species'].unique()))		# Prints every class of row species

# Awesome overview
# print(data.describe())

# Overview aboubt every col
# for col in data:
# 	print()
# 	print( data[col].describe())

# Awesome plotter. Here we can try to find patterns by ourself
sns.FacetGrid(data, hue='Species', height=6).map(plt.scatter, 'SepalLengthCm', 'SepalWidthCm').add_legend()
# plt.show()

# 2. Data Preprocessing ##########################################################

# Feature and target seperation

all_cols = list(data.columns[:])
features = list(data.columns[1:5])
target = data.columns[5]			
# print('All colums: {}'.format(all_cols))
# print('All feature columns: {}'.format(features))
# print('Target column: {}'.format(target))

x = data.iloc[:,1:5]
y = data.iloc[:,5]
# print(x.shape)
# print(y.shape)

# Turning the classes of species into numbers
le = preprocessing.LabelEncoder()		
le.fit(y)
y = le.transform(y)

# Turn this into a col in dataset
# data['EncodedSpecies'] = y

# 3. Learning the data via Logsitic Regresssion ########################################

# Create default learning model
logreg = LogisticRegression()			

# fit data into model
logreg.fit(x, y)					# Thats where it learns

y_pred = logreg.predict(x)			# Trying to predict the given data
print('Prediction acc for all data: {}'.format(metrics.accuracy_score(y, y_pred)))

# 4. Model evaluation #################################################################

x_test = [[5.1, 3.5, 1.4, 0.2]]		# First row of data set: Class = Iris-setosa[0]
y_test = logreg.predict(x_test)
print(y_test)

x_test = [[6.0, 3.2, 4.7, 1.4]]		# Second row of data set: Class = Iris-versicolor[1]
y_test = logreg.predict(x_test)
print(y_test)

x_test = [[6.4, 2.7, 5.3, 1.9 ]]	# Third row of data set: Class = Iris-virginica[2]
y_test = logreg.predict(x_test)
print(y_test)

# Back to the tutorial
# Sample test data by splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=4)

logres = LogisticRegression()
logres.fit(x_train,y_train) # train data

# predict from test
log_pred = logres.predict(x_test)

# check accuracy
print(mt.accuracy_score(log_pred,y_test))
