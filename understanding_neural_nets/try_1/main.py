# https://medium.com/analytics-vidhya/build-your-first-neural-network-model-on-a-structured-dataset-using-keras-d9e7de5c6724

import pandas as pd
import numpy as np

# DATA PREPARATION ####################################################

# Load files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Combine data for prepocessing
train['Type'] = 'Train' 
test['Type'] = 'Test'
print('DEBUG')
fullData = pd.concat([train,test],axis=0) 	# preprocessing
print('DEBUG')

# Define cols
ID_col = ['User_ID','Product_ID']			# Id col
flag_col= ['Type']
target_col = ['Purchase']					# Prediction goal
# Categorie cols
cat_cols= ['Gender','Age','City_Category','Stay_In_Current_City_Years']
# Number cols
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col)-set(flag_col))

# combine number and categorie cols
num_cat_cols = num_cols+cat_cols

# Identify missing entries and assign _NA 
for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1

# Impute numerical missing values with mean
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean())

# Impute categorical missing values with -9999
fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)



# Assign numbers to categorical data 
from sklearn.preprocessing import LabelEncoder

for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))

# Normalize data to make computing faster
features = list(set(list(fullData.columns))-set(ID_col)-set(target_col))
fullData[features] = fullData[features]/fullData[features].max()



# Creata a validation set from the merged test set
from sklearn.model_selection import train_test_split

train=fullData[fullData['Type']==1]
test=fullData[fullData['Type']==0]

features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(flag_col))

X = train[features].values
y = train[target_col].values

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=42)



# LEARNING THE DATA ####################################################

# Define model
model = Sequential()
model.add(Dense(100, input_dim=11, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
model.summary() #Print model Summary

# Compile model
model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

# Fit Model (learn)
model.fit(X_train, y_train, epochs=10)

# Evaluate model
pred= model.predict(X_valid)
score = np.sqrt(mean_squared_error(y_valid,pred))
print ('Final score: {}'.format(score))