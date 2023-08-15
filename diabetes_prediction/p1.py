# Model Creation

#import lib

import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report
import pickle

#load the data

data = pd.read_csv("diabetes_m2023.csv")
print(data)

# check for null data 
print(data.isnull().sum())

# features and target 
features = data[["FS", "FU"]]
target = data["Diabetes"]

#handle cat data 
nfeatures= pd.get_dummies (features)
print(features) 
print(nfeatures)

# train and test

x_train, x_test, y_train, y_test  = train_test_split(nfeatures, target)

#model

model = LogisticRegression() 
model.fit(x_train, y_train)

# performance

cr = classification_report(y_test, model.predict(x_test))
print(cr)
print (nfeatures)

# train and test

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

#model

model= LogisticRegression()
model.fit(x_train, y_train)

# performance

cr = classification_report(y_test, model.predict(x_test))
print(cr)

# model save 
f = open("db.model", "wb")
pickle.dump (model, f) 
f.close()