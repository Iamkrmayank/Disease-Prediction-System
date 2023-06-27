

#for dealing with multidimensional arrays
import numpy as np
#for data analysis on the dataset
import pandas as pd
#for data mining and data processing
import sklearn
#for visualizations & graphs
import matplotlib.pyplot as plt
import seaborn as sns
#to standardize data in a particular range
from sklearn.preprocessing import StandardScaler
#to spit data into train & test respectively
from sklearn.model_selection import train_test_split
#importing the required ML algorithms
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import os

a=os.getcwd():
print(a)
path = a + '\data\diabetes.csv'
#load diabetes dataset to pandas dataframe
df=pd.read_csv(path)
#Removing the null values
diabetes=df.dropna()
#print first 5 rows
df.head()
#check whether cols are true/false based on dtypes
df.info()
#data correlation
df.corr()

#number of rows and columns in the dataset
df.shape

#getting statistical measures of data
df.describe()

#counts for diabetic and non diabetic
df['Outcome'].value_counts()

df.groupby('Outcome').mean()

#seperate data and labels
X= df.drop(columns='Outcome',axis=1)
Y= df['Outcome']

print(X)

print(Y)

scaler= StandardScaler()

scaler.fit(X)

standardized_data=scaler.transform(X)

#in order to bring values within the similar range
print(standardized_data)
  
X= standardized_data
Y= df['Outcome']

print(X)
print(Y)

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.25,stratify=Y,random_state=42)

print(X.shape,X_train.shape,X_test.shape)



classifier = svm.SVC(kernel='linear')

#training the support vector machine classifier
classifier.fit(X_train, Y_train)

log_reg = LogisticRegression(random_state=1)

log_reg.fit(X_train,Y_train)

y_pred=log_reg.predict(X_test)
y_pred


#randomforest
forest=RandomForestClassifier(n_estimators=100,criterion="entropy",random_state=2)
forest.fit(X_train,Y_train)

#predict for forest classifier.
y_pred=forest.predict(X_test)
y_pred
# KNN 
KNN=KNeighborsClassifier(n_neighbors=1)
KNN.fit(X_train,Y_train)

#predict for KNN.
y_pred=KNN.predict(X_test)

print(confusion_matrix(Y_test,y_pred))
print(classification_report(Y_test,y_pred))


#accuracy score on the training data
X_train_prediction=classifier.predict(X_train)
#accuracy score on test data
X_test_prediction=classifier.predict(X_test)

training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print('Accuracy score of the training data:',training_data_accuracy)
print('Accuracy score of the test data:',test_data_accuracy)

#accuracy score for logisitic regression
print("Logistic Regression Classifier  Accuracy: ",accuracy_score(Y_test,y_pred))

#accuracy score for Random Forest
print("Random Forest Classifier Training Accuracy: ",forest.score(X_test,Y_test))

#accuracy score for KNN 
print("KNN classifier training accuracy: ",KNN.score(X_test,Y_test))
