# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 22:33:16 2019

@author: Pranjall
"""



#importing libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset.
dataset = pd.read_csv("Pima_Indian_diabetes.csv")
X = np.array(dataset.iloc[:, :-1].values)  
Y = np.array(dataset.iloc[:, 8].values)
Y = Y.reshape(len(Y), 1)





#exploratory data analysis.

#measuring correlations between attributes.
initial_corr = dataset.corr(method = 'pearson')

#Counting missing values.
len(dataset) - dataset.count()




    
#Cleaning data one dimension at a time.

#Pregnancies: No data implies patient is assumed to have zero pregancies (since gender of subjects is not mentioned).
"""Internet search: Pregnancy is correlated to glucose buildup in the body and weight gain, affecting BMI which are important factors for diabetes.""" 
#replacing missing values with zero.
X[np.isnan(X[:, 0]), 0] = 0
#plotting histogram.
fig1, ax1 = plt.subplots(2)
fig1.suptitle('Old and new histograms for Pregnancies')
ax1[0].hist(X[:, 0], bins = 50)
#making negative values zero and rounding off.
X[:, 0] = np.round(X[:, 0].clip(0, max(X[:, 0])))
#plotting new histogram.
ax1[1].hist(X[:, 0], bins = 50)
        

#Glucose: Zero glucose level is absurd, so we replace all missing values with zero and then replace them by median of non-zero values with sampling.
#Median is used since the histogram depicts that the data is skewed.
#replacing missing values with zero.
X[np.isnan(X[:, 1]), 1] = 0
#plotting histogram
fig2, ax2 = plt.subplots(2)
fig2.suptitle('Old and new histograms for Glucose')
ax2[0].hist(X[:, 1], bins = 100)
#replacing 0 glucose levels with median of data.
from sklearn import impute
imp_median = impute.SimpleImputer(missing_values = 0, strategy = "median")
imp_median.fit(X[:, 1:2])
X[:, 1:2] = imp_median.transform(X[:, 1:2])
#rounding off values.
X[:, 1] = np.round(X[:, 1])
#plotting new histogram
ax2[1].hist(X[:, 1], bins = 100)


#BloodPressure: No missing values.
#plotting histogram.
fig3, ax3 = plt.subplots(2)
fig3.suptitle('Old and new histograms for Blood Pressure')
ax3[0].hist(X[:, 2], bins = 100)
#making negative values zero and rounding off values.
X[:, 2] = np.round(X[:, 2].clip(0, max(X[:, 2])))
#replacing 0 glucose levels with median of data.
imp_median.fit(X[:, 2:3])
X[:, 2:3] = imp_median.transform(X[:, 2:3])
#plotting new histogram
ax3[1].hist(X[:, 2], bins = 100)


#SkinThickness is not a useful feature for diabetes prediction.
X = np.delete(X, 3, 1)


#Insulin: No missing data, and 0 insulin level is possible. So left as it is.
""" Internet search: Fasting insulin level should never be 0, which it might be in a person with untreated Type 1. 
It shouldn’t go below 3. But a high insulin level is just as problematic. A high insulin level is a sign of insulin resistance or prediabetes. 
It can also signify early-stage Type 2. """



#BMI:
#replacing missing values with zero.
X[np.isnan(X[:, 4]), 4] = 0
#making negative values zero and rounding off values.
X[:, 4] = np.round(X[:, 4].clip(0, max(X[:, 4])), decimals = 1)
#plotting histogram
fig5, ax5 = plt.subplots(2)
fig5.suptitle('Old and new histograms for BMI')
ax5[0].hist(X[:, 4], bins = 100)
#replacing 0 BMI levels with median of data.
imp_median.fit(X[:, 4:5])
X[:, 4:5] = imp_median.transform(X[:, 4:5])
#plotting new histogram
ax5[1].hist(X[:, 4], bins = 100)


#DiabetesPedigreeFunction: no missing values


#Age
#replacing missing values with zero.
X[np.isnan(X[:, 6]), 6] = 0
#making negative values zero and rounding off values.
X[:, 6] = np.round(X[:, 6].clip(0, max(X[:, 6])))
#plotting histogram
fig6, ax6 = plt.subplots(2)
fig6.suptitle('Old and new histograms for Ages')
ax6[0].hist(X[:, 6], bins = 100)
#replacing 0 BMI levels with median of data.
imp_median.fit(X[:, 6:7])
X[:, 6:7] = imp_median.transform(X[:, 6:7])
#plotting new histogram
ax6[1].hist(X[:, 6], bins = 100)


#Visualizing manipulated data correlations.
features = []
for i in range(X.shape[1]):
    features.append(X[:, i])
corr = np.corrcoef(features)
print(initial_corr)
print(corr)
"""Conclusion: All the data manipulations preserve the initial information"""





#Data Preprocessing

#feature scaling.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#Adding some flexibility for the hypothesis function, performs best with a quadratic polynomial.
#adding polynomial terms.
from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 2)
X = poly_regressor.fit_transform(X)


""" Removing outliers in the dataset did not affect the performance much in this case, Hence avoided """



#Model training and evaluation

#splitting data.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.13, random_state = 420)

#fitting the Logistic Regression
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression(solver = 'liblinear')
regressor.fit(X_train, Y_train.ravel())

#predicting values.
Y_pred = regressor.predict(X_test)

#confusion matrix.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print()
print(cm)
accuracy = str((cm[0][0]+cm[1][1])/len(Y_test)*100)
print()
print('Prediction Accuracy: ' + accuracy)