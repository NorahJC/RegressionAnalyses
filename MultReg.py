##Norah Jean-Charles
##2/4/2019
##Multiple Regression
##Dr.Aledhari
##CS4267-Machine Learning
##Section 1
##Spring 2019


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics

# Import dataset
dataset = pd.read_csv('3-Products-Multiple.csv')
# print(dataset)

X = dataset.iloc[:, :-1]
# print(X)
y = dataset.iloc[:, 4]
# print(y)

# convert column into categorical columns
cities = pd.get_dummies(X['Location'], drop_first=True)

# drop city column
X = X.drop('Location', axis=1)

# concat dummy variables
X = pd.concat([X, cities], axis=1)

# split data into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: \n', reg.coef_)

# variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

##from sklearn.metrics import r2_score
##score=r2_score(y_test, y_pred)
##print('r^2 (variance) = ')
##print(score)

# predicting the test set results
y_pred = reg.predict(X_test)
print('Prediction: ')
print(y_pred)

# print statment
print('Based on the results, product_1 would yield a better profit at a particular city  and overall.')

# plot for residual error

# setting plot style
plt.style.use('fivethirtyeight')

# plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

# plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

# plotting line for zero residual error
plt.hlines(y=0, xmin=-1000, xmax=200000, linewidth=2)

# plotting legend
plt.legend(loc='upper right')

# plot title
plt.title("Residual errors")

# function to show plot
plt.show()

