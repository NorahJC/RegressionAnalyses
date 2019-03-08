# Norah Jean-Charles
# 2/11/2019
# Logistic Regression
# Dr.Aledhari
# CS4267-Machine Learning
# Section 1
# Spring 2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('advertising_appeals.csv')
# print(dataset)

# Split dataset
X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, 4].values
# y = dataset.iloc[:, 4:5].values
# print(X)
# print(y)

# Perform Logistic Regression
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        # print(intercept)
        return np.concatenate((intercept, X), axis=1)

    # logistic function
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        # print((-y * np.log(h) - (1 - y) * np.log(1 - h)).mean())
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if self.verbose == True and i % 10000 == 0:
                print(f'loss: {loss} \t')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

model = LogisticRegression(lr=0.1, num_iter=300000)
model.fit(X, y)
preds = model.predict(X)
(preds == y).mean()
print(model.theta)

# Print graph
plt.figure(figsize=(10, 6))
plt.title('Predicting Purchasing Habits Based on Age and Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='y = 0')  # Not purchased
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='y = 1')  # Purchased
plt.legend()
x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
# xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# grid = np.c_[xx1.ravel(), xx2.ravel()]
# probs = model.predict_prob(grid).reshape(xx1.shape)
# plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
# plt.plot(x1, x2, c='k', label='reg line')
xx1 = np.linspace(20, 60, endpoint= True)
xx2 = np.linspace(150000, 15000, endpoint= True)
plt.plot(xx1, xx2, '-', c='k', label='reg line')
# plt.ylim(bottom=0)
# plt.ylim(top=150000)
# plt.xlim(left=15)
plt.show()

# Helpful Sites
# https://www.geeksforgeeks.org/numpy-zeros-python/
# https://github.com/martinpella/logistic-reg/blob/master/logistic_reg.ipynb
# https://github.com/nikhilkumarsingh/Machine-Learning-Samples/blob/master/Logistic_Regression/dataset1.csv
# https://www.geeksforgeeks.org/understanding-logistic-regression/
