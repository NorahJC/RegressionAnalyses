##Norah Jean-Charles
##2/4/2019
##Polynomial Regression
##Dr.Aledhari
##CS4267-Machine Learning
##Section 1
##Spring 2019


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Propose-Salaries-Polynomial.csv')
#print(dataset)

X = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression()   
lin.fit(X, y) 

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression()
lin2.fit(X_poly, y)

# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Predicting Salaries Based on Employee Levels') 
plt.xlabel('Employee Levels') 
plt.ylabel('Salary') 
plt.show()

# Predicting a new result with Polynomial Regression
##lin2.predict(poly.fit_transform(6.5))
##print('Done')

#print ('A close approximation is 1726723')
