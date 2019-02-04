##Norah Jean-Charles
##2/4/2019
##Simple Linear Regression
##Dr.Aledhari
##CS4267-Machine Learning
##Section 1
##Spring 2019

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Salaries-Simple_Linear.csv')
print(dataset)

# Gives (# of rows, # of col)
##print(dataset.shape)

# Divide data set into x(years) and y(salaries)
x =  dataset.iloc[:, :-1].values
##x =  dataset['Years_of_Expertise'].values##does not transpose values
##print(x)
y =  dataset.iloc[:, 1].values
##y = dataset['Salary'].values##does not do transpose
##print(y)

# Get mean
mean_x = np.mean(x)
mean_y = np.mean(y)
print('Means: x = %.3f, y = %.3f' % (mean_x, mean_y))

###or
##
###calc the mean value of a list of #s
##def mean(values):
##    return sum(values) / float(len(values))
##
###calc variance of a list of numbers
##def variance(values, mean):
##    return sum([(x-mean)**2 for x in values])
###get mean and variance
##mean_x, mean_y = mean(x), mean(y)
##var_x, var_y = variance(x, mean_x), variance(y, mean_y)
##print('x stats: mean=%.3f variance=%.3f' % (mean_x, var_x))
##print('y stats: mean=%.3f variance=%.3f' % (mean_y, var_y))
##
##
###calc covariance between x and y
##def covariance(x, mean_x, y, mean_y):
##    covar = 0.0
##    for i in range(len(x)):
##        covar += (x[i] - mean_x) * (y[i] - mean_y)
##    return covar
##
###calc the mean and variance
##covar = covariance (x, mean_x, y, mean_y)
##print('Corvariance: %.3f' % (covar))
##
###calc coefficients
##def coefficients(dataset):
##	x_mean, y_mean = mean(x), mean(y)
##	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
##	b0 = y_mean - b1 * x_mean
##	return [b0, b1]
##
### calc the coefficients
##b0, b1 = coefficients(dataset)
##print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))

# Total number of values
m = len(x)

# Use regression model to calculate b1 and b2
numer = 0
denom = 0
for i in range(m):
    numer += (x[i] - mean_x) * (y[i] - mean_y)
    denom += (x[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

# Print coefficients
print('Coefficents: b0 = %.3f, b1 = %.3f' % (b0, b1))

# Print cost function/ linear regression model
a = 'Salary'
b = 'Years_of_Expertise'
cfunct =  '%s = %.3f + %.3f * %s' % (a, b0, b1, b)

# Print cost function equation
print('Based on the linear regression model, y = B0 + B1 * x, the cost function is \n\t %s' % (cfunct))

##
### Plot values and regression line
##max_x = np.max(x) + 100
##min_x = np.min(x) - 100
##
### Calculating line values x and y
##x1 = np.linspace(min_x, max_x, 1000)
##y1 = b0 + b1 * x1
### Ploting Line
##plt.plot(x1, y1, color='#58b970', label='Regression Line')
### Ploting Scatter Points
##plt.scatter(x, y, color='#ef5423', label='Scatter Plot')
##
##plt.xlabel('Years of Expertise')
##plt.ylabel('Salary')
##plt.legend()
##plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
simplelinearRegression = LinearRegression()
simplelinearRegression.fit(x_train, y_train)

y_predict = simplelinearRegression.predict(x_test)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, simplelinearRegression.predict(x_train))
plt.title('Predicting Salary Based on Years of Expertise')
plt.xlabel('Years of Expertise')
plt.ylabel('Salary')
plt.show()






