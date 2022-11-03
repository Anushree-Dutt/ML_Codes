# -*- coding: utf-8 -*-
"""Copy of ML EXP 1A.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FBumvYtHgj4g9MT8xJBfDjn0YMkegPZf

# **SIMPLE LINEAR REGRESSION USING ANALYTICAL METHOD**
"""

import pandas as pd

# Reading csv file from github repo
advertising = pd.read_csv('tvmarketing.csv')

# Display the first 5 rows
advertising.head()

# Let's check the columns
advertising.info()

# Check the shape of the DataFrame (rows, columns)
advertising.shape

# Let's look at some statistical information about the dataframe.
advertising.describe()

# Visualise the relationship between the features and the response using scatterplots
advertising.plot(x='TV',y='Sales',kind='scatter')

# Putting feature variable to X
x = advertising['TV']

# Print the first 5 rows
x.head()

# Putting response variable to y
y = advertising['Sales']

# Print the first 5 rows
y.head()

#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7 , random_state=0000)

print(type(x_train))
print(type(x_test))
print(type(y_train))
print(type(y_test))

train_test_split

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

def linear_regression(x, y):     
    N = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1*x_mean)
    
    reg_line = 'y = {} + {}x'.format(B0, round(B1, 3))
    
    return (B0, B1, reg_line)

B0, B1, reg_line = linear_regression(x_train, y_train)
print('Regression Line: ', reg_line)

x_input = int(input())
y_output = B0 + round(B1,3) * x_input
print("y = ",y_output)