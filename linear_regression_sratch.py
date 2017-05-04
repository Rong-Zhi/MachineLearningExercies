from __future__ import division
import numpy as np
from sklearn import cross_validation
from sklearn import datasets
import matplotlib.pyplot as plt
import math

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = \
    cross_validation.train_test_split(X, y, test_size=0.2)

iterations = 1500
alpha = 0.01
theta = np.array([0, 0])
m = len(y_train)
# print(np.shape(X_train))

def cost_function(X, y, theta):
    """
    cost_function(X, y, beta) computes the cost of using beta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y)

    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(theta)-y)**2)/2/m

    return J


def gradient_descent(X, y, theta, alpha, iterations):
    """
    gradient_descent Performs gradient descent to learn theta
    theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    """
    cost_history = [0] * iterations

    for iteration in range(iterations):
        hypothesis = X.dot(theta)
        loss = hypothesis - y
        gradient = X.T.dot(loss) / m
        theta = theta - alpha * gradient
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost

    return theta, cost_history

# print(cost_function(X_train,y_train,theta))

(t, c) = gradient_descent(X_train,y_train,theta,alpha, iterations)
print(t,c)
##############################
# not finished yet