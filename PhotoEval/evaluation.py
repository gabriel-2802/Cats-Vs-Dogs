
from DataExtract import load_image

# the function maps any real value into another value between 0 and 1
# it is used to normalize the data
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# the function unrolls the parameters into 2 matrixes
def roll_parameters(theta, n):
    # k = number of neurons in hidden layer
    k = 40
    # reshape theta into 2 matrixes
    theta1 = np.reshape(theta[:k * (n + 1)], (k, n + 1))
    theta2 = np.reshape(theta[k * (n + 1):], (1, k + 1))

    return theta1, theta2

# the function calculates the layers of our neural network
def compute_layers(x, theta):
    rows, cols = x.shape
    theta1, theta2 = roll_parameters(theta, cols)

    # Add the bias unit to the input layer
    a_1 = np.hstack([np.ones((rows, 1)), x])

    # Calculate the hidden layer
    z_2 = np.dot(theta1, a_1.T)
    # Add the bias unit to the hidden layer
    a_2 = np.vstack([np.ones((1, z_2.shape[1])), sigmoid(z_2)])

    # Calculate the output layer
    z_3 = np.dot(theta2, a_2)
    a_3 = sigmoid(z_3)

    return a_1, a_2, a_3

# the function calculates the output layer of our neural network
def forward_progation(x, theta):
    a1, a2, a_3 = compute_layers(x, theta)

    # our prediction is the output layer
    return a_3

# the function predicts if the photo is a cat or a dog
def predict_photo(path):
    x = load_image(path)
    x = x.reshape(1, -1)

    # load the pretrained parameters
    optimal_theta = np.load("optimal_theta.npz")['optimal_theta']

    prediction = np.round(forward_progation(x, optimal_theta))

    if prediction == 0:
        print("image from " + path + " is a cat")
    else:
        print("image from " + path + " is a dog")


def cost_function(theta, X, y, lambda_):
    m, n = X.shape

    # Roll parameters
    # Assuming 'roll_parameters' appropriately converts 'theta' into Theta_1 and Theta_2
    Theta_1, Theta_2 = roll_parameters(theta, n)

    # Forward propagation
    a_1 = np.vstack((np.ones((1, m)), X.T))
    z_2 = Theta_1.dot(a_1)
    a_2 = np.vstack((np.ones((1, m)), sigmoid(z_2)))
    z_3 = Theta_2.dot(a_2)
    a_3 = sigmoid(z_3)  # predictions

    # Backpropagation
    delta_3 = a_3 - y.T
    delta_2 = (Theta_2.T.dot(delta_3)) * (a_2 * (1 - a_2))
    delta_2 = delta_2[1:, :]

    # Unregularized gradients
    Theta_grad_2 = 1 / m * delta_3.dot(a_2.T)
    Theta_grad_1 = 1 / m * delta_2.dot(a_1.T)

    # Cost function with regularization
    J = -1 / m * np.sum(y.T * np.log(a_3) + (1 - y.T) * np.log(1 - a_3))
    J += lambda_ / (2 * m) * (np.sum(Theta_1[:, 1:] ** 2) + np.sum(Theta_2[:, 1:] ** 2))

    # Regularization for gradients
    Theta_grad_2[:, 1:] += lambda_ / m * Theta_2[:, 1:]
    Theta_grad_1[:, 1:] += lambda_ / m * Theta_1[:, 1:]

    # Unroll gradients
    gradient = np.concatenate([Theta_grad_1.ravel(), Theta_grad_2.ravel()])

    return J, gradient

# TODO choose epsilon : epsilon = sqrt(6) / sqrt(L_in + L_out)
def initialize_weights(epsilon, rows, cols):
    return 2 * epsilon * np.random.rand(rows, cols) - epsilon

def gradient_descent(x, y, initial_theta, lr, lmb, iterations):
    # m is the number of training examples
    m = x.shape[0]

    theta = initial_theta

    for i in range(1, iterations + 1):
        J, grad = cost_function(theta, x, y, lmb)

        # Adjust the learning rate over time
        learning_rate = lr / (5000 + 0.8 * i)

        # Update the parameters theta by taking a step in the direction of the negative gradient
        # The step size is proportional to the gradient and the learning rate
        theta = theta - learning_rate / m * grad

        # Calculate the norm of the gradient, which is used as a convergence check
        err = np.linalg.norm(grad)

        # If the gradient norm is less than the threshold, stop the iteration
        if err < 1e-5:
            break

    # Return the optimized parameters and the cost associated with these parameters
    return theta, J


# -*- coding: cp1252 -*-

# Minimize a continuous differentialble multivariate function. Starting point
# is given by "X" (D by 1), and the function named in the string "f", must
# return a function value and a vector of partial derivatives. The Polack-
# Ribiere flavour of conjugate gradients is used to compute search directions,
# and a line search using quadratic and cubic polynomial approximations and the
# Wolfe-Powell stopping criteria is used together with the slope ratio method
# for guessing initial step sizes. Additionally a bunch of checks are made to
# make sure that exploration is taking place and that extrapolation will not
# be unboundedly large. The "length" gives the length of the run: if it is
# positive, it gives the maximum number of line searches, if negative its
# absolute gives the maximum allowed number of function evaluations. You can
# (optionally) give "length" a second component, which will indicate the
# reduction in function value to be expected in the first line-search (defaults
# to 1.0). The function returns when either its length is up, or if no further
# progress can be made (ie, we are at a minimum, or so close that due to
# numerical problems, we cannot get any closer). If the function terminates
# within a few iterations, it could be an indication that the function value
# and derivatives are not consistent (ie, there may be a bug in the
# implementation of your "f" function). The function returns the found
# solution "X", a vector of function values "fX" indicating the progress made
# and "i" the number of iterations (line searches or function evaluations,
# depending on the sign of "length") used.
# %
# Usage: X, fX, i = fmincg(f, X, options)
# %
# See also: checkgrad
# %
# Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13
# %
# %
# (C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen

# Permission is granted for anyone to copy, use, or modify these
# programs and accompanying documents for purposes of research or
# education, provided this copyright notice is retained, and note is
# made of any changes that have been made.

# These programs and documents are distributed without any warranty,
# express or implied.  As the programs were written for research
# purposes only, they have not been tested to the degree that would be
# advisable in any important application.  All use of these programs is
# entirely at the user's own risk.

# [ml-class] Changes Made:
# 1) Function name and argument specifications
# 2) Output display

# [Iago López Galeiras] Changes Made:
# 1) Python translation

# [Sten Malmlund] Changes Made:
# 1) added option['maxiter'] passing
# 2) changed a few np.dots to np.multiplys
# 3) changed the conatenation line so that now it can handle one item arrays
# 4) changed the printing part to print the Iteration lines to the same row

import numpy as np
import sys
import numpy.linalg as la

from math import isnan, isinf


def div(a, b):
    return la.solve(b.T, a.T).T


# Refference: https://github.com/stena/ml/blob/master/fmincg.py
import numpy as np

def fmincg(f, X, options, *args):
    if 'MaxIter' in options:
        length = options['MaxIter']
    else:
        length = 100

    RHO = 0.01
    SIG = 0.5
    INT = 0.1
    EXT = 3.0
    MAX = 20
    RATIO = 100

    if isinstance(length, tuple):
        red = length[1]
        length = length[0]
    else:
        red = 1

    i = 0
    ls_failed = False
    fX = []
    f1, df1 = f(X, *args)
    i += (length < 0)
    s = -df1
    d1 = -np.sum(s**2)
    z1 = red / (1 - d1)

    while i < abs(length):
        i += (length > 0)
        X0 = X.copy()
        f0 = f1
        df0 = df1.copy()
        X = X + z1 * s
        f2, df2 = f(X, *args)
        i += (length < 0)
        d2 = np.dot(df2, s)
        f3 = f1
        d3 = d1
        z3 = -z1

        if length > 0:
            M = MAX
        else:
            M = min(MAX, -length - i)
        success = False
        limit = -1

        while True:
            while ((f2 > f1 + z1 * RHO * d1) or (d2 > -SIG * d1)) and (M > 0):
                limit = z1
                if f2 > f1:
                    z2 = z3 - (0.5 * d3 * z3**2) / (d3 * z3 + f2 - f3)
                else:
                    A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
                    B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
                    z2 = (np.sqrt(B**2 - A * d2 * z3**2) - B) / A
                if np.isnan(z2) or np.isinf(z2):
                    z2 = z3 / 2
                z2 = max(min(z2, INT * z3), (1 - INT) * z3)
                z1 = z1 + z2
                X = X + z2 * s
                f2, df2 = f(X, *args)
                M -= 1
                i += (length < 0)
                d2 = np.dot(df2, s)
                z3 = z3 - z2

            if f2 > f1 + z1 * RHO * d1 or d2 > -SIG * d1:
                break
            elif d2 > SIG * d1:
                success = True
                break
            elif M == 0:
                break

            A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3)
            B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2)
            z2 = -d2 * z3**2 / (B + np.sqrt(np.abs(B**2 - A * d2 * z3**2)))

            if not np.isreal(z2) or np.isnan(z2) or np.isinf(z2) or z2 < 0:
                if limit < -0.5:
                    z2 = z1 * (EXT - 1)
                else:
                    z2 = (limit - z1) / 2
            elif (limit > -0.5) and (z2 + z1 > limit):
                z2 = (limit - z1) / 2
            elif (limit < -0.5) and (z2 + z1 > z1 * EXT):
                z2 = z1 * (EXT - 1.0)
            elif z2 < -z3 * INT:
                z2 = -z3 * INT
            elif (limit > -0.5) and (z2 < (limit - z1) * (1.0 - INT)):
                z2 = (limit - z1) * (1.0 - INT)

            f3 = f2
            d3 = d2
            z3 = -z2
            z1 = z1 + z2
            X = X + z2 * s
            f2, df2 = f(X, *args)
            M -= 1
            i += (length < 0)
            d2 = np.dot(df2, s)

        if success:
            f1 = f2
            fX.append(f1)
            # print(f"Iteration {i} | Cost: {f1}")
            s = (np.dot(df2, df2) - np.dot(df1, df2)) / np.dot(df1, df1) * s - df2
            tmp = df1
            df1 = df2
            df2 = tmp
            d2 = np.dot(df1, s)
            if d2 > 0:
                s = -df1
                d2 = -np.sum(s**2)
            z1 = z1 * min(RATIO, d1 / (d2 - np.finfo(float).tiny))
            d1 = d2
            ls_failed = False
        else:
            X = X0
            f1 = f0
            df1 = df0
            if ls_failed or i > abs(length):
                break
            tmp = df1
            df1 = df2
            df2 = tmp
            s = -df1
            d1 = -np.sum(s**2)
            z1 = 1 / (1 - d1)
            ls_failed = True

    return X, np.array(fX), i

# Example usage:
# Define a function 'f' that returns function value and gradient
# X, fX, i = fmincg(f, X, options, P1, P2, P3, P4, P5