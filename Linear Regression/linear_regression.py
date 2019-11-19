"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    y_predict = np.dot(X, w)
    err = np.mean(np.absolute(y_predict - y))
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################
  w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    target_matrix = X.T.dot(X)
    to_add = np.identity(target_matrix.shape[0]) * 0.1
    while True:
        eigen_value = np.linalg.eigvals(target_matrix)
        if np.amin(np.absolute(eigen_value)) < 10**(-5):
            target_matrix += to_add
            break

    w = np.linalg.inv(target_matrix).dot(X.T).dot(y)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################
    target_matrix = X.T.dot(X)
    toAdd = lambd * np.identity(target_matrix.shape[0])
    w = np.linalg.inv(target_matrix + toAdd).dot(X.T).dot(y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    least_mae = None
    for i in range(-19, 20):
        lambd = 10 ** i
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        mae = mean_absolute_error(w, Xval, yval)
        if least_mae is None or mae < least_mae:
            least_mae = mae
            bestlambda = lambd
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    print("Original shape is " + str(X.shape[1]))
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    original = X
    for i in range(2, power+1):
        new_matrix = original ** i
        X = np.append(X, new_matrix, 1)
    return X

    '''
    X_array = [X]
    for i in range(1, power):
        X_array.append(X**(i+1))
    return np.concatenate(X_array, axis=1)
    '''


