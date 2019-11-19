import numpy as np

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
    err = np.sum(np.absolute(y_predict - y))
    return err


X = np.array([[1, 2, 3], [4, 5, 6], [1, 2, 5], [3, 2, 7]])

y = np.array([[1], [0], [0], [1]])

w = np.array([[1], [1], [2]])

print(mean_absolute_error(w, X, y))


a = np.array([[1, 2], [3, 4], [5, 6]])
print(a.shape[1])
b = np.array([[0,1], [2, 3], [4, 5]])
a = np.append(a, b, 1)
print(a)