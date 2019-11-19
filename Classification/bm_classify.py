import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    y_update = np.where(y == 0, -1, 1)
    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        step_size /= N
        for i in range(max_iterations):
            sign = y_update * (X.dot(w) + b)
            sign = np.where(sign <= 0, 1, 0)
            update = (sign * y_update).T.dot(X)
            w += step_size * update
            b += step_size * np.sum(sign * y_update)
        '''
        step_size /= N
        for i in range(max_iterations):
            y_predict = binary_predict(X, w, b, loss)
            diff = y - y_predict
            #diff = np.where(diff <= 0, 0, 1)
            update = np.dot(diff, X)
            w = w + step_size * update
            b = b + step_size * sum(diff)
        '''
        ####

        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        step_size /= N
        for i in range(max_iterations):
            sign = (-1) * y_update * (X.dot(w) + b)
            sign = sigmoid(sign)
            update = (sign * y_update).T.dot(X)
            w += step_size * update
            b += step_size * np.sum(sign * y_update)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        # preds = np.zeros(N)
        tmp_predict = np.dot(w, X.T) + b
        preds = np.where(tmp_predict > 0, 1, 0)
        ############################################


    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        # preds = np.zeros(N)
        tmp_predict = np.dot(X, w) + b
        preds = np.where(tmp_predict > 0, 1, 0)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        for i in range(max_iterations):
            index = np.random.choice(N)
            y_predict = X[index].dot(w.T) + b
            y_predict -= y_predict.max()
            numerator = np.exp(y_predict)
            denominator = numerator.sum()

            softmax = numerator / denominator
            softmax[y[index]] -= 1
            update = np.dot(softmax.reshape(C, 1), X[index].reshape(1, D))
            w -= step_size * update
            b -= step_size * softmax
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        step_size /= N
        y_matrix = np.zeros([N, C])
        for i, c in enumerate(y):
            y_matrix[i][c] = 1
        for i in range(max_iterations):
            y_predict = X.dot(w.T) + b  # N*C
            numerator = np.exp(y_predict)
            denominator = numerator.sum(axis=1).reshape(N, 1)
            softmax = numerator / denominator
            softmax -= y_matrix
            update = np.dot(softmax.T, X)
            w -= step_size * update
            b -= step_size * softmax.sum(axis=0)
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    # preds = np.zeros(N)
    preds = np.dot(X, w.T) + b
    preds = np.argmax(preds, axis=1)
    ############################################

    assert preds.shape == (N,)
    return preds




