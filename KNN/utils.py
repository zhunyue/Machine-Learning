import numpy as np
from knn import KNN


############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    real_array = np.asarray(real_labels)
    predict_array = np.asarray(predicted_labels)
    real_sum = np.sum(real_array)
    predict_sum = np.sum(predict_array)
    print(np.dot(predict_array, real_array))
    return 2 * np.dot(predict_array, real_array) / (real_sum + predict_sum)


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.linalg.norm(np.array(point1) - np.array(point2), ord=3)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        return np.inner(point1, point2)

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1_array = np.array(point1)
        point2_array = np.array(point2)
        norm1 = np.linalg.norm(point1_array)
        norm2 = np.linalg.norm(point2_array)
        inner_product = np.dot(point1_array, point2_array)
        return 1 - inner_product / (norm1 * norm2)

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        diff = np.asarray(point1) - np.asarray(point2)
        return -1 * np.exp((-1 / 2) * np.inner(diff, diff))


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        priority = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist']
        score = -1
        for k in range(1, 30, 2):
            for key, value in distance_funcs.items():
                model = KNN(k, value)
                model.train(x_train, y_train)
                predict_val = model.predict(x_val)
                tmp_score = f1_score(predict_val, y_val)
                if tmp_score > score:
                    score = tmp_score
                    self.best_k = k
                    self.best_distance_function = key
                    self.best_model = model
                elif tmp_score == score and priority.index(key) < priority.index(self.best_distance_function):
                    score = tmp_score
                    self.best_k = k
                    self.best_distance_function = key
                    self.best_model = model

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        priority_dis = ['euclidean', 'minkowski', 'gaussian', 'inner_prod', 'cosine_dist']
        priority_scaler = ['min_max_scale', 'normalize']
        score = -1
        for key, value in scaling_classes.items():
            scaler = value()
            train_features_scaled = scaler(x_train)
            test_features_scaled = scaler(x_val)
            for k in range(1, 30, 2):
                for key_dis, value_dis in distance_funcs.items():
                    model = KNN(k, value_dis)
                    model.train(train_features_scaled, y_train)
                    predict_val = model.predict(test_features_scaled)
                    tmp_score = f1_score(predict_val, y_val)
                    if tmp_score > score:
                        score = tmp_score
                        self.best_k = k
                        self.best_distance_function = key_dis
                        self.best_model = model
                        self.best_scaler = key
                    elif tmp_score == score:
                        if priority_scaler.index(key) < priority_scaler.index(key):
                            score = tmp_score
                            self.best_k = k
                            self.best_distance_function = key_dis
                            self.best_model = model
                            self.best_scaler = key
                        elif priority_scaler.index(key) == priority_scaler.index(key) and priority_dis.index(
                                key_dis) < priority_dis.index(self.best_distance_function):
                            score = tmp_score
                            self.best_k = k
                            self.best_distance_function = key_dis
                            self.best_model = model
                            self.best_scaler = key


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        """
        normalize_features = []
        feature_array = np.asarray(features)
        for feature in feature_array:
            inner_product = np.sqrt(np.inner(feature, feature))
            if inner_product != 0:
                normalize_for_feature = feature/np.linalg.norm(feature)
            else:
                normalize_for_feature = feature - feature
            normalize_features.append(normalize_for_feature)
        return normalize_features
        """
        feature_array = np.array(features)
        normalize_data = feature_array / np.linalg.norm(feature_array, ord=2, axis=1, keepdims=True)
        normalize_data[np.isnan(normalize_data)] = 0
        return normalize_data


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.train = True
        self.min = []
        self.max = []

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features_array = np.array(features)
        if self.train:
            self.max = np.amax(features_array, axis=0)
            self.min = np.amin(features_array, axis=0)
            self.train = False
        normalized = (features_array - self.min) / (self.max - self.min)
        return normalized
