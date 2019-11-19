import numpy as np
from scipy.spatial import distance
from collections import Counter

real_labels = [1,2,3]
predicted_labels = [2,3,4]
real_array = np.asarray(real_labels);
predict_array = np.asarray(predicted_labels)
real_sum = np.sum(real_array);
print(real_sum)
predict_sum = np.sum(predict_array);
print(predict_sum)
print( 2 * np.dot(predict_array, real_array) / (real_sum + predict_sum))


point1 = [1, 0, 0]
point2 = [0,1,0]
diff = np.asarray(point1) - np.asarray(point2)
print(distance.minkowski(point1, point2, 3))

print(distance.euclidean([1, 1, 0], [0, 1, 0]))

print(np.inner(real_labels, predicted_labels))

print(distance.cosine(point1, point2))

print( -1 * np.exp((-1/2)*np.inner(diff, diff)))


x = [('abc', 121),('abc', 231),('abc', 148), ('abc',221)];
print(sorted(x, key=lambda x: x[1]))
y = np.asarray(x)


def euclidean_distance(point1, point2):
    """
    :param point1: List[float]
    :param point2: List[float]
    :return: float
    """
    return distance.euclidean(point1, point2)

def get_k_neighbors(point):
    """
    This function takes one single data point and finds k-nearest neighbours in the training set.
    You already have your k value, distance function and you just stored all training data in KNN class with the
    train function. This function needs to return a list of labels of all k neighours.
    :param point: List[float]
    :return:  List[int]
    """
    train_features = [[2, 1], [3, 1], [4, 1]]
    train_labels = [1, 3, 0]
    distances = []
    neighbors = []
    for i in range(len(train_labels)):
        dist = euclidean_distance(train_features[i], point)
        distances.append([train_labels[i], dist])
        print()
    sorted_distances = sorted(distances, key=lambda x: x[1])
    for j in range(3):
        print(sorted_distances[j][0])
        neighbors.append(sorted_distances[j][0])
    return neighbors

def predict(features):
        """
        This function takes 2D list of test data points, similar to those from train function. Here, you need process
        every test data point, reuse the get_k_neighbours function to find the nearest k neighbours for each test
        data point, find the majority of labels for these neighbours as the predict label for that testing data point.
        Thus, you will get N predicted label for N test data point.
        This function need to return a list of predicted labels for all test data points.
        :param features: List[List[float]]
        :return: List[int]
        """
        predict_label = []
        for point in features:
            neighbors = get_k_neighbors( point)
            counter = Counter(neighbors)
            predict_label.append(counter.most_common(1)[0][0])
        return predict_label


x= [[1,2], [2,3], [3,4]]

print("***")
print(predict(x))


for k in range(1, 30, 2):
    print(k)


def test_normalize(features):
    normalize_features = []
    feature_array = np.asarray(features)
    for feature in feature_array:
        inner_product = np.sqrt(np.inner(feature, feature))
        if inner_product != 0:
          normalize_for_feature = feature/np.sqrt(np.inner(feature, feature))
        else:
          normalize_for_feature = [0, 0]
        normalize_features.append(normalize_for_feature)

    return normalize_features

print("*********")
features = [[2, -1], [-1, 5], [0, 0]]
feature2 = [[2, -1], [-1, 5], [0, 0]]
feature_array = np.array(features)
normalize_data = feature_array / np.linalg.norm(feature_array, ord=2, axis=1, keepdims=True)
print("$$$")
print(normalize_data)
normalize_data[np.isnan(normalize_data)] = 0
print("$$$$")
print(normalize_data)
test = test_normalize(features)
print(test)

features_array = np.asarray(features)
max = np.amax(features_array, axis=0)
min = np.min(features_array, axis=0)
print(max)
print(min)
print((features_array-min)/(max - min))

distance_funcs = {
        'euclidean': 1,
        'minkowski': 2,
        'gaussian': 3,
        'inner_prod': 4,
        'cosine_dist': 5,
    }

print(distance_funcs[None])


