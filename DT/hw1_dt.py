import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        num_cls = np.unique(labels).size
        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

    def predict_with_node(self, features, node):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            # print("#######For feature "+ str(feature))
            pred = self.root_node.predict_with_node(feature, node)
            y_pred.append(pred)
        return y_pred


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        # print("Feature is "+str(features))
        # print("Label is " + str(labels))
        # print("split on attribute " + str(self.dim_split))
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split
        self.feature_uniq_split = None  # the possible unique values of the feature to be split
        # if self.splittable:
        #   self.split()

    def entropy(self, label):
        count = []
        for l in np.unique(label):
            count.append(label.count(l))
        total = np.sum(np.array(count))
        weight = count / total
        entropy_val = np.sum([(-1) * w * np.log2(w) for w in weight])
        return entropy_val

    def calculate_information_gain(self, features_array, label_array, thisdict, S):
        inforamtion_gain = []
        num_features = len(features_array[0])
        for i in range(num_features):
            cur_features = features_array[:, i]
            uni_features = np.unique(cur_features)
            branches = []
            for val in uni_features:
                label_for_each_value = label_array[features_array[:, i] == val]
                attribute_num = len(np.unique(label_for_each_value))
                num_for_each_class = [0] * len(thisdict)
                for element in np.unique(label_for_each_value):
                    times = label_for_each_value[label_for_each_value == element].size
                    num_for_each_class[thisdict[element]] = times
                branches.append(num_for_each_class)
            res = Util.Information_Gain(S, branches)
            inforamtion_gain.append((res, uni_features, i))
        return sorted(inforamtion_gain, key=lambda x: (-x[0], -len(x[1]), x[2]))

    # TODO: try to split current node
    def split(self):
        S = self.entropy(self.labels)
        thisdict = {}
        unique_label = np.unique(self.labels)
        for i in range(len(unique_label)):
            thisdict[unique_label[i]] = i
        features_array = np.array(self.features)
        label_array = np.array(self.labels)
        information_gain = self.calculate_information_gain(features_array, label_array, thisdict, S)
        if information_gain[0][0] <= 0:
            self.splittable = False
            return
        self.dim_split = information_gain[0][2]
        self.feature_uniq_split = information_gain[0][1]
        for attribute_val in np.unique(features_array[:, self.dim_split]):
            features_tmp = features_array[features_array[:, self.dim_split] == attribute_val]
            features = np.delete(features_tmp, self.dim_split, axis=1)
            labels = label_array[features_array[:, self.dim_split] == attribute_val]
            num_cls = np.unique(labels).size
            # print("！！！！！For atrribute value " + str(attribute_val) +" "+ str(features) + " " + str(labels) + " and there are " + str(num_cls) +" classes")
            children = TreeNode(features.tolist(), labels.tolist(), num_cls)
            if np.array(children.features).size == 0:
                children.splittable = False
            # print("%%%"+str(len(children.children)))
            self.children.append(children)

        for child in self.children:
            if child.splittable:
                child.split()

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable:
            index = self.feature_uniq_split.tolist().index(feature[self.dim_split])  # find the branch index to follow
            feature = np.delete(feature,
                                self.dim_split)  # update feature array to make sure it has the same index as feature_uniq_split
            return self.children[index].predict(feature)
        else:
            return self.cls_max

    def predict_with_node(self, feature, node):
        # feature: List[any]
        # return: int
        if self.splittable:
            if self == node:
                return self.cls_max
            index = self.feature_uniq_split.tolist().index(feature[self.dim_split])  # find the branch index to follow
            feature = np.delete(feature,
                                self.dim_split)  # update feature array to make sure it has the same index as feature_uniq_split
            return self.children[index].predict_with_node(feature, node)
        else:
            return self.cls_max