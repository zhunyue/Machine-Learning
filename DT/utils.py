import numpy as np


# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches_array = np.array(branches)
    sum = np.sum(branches_array)
    sum_dimension = np.sum(branches_array, axis=1)
    weight = sum_dimension / sum  # weight for each attribute value
    weight_for_attribute = []
    for i in range(len(branches_array)):
        weight_for_attribute.append(np.divide(branches_array[i], sum_dimension[i]))
    logSum = np.log2(weight_for_attribute)
    logSum[np.isinf(logSum)] = 0
    entropy = np.sum((weight_for_attribute * logSum) * (-1), axis=1)
    weighted_entropy_for_branches = np.sum(np.multiply(np.array(entropy), np.array(weight)))
    return S - weighted_entropy_for_branches

def calculate_accuracy(predict_label, real_label):
    same = 0
    for l1, l2 in zip(predict_label, real_label):
        if l1 == l2:
            same += 1
    return same/len(predict_label)

def bottom_up_level_order(node):
    res = []
    queue = []
    queue.append(node)

    while (len(queue) > 0):

        root = queue.pop(0)
        res.append(root)
        for child_node in reversed(root.children):
            queue.append(child_node)
    res.reverse()
    return res

def prune(decisionTree, X_test, y_test):
    y_predict_with_original = decisionTree.predict(X_test)
    # print("Predict using original Tree: " + str(y_predict_with_original))
    bottom_up = bottom_up_level_order(decisionTree.root_node)
    original_accuracy = calculate_accuracy(y_predict_with_original, y_test)
    # print("Original accuracy is : " + str(original_accuracy))
    max_node = None
    max_accuracy = -1

    for i, x in enumerate(bottom_up):
        print(x.labels)
        if x.splittable:
            tmp_predict = decisionTree.predict_with_node(X_test, bottom_up[i])
            accuracy = calculate_accuracy(tmp_predict, y_test)
            if accuracy > max_accuracy:
                max_node = bottom_up[i]
                max_accuracy = accuracy
    print("The max accuracy is " + str(max_accuracy))
    if max_accuracy >= original_accuracy and max_node is not None:
        print("The index to prune is " + str(max_node.labels))
        max_node.splittable = False
        max_node.children = []
        max_node.dim_split = None
        max_node.feature_uniq_split = None
    else:
        max_node = None
    return max_node

# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    while True:
        max_node = prune(decisionTree, X_test, y_test)
        if max_node is None:
            break


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')
