import numpy as np

x = [[1,2], [2,1], [2,0]]
x_array = np.array(x)
sum_dimension = np.sum(x_array, axis=1)
sum_all = np.sum(x_array)
print(x_array)
print(sum_dimension)
print(sum_dimension/sum_all)
weight_for_attribute = []

for i in range(len(x_array)):
    weight_for_attribute.append(np.divide(x_array[i], sum_dimension[i]))
print("weight for each attribute:")
print(weight_for_attribute)



# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    branches_array = np.array(branches)
    sum = np.sum(branches_array)
    sum_dimension = np.sum(branches_array,axis=1)
    weight = sum_dimension/sum  #weight for each attribute value
    weight_for_attribute = []
    for i in range(len(branches_array)):
        weight_for_attribute.append(np.divide(branches_array[i], sum_dimension[i]))
    logSum = np.log2(weight_for_attribute)
    logSum[np.isinf(logSum)] = 0
    entropy = np.sum((weight_for_attribute*logSum)*(-1), axis=1)
    weighted_entropy_for_branches = np.sum(np.multiply(np.array(entropy), np.array(weight)))
    return S-weighted_entropy_for_branches

#print("********test information gain function")
branch = [[0,4,2], [2,0,4]]
information_gain = Information_Gain(0, branch)
print(information_gain)


def entropy(label):
    count = []
    for l in np.unique(label):
        count.append(label.count(l))
    sum_val = np.sum(np.array(count))
    weight = count/sum_val
    entropy_val = np.sum([(-1)*w*np.log2(w) for w in weight])
    #logSum = np.log2(weight)
    #logSum[np.isinf(weight)] = 0
    #entropy_val = np.sum((weight * logSum) * (-1))
    return entropy_val
entropy([0,0,1,1])


def calculate_information_gain(features_array, label_array, thisdict, S):
    inforamtion_gain = []
    num_features = len(features_array[0])
    for i in range(num_features):
        cur_features = features_array[:, i]
        uni_features = np.unique(cur_features)
        #print("Feature"+str(i))
        branches = []
        for val in uni_features:
            label_for_each_value = label_array[features_array[:, i] == val]
            attribut_num = len(np.unique(label_for_each_value))
            #print("## For value "+str(val)+", it has label "+str(attribut_num))
            num_for_each_class = [0] * len(thisdict)
            for element in np.unique(label_for_each_value):
                times = label_for_each_value[label_for_each_value == element].size
                num_for_each_class[thisdict[element]] = times
            branches.append(num_for_each_class)
        res = Information_Gain(S, branches)
        inforamtion_gain.append((res, attribut_num, i))
    return sorted(inforamtion_gain, key=lambda x: (-x[0], -x[1], x[2]))


print("*************************************************")
def split(features, labels):
    S = entropy(labels)
    thisdict = {}
    unique_label = np.unique(labels)
    for i in range(len(unique_label)):
        #print("^^^^"+str(i)+"^^^"+str(unique_label[i]))
        thisdict[unique_label[i]] = i
    #print("The dictionary is "+str(thisdict) + " and the length is "+str(len(thisdict)))
    features_array = np.array(features)
    label_array = np.array(labels)
    information_gain = calculate_information_gain(features_array, label_array, thisdict, S)
    print("The information gain array is "+str(information_gain))
    max_index = information_gain.index(max(information_gain))


split([[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[1,2],[0,2],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]],[0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1])



test = np.array([1,1,1,1])
test[test == 1] = 0
print(test)




print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
labels = ['b','b','a','c']
print(np.unique(labels))




