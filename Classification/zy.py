import numpy as np
a1 = np.asarray([-3, 2]) *5
a2 = np.asarray([-1, 1]) *5
a3 = np.asarray([5, 2])*3
a4 = np.asarray([2, 2])*4
a5 = np.asarray([1, -2])*3

print(a1-a2+a3-a4+a5)




x = np.array([[1,3],[-1,2]])
y = np.linalg.inv(x)
print(y)