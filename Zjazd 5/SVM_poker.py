import numpy as np
from sklearn import svm
from numpy import random

input_file = 'data_decision_trees2.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :9], data[:, -1]

svc = svm.SVC(kernel='linear', C=1, gamma=100).fit(X, y)

XX = []
for i in range(9):
    if i % 2 == 0:
        min, max = 1, 4
    else:
        min, max = 1, 13
    XX.append(random.randint(min, max, size=100))

XX = np.asarray(XX, dtype=np.float32)
Z = svc.predict(np.c_[XX.transpose()])
print(Z)