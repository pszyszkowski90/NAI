import numpy as np
from sklearn import svm
from numpy import random

input_file = 'data_happy.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, 1:], data[:, 0]

svc = svm.SVC(kernel='linear', C=1, gamma=100).fit(X, y)

XX = random.randint(1,6, size=(100, 6))
XX = np.asarray(XX)

Z = svc.predict(np.c_[XX])

print(Z)
