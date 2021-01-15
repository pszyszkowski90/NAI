import numpy as np
from sklearn import svm


input_file = 'data_sonar.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :60], data[:, -1]

svc = svm.SVC(kernel='linear', C=1, gamma=100).fit(X, y)

XX = []
for i in range(60):
    tmp_min, tmp_max = X[:, i].min(), X[:, i].max()
    XX.append(np.arange(tmp_min, tmp_max, (tmp_max-tmp_min)/101)[:100])

XX = np.asarray(XX, dtype=np.float32)
Z = svc.predict(np.c_[XX.transpose()])
print(Z)
