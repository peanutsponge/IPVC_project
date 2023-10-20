import scipy as sp

mat = sp.io.loadmat('data/calibration_test.mat')

print(mat.keys())
print(mat['x'])