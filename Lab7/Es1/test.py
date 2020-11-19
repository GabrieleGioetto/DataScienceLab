import numpy as np

print([[0] * 4])
print([[0, 1] * 4])

data = np.array([[[0, 1]] * 4] * 5)
print(data)
print(data.shape)

print(data[0, 0, :])
