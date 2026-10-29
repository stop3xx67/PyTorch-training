import numpy as np

x = list(map(int, input().split())) # чтение вектора [x1, x2] из входного потока

# здесь продолжайте программу
W = np.array([1, 1])
X = np.array([x[0] ** 2, x[1] ** 2])
b = -5

y = np.dot(W, X.T) + b

print(round(float(y), 3))