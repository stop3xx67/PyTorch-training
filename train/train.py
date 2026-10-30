import numpy as np

x = list(map(int, input().split())) # чтение вектора [x1, x2] из входного потока

# здесь продолжайте программу
W = np.array([0.5, 0.5, -1])
X = np.array([x[0], x[1], x[0] * x[1]])
b = 2.5

Z = np.dot(W, X.T) + b
y = Z

print(f"{y.item():.1f}") # вывод выходного значения y с точностью до десятых