import numpy as np

x = list(map(int, input().split())) # чтение вектора [x1, x2, x3] из входного потока

# здесь продолжайте программу
W = np.array([[0.3, 0.2, 0.5], [0.5, -0.1, -0.2]])
X = np.array(x)
b = np.array([-1.5, 2])

z = np.dot(W, X) + b

W2 = np.array([-1.5, 0.5])
b2 = -1
X2 = np.array([np.sign(z[0]), np.sign(z[1])])

y = np.dot(W2, X2) + b2

print(f"{y.item():.1f}") # вывод выходного значения y с точностью до десятых