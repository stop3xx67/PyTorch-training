import numpy as np

x = list(map(int, input().split())) # чтение вектора [x1, x2, x3] из входного потока

# здесь продолжайте программу
W = np.array([1.5, 1.5, 4])
X = np.array([x])
b = -5
y = np.dot(X, W) + b

print(f"{y.item():.2f}") # вывод выходного значения y с точностью до сотых