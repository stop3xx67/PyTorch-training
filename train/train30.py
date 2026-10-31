import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# здесь объявляйте класс TriagModel
class TriagModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(3, 1, bias=False)


    def forward(self, x):
        x = self.layer1(x)
        return x

np.random.seed(1)
torch.manual_seed(1)

total = 100
x_train = torch.randint(1, 10, (total, 3), dtype=torch.float32)
y_train = x_train.sum(dim=1) / 3

# здесь создавайте модель (model)
model = TriagModel()
# переведите модель в режим обучения
model.train()

lr = 0.01 # шаг обучения
N = 1000 # число итераций SGD

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr) # здесь сформируйте оптимизатор Adam с параметрами модели и шагом обучения lr
loss_func = torch.nn.MSELoss() # здесь создайте функцию потерь с помощью класса nn.MSELoss

for _ in range(N):
    k = np.random.randint(0, total)
    # пропустите через модель k-й образ выборки x_train и вычислите прогноз predict
    predict = model(x_train[k])
    loss = loss_func(predict, x_train[k]) # вычислите значение функции потерь и сохраните результат в переменной loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()