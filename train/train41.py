import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class WineDataset(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x # тензор размерностью (178, 13), тип float32
        self.target = _global_var_target # тензор размерностью (178, ), тип int64 (long)

        self.length = len(self.data) # размер выборки
        self.categories = ['class_0', 'class_1', 'class_2'] # названия классов

    def __getitem__(self, item):
        return self.data[item], self.target[item]
        # возврат образа по индексу item в виде кортежа: (данные, целевое значение)

    def __len__(self):
        return self.length
        # возврат размера выборки

# здесь продолжайте программу
class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(13, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

model = MyModel()

dataset = WineDataset()
train_data = data.DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True
    )

optimizer = optim.Adam(params=model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
model.train()

for epoch in range(20):
    for X, y in train_data:
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
pr = torch.argmax(model(dataset.data), dim = 1)
Q = torch.mean((pr == dataset.target).float()).item()