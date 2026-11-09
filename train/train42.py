import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


# здесь продолжайте программу
class MyData(data.Dataset):
    def __init__(self):
        self.data = _global_var_data_x
        self.target = _global_var_target
        self.length= len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return self.length


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(8*8, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x

model = MyModel()
model.train()

dataset = MyData()
train_data = data.DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True
    )

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

for epoch in range(10):
    for X, y in train_data:
        prediction = model(X)
        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
predictions = torch.argmax(model(dataset.data), dim=1)
Q = torch.mean((predictions == dataset.target).float()).item()
