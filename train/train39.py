import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


# здесь продолжайте программу
class MyData(data.Dataset):
    def __init__(self):
        self.data = torch.arange(-6, 6, 0.1)
        self.targets = (
            0.5 * self.data
            + torch.sin(2 * self.data)
            - 0.1 * torch.exp(self.data / 2)
            - 12.5
        )
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.length


class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Linear(3, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([x, x**2, x**3], dim=1)
        x = self.layer1(x)
        return x.squeeze(1)

model = Model()
model.train()

dataset = MyData()
train_data = data.DataLoader(
    dataset=dataset,
    batch_size=10,
    shuffle=True
    )

loss_fn = nn.MSELoss()

optimizer = optim.RMSprop(params=model.parameters(), lr=0.1)

for epoch in range(20):
    for X, y in train_data:
        optimizer.zero_grad()
        prediction = model(X)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

model.eval()
predictions = model(dataset.data)
Q = loss_fn(predictions, dataset.targets).item()