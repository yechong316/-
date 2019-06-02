import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


data=np.loadtxt(r"C:\Users\dell\Downloads\german.data-numeric")

mean = np.mean(data, axis=0)
std = np.std(data, 0)
print(std.shape)


# print(data[0])
np.random.shuffle(data)
# print('*'*50)
# print(data[0])
n,l=data.shape
print(data.shape)
# x_train, x_label = data[:800, :data.shape[1] - 1], data[:800, data.shape[1] - 1] - 1
# x_test, y_test = data[800:, :data.shape[1] - 1], data[800:, -1] - 1
train_data=data[:900,:l-1]
train_lab=data[:900,l-1]-1
test_data=data[900:,:l-1]
test_lab=data[900:,l-1]-1
# train_data=data[:900,:l-1]
# train_lab=data[:900,l-1]-1

# x_label = torch.from_numpy(x_label).float().view(-1, 1)

# print(x_train.shape)
# print(x_label.shape)

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(24, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 2)

    def forward(self, x):

        net = F.relu(self.layer1(x))
        net = F.relu(self.layer2(net))
        net = torch.sigmoid(self.layer3(net))

        return net

net = Net()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters())

epochs = 1000



for i in range(epochs):
    # pass
    net.train()

    x=torch.from_numpy(train_data).float()
    y=torch.from_numpy(train_lab).long()
    y_hat=net(x)

    print(y_hat.shape)
    loss=criterion(y_hat,y) # 计算损失

    opt.zero_grad()
    loss.backward()
    opt.step()

    if (i + 1) % 10 == 0:

        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)

        accu = net(test_in)

        print('epochs:{}, lost:{}, acc:{}'.format(i + 1, loss, accu[0][0]))

