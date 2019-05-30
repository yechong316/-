import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
x = torch.tensor([1, 2]).view(1, -1).float()

class Net(nn.Module):

    def __init__(self):

        # super函数类的意思是将继承的父类里面的成员全部继承到本类中
        super(Net, self).__init__()

        self.w1 = nn.Linear(2, 10)
        self.w2 = nn.Linear(10, 10)
        self.w3 = nn.Linear(10, 2)

    def forward(self, input):

        net = F.relu(self.w1(x))
        net = F.relu(self.w2(net))
        out = self.w3(net)
        return out
# print(out)

criterion = nn.MSELoss()
y = torch.tensor([2, 1]).view(1, -1).float()

net = Net()
params = list(net.parameters())
out = net(x)

print('before update weight:', params[0])  # conv1's .weight
# print('out:', out)
# print('损失：', cost)

# 优化
optimizer = optim.SGD(net.parameters(), lr=10)
optimizer.zero_grad()
cost = criterion(out, y)
cost.backward()
optimizer.step()
print('after update weight:', params[0])