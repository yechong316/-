import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
        self.fc = nn.Linear(675, 10)

    def forward(self, input):

        net = F.max_pool2d(F.relu(self.conv(input)), (2, 2))

        net = net.view(net.size()[0], -1)

        net = self.fc(net)
        return net


net = Net()

# print('net.named_parameters：', net.named_parameters)
# print('net.parameters：', net.parameters)

input = torch.randn(1, 1, 32, 32)
out = net(input)
# print(out)

# 定义损失函数
y = torch.randn(10).view(1, 10)
criterion = nn.MSELoss()
cost = criterion(out, y)

# 优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
cost.backward()

# 更新参数
params = net.parameters()

for i in params:
    print(i)
optimizer.step()

print('~'*50)
params = net.parameters()
for i in params:
    print( i)