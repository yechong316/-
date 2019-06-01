import torch
from torch.nn import Linear, Module, MSELoss
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
x = np.random.rand(100)

y = x * 2 + 1 + np.random.randn(100)/100

# plt.plot(x,y)
# plt.show()

model = Linear(1, 1)
criterion = MSELoss()

opt = optim.SGD(model.parameters(), lr=0.01)
epochs = 1000

x_train = x.reshape((-1, 1)).astype('float32')
y_train = y.reshape((-1, 1)).astype('float32')

for i in range(epochs):

    input = torch.from_numpy(x_train)
    target = torch.from_numpy(y_train)
    output = model(input)

    # define initial grad = 0
    opt.zero_grad()

    # lost functional
    cost = criterion(target, output)

    # backward
    cost.backward()

    # 优化器优化朝默认方向优化
    opt.step()

    if i % 10 == 0:

        print('epoch:{}, loss:{}'.format(i, cost.data.item()))


[w, b] = model.parameters()
print('w, b', [w, b] )