# coding: utf-8

# In[1]:

# get_ipython().magic('matplotlib inline')


# 
# Neural Networks
# ===============
# 
# 使用torch.nn包来构建神经网络。
# 
# 上一讲已经讲过了``autograd``，``nn``包依赖``autograd``包来定义模型并求导。
# 一个``nn.Module``包含各个层和一个``forward(input)``方法，该方法返回``output``。
# 
# 
# 
# 例如：
# 
# ![](https://pytorch.org/tutorials/_images/mnist.png)
# 
# 它是一个简单的前馈神经网络，它接受一个输入，然后一层接着一层地传递，最后输出计算的结果。
# 
# 神经网络的典型训练过程如下：
# 
# 1. 定义包含一些可学习的参数(或者叫权重)神经网络模型； 
# 2. 在数据集上迭代； 
# 3. 通过神经网络处理输入； 
# 4. 计算损失(输出结果和正确值的差值大小)；
# 5. 将梯度反向传播会网络的参数； 
# 6. 更新网络的参数，主要使用如下简单的更新原则： 
# ``weight = weight - learning_rate * gradient``
# 
#   
# 
# 定义网络
# ------------------
# 
# 开始定义一个网络：
# 
# 

# In[2]:

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):

        # super函数类的意思是将继承的父类里面的成员全部继承到本类中
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel， nn.Conv2d(输入通道，输出通道，卷积核尺寸)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b， nn.Linear(输入特征, 输出特征)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        # 将x拉成 【-1， 其他特征的乘积】
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # x的第二个开始的维度相乘
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

params = list(net.parameters())
# print(len(params))
print('before update weight:', params[0][0][0][0])  # conv1's .weight


# 测试随机输入32×32。
# 注：这个网络（LeNet）期望的输入大小是32×32，如果使用MNIST数据集来训练这个网络，请把图片大小重新调整到32×32。
# 
# 

# In[5]:

input = torch.randn(10, 1, 32, 32)

# r"""Sets gradients of all model parameters to zero."""
net.zero_grad()

output = net(input)
target = torch.randn((1, 10))
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=10)
optimizer.zero_grad()
cost = criterion(output, target)
cost.backward()
optimizer.step()
print('after update weight:', params[0][0][0][0])