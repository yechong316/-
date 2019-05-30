# coding: utf-8

import torch
import torchvision
import torchvision.transforms as transforms


# torchvision的输出是[0,1]的PILImage图像，我们把它转换为归一化范围为[-1, 1]的张量。
# 

# In[2]:

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=r"D:\迅雷下载", train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=r"D:\迅雷下载", train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# 
# 3. 定义损失函数和优化器
# ----------------------------------------
# 
# 我们使用交叉熵作为损失函数，使用带动量的随机梯度下降。
# 
# 

# In[5]:

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# 
# 4. 训练网路
# --------------------------------
# 有趣的时刻开始了。
# 我们只需在数据迭代器上循环，将数据输入给网络，并优化。
# 
# 

# In[ ]:
for epoch in range(2):  # 多批次循环

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入
        inputs, labels = data

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 打印状态信息
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000批次打印一次
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# 
# 5. 在测试集上测试网络
# -------------------------------------
# 
# 我们在整个训练集上进行了2次训练，但是我们需要检查网络是否从数据集中学习到有用的东西。
# 通过预测神经网络输出的类别标签与实际情况标签进行对比来进行检测。
# 如果预测正确，我们把该样本添加到正确预测列表。
# 第一步，显示测试集中的图片并熟悉图片内容。
# 
# 

# In[6]:

# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # 显示图片
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#

# 让我们看看神经网络认为以上图片是什么。
# 
# 

# In[7]:

# outputs = net(images)


# 输出是10个标签的能量。
# 一个类别的能量越大，神经网络越认为它是这个类别。所以让我们得到最高能量的标签。
# 

# In[8]:
#
# _, predicted = torch.max(outputs, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))


# 结果看来不错。
# 
# 接下来让看看网络在整个测试集上的结果如何。
# 
# 

# In[9]:
#
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))


# 结果看起来不错，至少比随机选择要好，随机选择的正确率为10%。
# 似乎网络学习到了一些东西。
# 
# 
# 
# 在识别哪一个类的时候好，哪一个不好呢？
# 
# 

# In[10]:

# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(4):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
#
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))


# 下一步?
# 
# 我们如何在GPU上运行神经网络呢？
# 
# 在GPU上训练
# ----------------
# 把一个神经网络移动到GPU上训练就像把一个Tensor转换GPU上一样简单。并且这个操作会递归遍历有所模块，并将其参数和缓冲区转换为CUDA张量。
# 

# In[ ]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 确认我们的电脑支持CUDA，然后显示CUDA信息：

print(device)


# 本节的其余部分假定`device`是CUDA设备。
# 
# 然后这些方法将递归遍历所有模块并将模块的参数和缓冲区
# 转换成CUDA张量：
# 
# 
# ```python
# 
#     net.to(device)
# ```
# 
# 记住：inputs 和 targets 也要转换。
# 
# ```python
# 
#         inputs, labels = inputs.to(device), labels.to(device)
# ```
# 为什么我们没注意到GPU的速度提升很多？那是因为网络非常的小。
# 
# **实践:** 
# 尝试增加你的网络的宽度（第一个``nn.Conv2d``的第2个参数，第二个``nn.Conv2d``的第一个参数，它们需要是相同的数字），看看你得到了什么样的加速。
# 
# **实现的目标**:
# 
# - 深入了解了PyTorch的张量库和神经网络
# - 训练了一个小网络来分类图片
# 
# ***译者注：后面我们教程会训练一个真正的网络，使识别率达到90%以上。***
# 
# 多GPU训练
# -------------------------
# 如果你想使用所有的GPU得到更大的加速，
# 请查看[数据并行处理](5_data_parallel_tutorial.ipynb)。
# 
# 下一步？
# -------------------
# 
# 
# 
# 
# 
# -  :doc:`训练神经网络玩电子游戏 </intermediate/reinforcement_q_learning>`
# -  `在ImageNet上训练最好的ResNet`
# -  `使用对抗生成网络来训练一个人脸生成器`
# -  `使用LSTM网络训练一个字符级的语言模型`
# -  `更多示例`
# -  `更多教程`
# -  `在论坛上讨论PyTorch`
# -  `Slack上与其他用户讨论`
# 
# 
# 

# In[ ]:



