import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# hyper parameters
epochs = 1
batch_size = 64
time_step = 28 # 图片的每行信息
input_size = 28 # 图片的行数
lr = 0.01
dowdload_mnist = True

train_data = dsets.MNIST(root="D:\预训练模型\mnist",
                         train=True,
                         transform=transforms.ToTensor(),
                         download=dowdload_mnist)

test_data = dsets.MNIST(root="D:\预训练模型\mnist",
                         train=False,
                         transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_data.numpy()[:2000]

class RNN(nn.Module):

    def __init__(self):

        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
        input_size=input_size,
        hidden_size=32,
        num_layers=2,
        bias=False,
        batch_first=True,
        dropout=0.8,
        bidirectional=False
        )

        self.out = nn.Linear(32, 10)

    def forward(self, x):

        r_out, (_, __) = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()

opt = torch.optim.Adam(rnn.parameters(), 0.01)
loss_func = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch, (train_x, train_y) in enumerate(train_loader):

        inputs = train_x.view(-1, 28, 28)
        # print(train_x.shape)
        out = rnn(inputs)

        loss = loss_func(out, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if batch % 50 == 0:

            test_output = rnn(test_x)
            y_pred = torch.max(test_output, 1)[1].data.numpy()
            acc = float(np.array((y_pred == test_y)).astype(int).sum())/ float(test_y.size)
    # n  accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)

            print('batch: {} | loss: {} | acc: {}'.format(batch, loss, acc))
