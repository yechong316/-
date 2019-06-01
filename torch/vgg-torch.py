import torch
import torch.nn as nn
import torch.nn.functional as F

class Vgg11(nn.Module):

    def __init__(self):
        super(Vgg11, self).__init__()

        # ####################################################
        # 卷积层1
        # ####################################################
        self.conv1_1 = nn.Conv2d(3, 64, 3)

        # ####################################################
        # 卷积层2
        # ####################################################
        self.conv2_1 = nn.Conv2d(64, 128, 3)

        # ####################################################
        # 卷积层3
        # ####################################################
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)

        # ####################################################
        # 卷积层4
        # ####################################################
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)

        # ####################################################
        # 卷积层5
        # ####################################################
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.conv5_3 = nn.Conv2d(512, 512, 3)

        # ####################################################
        # 全连接层1
        # ####################################################
        self.fc_1 = nn.Linear(7*7*512, 4096)
        self.fc_2 = nn.Linear(4096, 4096)
        self.fc_3 = nn.Linear(4096, 10)

    def forward(self, x):
        # ####################################################
        # 卷积层1
        # ####################################################
        net = F.max_pool2d(F.relu(self.conv1_1(x)), (2, 2), padding=2)

        # ####################################################
        # 卷积层2
        # ####################################################
        net = F.max_pool2d(F.relu(self.conv2_1(net)), (2, 2))

        # ####################################################
        # 卷积层3
        # ####################################################
        net = F.relu(self.conv3_1(net))
        net = F.relu(self.conv3_2(net))
        net = F.relu(self.conv3_3(net))
        net = F.max_pool2d(net, (2, 2))

        # ####################################################
        # 卷积层4
        # ####################################################
        net = F.relu(self.conv4_1(net))
        net = F.relu(self.conv4_2(net))
        net = F.relu(self.conv4_3(net))
        net = F.max_pool2d(net, (2, 2))

        # ####################################################
        # 卷积层5
        # ####################################################
        net = F.relu(self.conv5_1(net))
        net = F.relu(self.conv5_2(net))
        net = F.relu(self.conv5_3(net))
        net = F.max_pool2d(net, (2, 2))

        # ####################################################
        # 全连接层1
        # ####################################################
        net = net.view(-1, self.num_flat_features(net))
        net = F.relu(self.fc_1(net))
        net = F.relu(self.fc_2(net))
        net = self.fc_3(net)

        return net
    # x的第二个开始的维度相乘
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == '__main__':

    vgg = Vgg11()
    input = torch.randn(1, 3, 224,224)
    out = vgg(input)
    print(out)