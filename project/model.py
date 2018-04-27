import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class ConvInputNN(nn.Module):
    def __init__(self, in_channel):
        super(ConvInputNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x


# f_phi
class FCNet2(nn.Module):
    def __init__(self, input_size, n):
        super(FCNet2, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n)

    def forward(self, g):
        g = self.fc1(g)
        g = F.relu(g)
        g = self.fc2(g)
        g = F.relu(g)
        g = F.dropout(g, p=0.5)
        g = self.fc3(g)
        return g


# g_theta
class FCNet1(nn.Module):
    def __init__(self, input_size):
        super(FCNet1, self).__init__()
        self.g1 = nn.Linear(input_size, 256)
        self.g2 = nn.Linear(256, 256)
        self.g3 = nn.Linear(256, 256)
        self.g4 = nn.Linear(256, 256)

    def forward(self, x):
        x = self.g1(x)
        x = F.relu(x)
        x = self.g2(x)
        x = F.relu(x)
        x = self.g3(x)
        x = F.relu(x)
        x = self.g4(x)
        x = F.relu(x)
        return x


class RN(nn.Module):
    def __init__(self, config):
        super(RN, self).__init__()

        self.config = config
        self.batch_size = int(self.config.batch_size)
        self.img_size = int(self.config.data_info[0])
        self.c_dim = int(self.config.data_info[2])
        self.q_dim = int(self.config.data_info[3])
        self.a_dim = int(self.config.data_info[4])
        self.conv_info = self.config.conv_info

        self.conv = ConvInputNN(self.c_dim)

        k = int(self.conv_info[-1])
        # g_theta
        self.g_theta = FCNet1((k+2)*2+self.q_dim)

        # coordinate tensor
        # output of convolutional network [B, K, d, d] k=24, d=8 (precompute)
        d = 8
        self.coord_tensor = torch.FloatTensor(self.batch_size, d*d, 2)
        if self.config.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)

        for i in range(d*d):
            coor_x, coor_y = self.concat_coor(i, d)
            self.coord_tensor[:, i, 0] = coor_x
            self.coord_tensor[:, i, 1] = coor_y

        # f_phi
        # after g_theta,
        self.f_phi = FCNet2(256, self.a_dim)

    def forward(self, img, q):
        img = img.float()
        q = q.float()
        x = self.conv(img)  # x = (B, 24, 5, 5)

        """g_theta"""
        d = x.size()[2]
        channel_size = x.size()[1]
        # flat to form objects
        batch_size = x.size()[0]
        x_flat = x.view(batch_size, channel_size, d*d).permute(0, 2, 1)

        # concatenate coordinates
        if batch_size == self.batch_size:
            coord_tensor = self.coord_tensor
        else:
            d = 8
            coord_tensor = torch.FloatTensor(batch_size, d*d, 2)
            if self.config.cuda:
                coord_tensor = coord_tensor.cuda()
            coord_tensor = Variable(coord_tensor)

            for i in range(d*d):
                coor_x, coor_y = self.concat_coor(i, d)
                coord_tensor[:, i, 0] = coor_x
                coord_tensor[:, i, 1] = coor_y

        x_flat = torch.cat([x_flat, coord_tensor], dim=-1)  # (B, 64, 26)

        # add question vectors
        # q = [B, 11]
        q = torch.unsqueeze(q, 1)  # (B, 1, 11)
        q = q.repeat(1, d*d, 1)  # (B, 64, 11)
        q = torch.unsqueeze(q, 2)  # (B, 64, 1, 11)

        # create object pairs
        x_i = torch.unsqueeze(x_flat, 1)  # (B, 1, 64, 26)
        x_i = x_i.repeat(1, d*d, 1, 1)  # (B, 64, 64, 26)
        x_j = torch.unsqueeze(x_flat, 2)  # (B, 64, 1, 26)
        x_j = torch.cat([x_j, q], dim=-1)  # (B, 64, 1, 26+11)
        x_j = x_j.repeat(1, 1, d*d, 1)  # (B, 64, 64, 26+11)

        # generate pairs
        pairs = torch.cat([x_i, x_j], 3)  # (B, 64, 64, 26+26+11)

        # flat pairs
        pairs = pairs.view(batch_size*d*d*d*d, 26+26+11)

        # go through the g_theta
        x = self.g_theta(pairs)

        # element-wise sum
        x = x.view(batch_size, d*d*d*d, 256)
        x_sum = x.sum(1).squeeze()  # (B, 256)

        """f_phi"""
        x_f = self.f_phi.forward(x_sum)  # (B, n_class)
        if len(x_f.size()) > 2:
            x_f = F.log_softmax(x_f, dim=1)
        else:
            x_f = F.log_softmax(x_f, dim=0)
        return x_f

    def concat_coor(self, i, d):
        return [(i/d), i % d]




class BaseLine(nn.Module):
    pass