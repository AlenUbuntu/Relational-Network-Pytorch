import torch.nn as nn
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
import numpy as np


class ConvInputNN(nn.Module):
    def __init__(self, conv_info, data_info):
        super(ConvInputNN, self).__init__()
        self.conv1 = nn.Conv2d(int(data_info[2]), int(conv_info[0]), int(data_info[2]), stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(int(conv_info[0]))
        self.conv2 = nn.Conv2d(int(conv_info[0]), int(conv_info[1]), int(data_info[2]), stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(int(conv_info[1]))
        self.conv3 = nn.Conv2d(int(conv_info[1]), int(conv_info[2]), int(data_info[2]), stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(int(conv_info[2]))
        self.conv4 = nn.Conv2d(int(conv_info[2]), int(conv_info[3]), int(data_info[2]), stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(int(conv_info[3]))

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


# f_phi
class FCNet2(nn.Module):
    def __init__(self, input_size, n):
        super(FCNet2, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, n)

    def forward(self, g):
        g = self.fc1(g)
        g = F.relu(g)
        g = self.fc2(g)
        g = F.relu(g)
        g = self.dp1(g)
        g = self.fc3(g)
        return g


class Model(nn.Module):
    def __init__(self, config, debug_information=False):
        super(Model, self).__init__()
        self.debug = debug_information
        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info
        self.all_pred = None
        self.writer = SummaryWriter()

        n = self.a_dim  # dimension of answer vector per question per shape in each image
        # number of feature mappings
        k = self.conv_info[-1]
        # CNN conv4 [B, d, d, k]
        self.convNet = ConvInputNN(self.conv_info, self.config.data_info)
        # g_theta layer
        self.g_theta = FCNet1((int(k)+2)*2+int(self.q_dim))
        # f_phi layer
        self.f_phi = FCNet2(256, int(n))

    def forward(self, img, q):
        # compute output of convNet
        img_out = self.convNet.forward(img)
        # k feature map of d*d size
        d = img_out.size()[2]
        all_g = []
        for i in range(d*d):
            o_i = img_out[:, :, int(i / d), int(i % d)]
            o_i = self.concat_coor(o_i, i, d)
            for j in range(d*d):
                o_j = img_out[:, :, int(j / d), int(j % d)]
                o_j = self.concat_coor(o_j, j, d)
                g_i_j = self.g_theta.forward(torch.cat([o_i, o_j, q], dim=1))
                all_g.append(g_i_j)

        all_g = torch.stack(all_g, dim=0)
        all_g = torch.mean(all_g, 0)

        logits = self.f_phi.forward(all_g)
        self.all_pred = torch.nn.functional.softmax(logits, dim=-1)

        return logits

    def concat_coor(self, o, i, d):
        coor = torch.unsqueeze(torch.from_numpy(np.array([float(int(i / d)) / d, (i % d) / d])).float(),
                               dim=0).repeat(self.batch_size, 1)
        if self.config.cuda:
            o = torch.cat((o, torch.autograd.Variable(coor.cuda().float())), dim=1)
        else:
            o = torch.cat((o, torch.autograd.Variable(coor.float())), dim=1)
        return o

    def build_loss(self, img, q, a, n_iter):
        img = img.float()
        q = q.float()

        # find 1D representation of labels
        a = a.long()
        _, labels = torch.max(a, dim=1)
        logits = self.forward(img, q)

        # compute cross-entropy loss
        loss = nn.CrossEntropyLoss()
        output = loss(logits, labels)

        # classification accuracy
        correct_prediction = torch.max(logits, 1)[1].eq(torch.max(a, 1)[1])
        accuracy = torch.mean(correct_prediction.float())

        # add summaries
        self.writer.add_scalar('loss/accuracy', accuracy, n_iter)
        self.writer.add_scalar('loss/cross_entropy', output, n_iter)
        return torch.mean(output), accuracy







