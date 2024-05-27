import torch.nn.functional as F
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, points):
        x = points.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        p_feature = x.transpose(1, 2)
        return p_feature


class PointNetDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.latent_dim = 256
        self.data_point = args.data_point
        self.fc1 = nn.Linear(self.latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.data_point * 3 // args.part_num)

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        recon_points = x.view(-1, self.data_point, 3)
        return recon_points
