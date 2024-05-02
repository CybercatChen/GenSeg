import torch.nn.functional as F
import torch
import torch.nn as nn


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=3):
#         super().__init__()
#         self.zdim = zdim
#         self.conv1 = nn.Conv1d(input_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(512)
#
#         # Mapping to [c], cmean
#         self.fc1_m = nn.Linear(512, 256)
#         self.fc2_m = nn.Linear(256, 128)
#         self.fc3_m = nn.Linear(128, zdim)
#         self.fc_bn1_m = nn.BatchNorm1d(256)
#         self.fc_bn2_m = nn.BatchNorm1d(128)
#
#         # Mapping to [c], cmean
#         self.fc1_v = nn.Linear(512, 256)
#         self.fc2_v = nn.Linear(256, 128)
#         self.fc3_v = nn.Linear(128, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(256)
#         self.fc_bn2_v = nn.BatchNorm1d(128)
#
#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.bn4(self.conv4(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 512)
#
#         m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
#         m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
#         m = self.fc3_m(m)
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = self.fc3_v(v)
#
#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v


class PointNetEncoder(nn.Module):
    def __init__(self, config):
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
        g_feature = torch.max(x, 2, keepdim=True)[0]
        g_feature = g_feature.view(-1, 1, 256)
        g_feat_repeated = g_feature.repeat(1, points.shape[1], 1)
        feature = torch.concat([p_feature, g_feat_repeated], -1)
        return feature


class PointNetDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2048 * 3)

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        recon_points = x.view(-1, 2048, 3)

        return recon_points
