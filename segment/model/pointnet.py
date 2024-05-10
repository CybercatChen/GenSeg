import torch.nn.functional as F
import torch
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
        # if args..part_decoder is True:
        #     self.data_point = args..data_point // args..part_num

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(256, self.data_point * 3)

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        recon_points = x.view(-1, self.data_point, 3)
        return recon_points


class PartDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.num_parts = args.part_num
        self.decoders = nn.ModuleList([PointNetDecoder(args) for _ in range(self.num_parts)])

    def forward(self, part_features):
        B, M, E = part_features.shape
        assert M == self.num_parts

        recon_parts = []
        for i in range(self.num_parts):
            part_feature = part_features[:, i, :].unsqueeze(1)  # B 1 E
            reconstructed_part = self.decoders[i](part_feature)  # B N 3
            recon_parts.append(reconstructed_part)
        # reconstructed_object = torch.cat(reconstructed_parts, dim=1)  # B (M*N) 3

        return recon_parts
