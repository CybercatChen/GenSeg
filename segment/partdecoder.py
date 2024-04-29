import torch.nn as nn
import torch
from superpoint import *
import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m


class Decoder(nn.Module):
    def __init__(self, config):
        """
        init of decoder
        :param input_dim:
        :param output_dim:
        :param hidden_dim: list, the output dim of hidden layers
        :param use_global: bool, True: add global feature to point-wise feature and then do one more conv
        """
        super().__init__()
        if config is None:
            self.input_dim = [256, 512]
            self.output_dim = 3
            self.use_global = True
            self.hidden_dim = [1024, 256, 64]
        else:
            self.input_dim = config.model.encoder.output_dim
            self.output_dim = 3
            self.use_global = config.model.encoder.use_global
            self.hidden_dim = config.model.encoder.hidden_dim
            self.point_num = 2048
        assert len(self.hidden_dim) > 0
        assert len(self.input_dim) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_dim[-1], self.input_dim[0], 1),
            nn.BatchNorm1d(self.hidden_dim[0]),
            # nn.ReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.input_dim[0]
        for c in self.hidden_dim[::-1]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                # nn.ReLU(inplace=True)
            ))
            hidden_in = c

        self.final_layer = nn.Sequential(
            nn.Conv1d(self.hidden_dim[0], self.output_dim, 1),
            nn.BatchNorm1d(self.output_dim),
            # nn.ReLU(inplace=True)
        )

        self.point_layer1 = nn.Sequential(
            nn.Conv1d(4, self.point_num // 2, 1),
            # nn.ReLU(inplace=True)
        )

        self.point_layer2 = nn.Sequential(
            nn.Conv1d(self.point_num // 2, self.point_num, 1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        forward of decoder
        :param features: input features, B(batch) N(num) C(feature dim)
        :return: reconstructed points, B N 3(xyz output)
        """
        x = self.input_layer[0](features)  # B 512 N
        for layer in self.hidden_layers:
            x = layer(x)  # B _ N
        x = self.final_layer(x).transpose(2, 1)  # B N 3
        x = self.point_layer1(x)
        x = self.point_layer2(x)
        return x


class PartDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_parts = config.model.superpoint_num
        self.decoders = Decoder(config)

    def forward(self, part_features):
        recon_part = self.decoders(part_features)  # B N 3
        return recon_part


class SegGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = SuperPoint(config)
        self.decoder = PartDecoder(config)

    def forward(self, points):
        p_feat, sp_feat = self.encoder(points)
        recon_points = self.decoder(sp_feat.transpose(2, 1))

        return recon_points, p_feat, sp_feat
