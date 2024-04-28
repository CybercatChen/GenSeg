import torch.nn as nn
import torch
from superpoint import *


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
            self.point_num = 2048 // config.model.superpoint_num
        assert len(self.hidden_dim) > 0
        assert len(self.input_dim) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_dim[-1], self.input_dim[0], 1),
            # nn.BatchNorm1d(self.hidden_dim[0]),
            nn.ReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.input_dim[0]
        for c in self.hidden_dim[::-1]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                # nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            ))
            hidden_in = c

        self.final_layer = nn.Sequential(
            nn.Conv1d(self.hidden_dim[0], self.output_dim, 1),
            # nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True)
        )

        self.point_layer1 = nn.Sequential(
            nn.Conv1d(1, self.point_num // 2, 1),
            nn.ReLU(inplace=True)
        )

        self.point_layer2 = nn.Sequential(
            nn.Conv1d(self.point_num // 2, self.point_num, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        """
        forward of decoder
        :param features: input features, B(batch) N(num) C(feature dim)
        :return: reconstructed points, B N 3(xyz output)
        """
        B, N, C = features.shape
        features = features.transpose(2, 1)  # B C N

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
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(self.num_parts)])

    def forward(self, part_features):
        B, M, E = part_features.shape
        assert M == self.num_parts

        reconstructed_parts = []
        for i in range(self.num_parts):
            part_feature = part_features[:, i, :].unsqueeze(1)  # B 1 E
            reconstructed_part = self.decoders[i](part_feature)  # B N 3
            reconstructed_parts.append(reconstructed_part)
        reconstructed_object = torch.cat(reconstructed_parts, dim=1)  # B (M*N) 3

        return reconstructed_object.transpose(2, 1)


class SegGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoder = PartDecoder(config)
        self.encoder = SuperPoint(config)

    def forward(self, points):
        p_feat, sp_atten, sp_feat, sp_param = self.encoder(points)
        recon_points = self.decoder(sp_feat)

        return recon_points, p_feat, sp_atten, sp_feat, sp_param
