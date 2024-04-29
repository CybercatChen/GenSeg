import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('..')
from segment.utils.superquadrics import *
from utils.PyTorchEMD import emd


class Encoder(nn.Module):
    def __init__(self, config):
        """
        init of encoder
        :param input_dim:
        :param output_dim:
        :param hidden_dim: list, the output dim of hidden layers
        :param use_global: bool, True: add global feature to point-wise feature and then do one more conv
        """
        super().__init__()
        if config is None:
            self.input_dim = 3
            self.output_dim = [512, 256]
            self.use_global = True
            self.hidden_dim = [64, 256, 1024]
        else:
            self.input_dim = config.model.encoder.input_dim
            self.output_dim = config.model.encoder.output_dim
            self.use_global = config.model.encoder.use_global
            self.hidden_dim = config.model.encoder.hidden_dim

        assert len(self.hidden_dim) > 0
        assert len(self.output_dim) > 0

        self.input_layer = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim[0], 1),
            nn.BatchNorm1d(self.hidden_dim[0]),
            nn.LeakyReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.hidden_dim[0]
        for c in self.hidden_dim[1:]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                nn.LeakyReLU(inplace=True)
            ))
            hidden_in = c

        self.final_layer = nn.Sequential()
        if self.use_global:
            hidden_in = self.hidden_dim[-1] + self.hidden_dim[0]
        else:
            hidden_in = self.hidden_dim[-1]
        for c in self.output_dim:
            self.final_layer.add_module(str(len(self.final_layer)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                nn.BatchNorm1d(c),
                nn.LeakyReLU(inplace=True)
            ))
            hidden_in = c

    def forward(self, points):
        """
        forward of encoder
        :param points: input data, B(batch) N(num) 3(xyz input)
        :return: point-wise feature, B N C
        """
        B, N, C = points.shape
        points = points.transpose(2, 1)  # B 3 N

        x = self.input_layer(points)  # B 64(or other) N
        px = x  # B 64(or other) N

        for layer in self.hidden_layers:
            x = layer(x)  # B _ N

        if self.use_global:
            x_global = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, N)  # B 1024(or other) N
            x = torch.cat([x_global, px], dim=1)
        x = self.final_layer(x).transpose(2, 1)

        return x


class SuperPoint(nn.Module):
    def __init__(self, config=None):
        """
        init SuperPoint model
        :param config: config
        """
        super().__init__()
        self.part_para = None
        if config is None:
            self.encoder_output_dim = [512, 256]
            self.superpoint_num = 15
            self.param_num = 14
            self.mlp_hidden_dim = [256, 64]
        else:
            self.encoder_output_dim = config.model.encoder.output_dim
            self.superpoint_num = config.model.superpoint_num
            self.param_num = config.model.param_num
            self.mlp_hidden_dim = config.model.mlp_hidden_dim

        self.encoder = Encoder(config)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(2048, self.superpoint_num, 1)
        )

        self.param_mlp = nn.Sequential()
        mlp_in = self.encoder_output_dim[-1]
        for c in self.mlp_hidden_dim:
            self.param_mlp.add_module(str(len(self.param_mlp)), nn.Sequential(
                nn.Conv1d(mlp_in, c, 1),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            ))
            mlp_in = c

        self.assign_linear = nn.Linear(1, self.superpoint_num)

    def forward(self, points):

        p_feat = self.encoder(points)  # B N C
        B, N, C = points.shape
        sp_feat = self.attention_layer(p_feat)  # B 50(sp num) N

        return p_feat, sp_feat

    def get_loss(self, points, recon):
        # recon loss
        loss_emd = emd.earth_mover_distance(recon, points.transpose(2, 1)).sum()
        return loss_emd
