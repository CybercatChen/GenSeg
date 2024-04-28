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
            # nn.BatchNorm1d(self.hidden_dim[0]),
            nn.ReLU(inplace=True)
        )

        self.hidden_layers = nn.Sequential()
        hidden_in = self.hidden_dim[0]
        for c in self.hidden_dim[1:]:
            self.hidden_layers.add_module(str(len(self.hidden_layers)), nn.Sequential(
                nn.Conv1d(hidden_in, c, 1),
                # nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
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
                # nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
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
            nn.Conv1d(self.encoder_output_dim[-1], self.superpoint_num, 1)
        )

        self.param_mlp = nn.Sequential()
        mlp_in = self.encoder_output_dim[-1]
        for c in self.mlp_hidden_dim:
            self.param_mlp.add_module(str(len(self.param_mlp)), nn.Sequential(
                nn.Conv1d(mlp_in, c, 1),
                # nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)
            ))
            mlp_in = c

        self.part_para = nn.Parameter(torch.randint(low=0, high=config.model.superpoint_num+1, size=(2048, 1)))

    def forward(self, points):
        """
        forward of SuperPoint
        :param points: input data, B(batch) N(num) 3(xyz input)
        :return: p_feat: point-wise features
                 sp_atten: superpoint attention map of points
                 sp_feat: superpoint-wise features
                 sp_param: superpoint-wise parameters
        """
        # get point-wise features O
        p_feat = self.encoder(points)  # B N C

        # get superpoint attention map A
        # sp_atten = self.attention_layer(p_feat.transpose(2, 1))  # B 50(sp num) N
        # TODO
        sp_atten = self.linear()
        sp_atten = F.softmax(sp_atten, dim=1)  # B 50(sp num) N, softmax on superpoint dim: dim-1

        # get superpoint features S
        sp_feat = torch.bmm(F.normalize(sp_atten, p=1, dim=2),
                            p_feat)  # B 50(sp num) C, l1-norm on attention map last dim: dim-2

        return p_feat, sp_atten, sp_feat

    def get_loss(self, points, p_feat, sp_atten, sp_feat, recon):
        """
        calculate loss
        :param points: xyz coordinates, B N 3
        :param p_feat: point-wise features, B N C
        :param sp_atten: superpoint attention map, B M N
        :param sp_feat: superpoint-wise features, B M C
        :param sp_param: superpoint parameters, B M 14
        :return:
        """
        B, N, C = p_feat.shape
        _, M, _ = sp_atten.shape

        # ss loss
        sp_feat_un = sp_feat.unsqueeze(2)  # B M 1 C
        p_feat_un = p_feat.unsqueeze(1)  # B 1 N C
        feat_dist = sp_feat_un - p_feat_un  # B M N C
        feat_dist = torch.norm(feat_dist, dim=-1)  # B M N
        feat_dist = feat_dist * sp_atten  # B M N
        # loss_ss = torch.sum(feat_dist) / (M * N)
        loss_ss = torch.sum(feat_dist)  # paper

        # loc loss
        centriods = torch.bmm(F.normalize(sp_atten, p=1, dim=2), points)  # B M 3
        centriods = centriods.unsqueeze(2)  # B M 1 3
        points_un = points.unsqueeze(1)  # B 1 N 3
        coord_dist = centriods - points_un  # B M N 3
        coord_dist = torch.norm(coord_dist, dim=-1)  # B M N
        coord_dist = coord_dist * sp_atten  # B M N
        # loss_loc = torch.sum(coord_dist) / (M * N)
        loss_loc = torch.sum(coord_dist)  # paper

        # sp balance loss
        # print(sp_atten.max(dim=-1)[0] - sp_atten.min(-1)[0])
        sp_atten_per_sp = torch.sum(sp_atten, dim=-1)  # B M
        sp_atten_sum = torch.sum(sp_atten_per_sp, dim=-1, keepdim=True) / M  # B 1
        loss_sp_balance = torch.sum((sp_atten_per_sp - sp_atten_sum) ** 2) / M

        # recon loss
        loss_emd = emd.earth_mover_distance(recon, points.transpose(2, 1)).sum()
        label = torch.argmax(sp_atten, axis=-2)
        return loss_ss, loss_loc, loss_sp_balance, loss_emd, label


if __name__ == "__main__":
    from torchsummary import summary

    net = SuperPoint().cuda()
    points = torch.rand(1, 20, 3).cuda()  # 1 batch, 2 points, 3 coords
    p_feat, sp_atten, sp_feat, sp_param = net(points)
    net.get_loss(points, p_feat, sp_atten, sp_feat, sp_param)
