import torch.nn.functional as F
import torch.nn as nn
import torch
from pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


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
        if args.part_decoder is True:
            self.data_point = 400

        self.fc1 = nn.Linear(self.latent_dim, 256)
        self.fc2 = nn.Linear(256, 256)
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

        self.mean_linears = nn.ModuleList([nn.Linear(args.latent_dim, args.latent_dim) for _ in range(self.num_parts)])
        self.var_linears = nn.ModuleList([nn.Linear(args.latent_dim, args.latent_dim) for _ in range(self.num_parts)])

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def forward(self, part_features):
        B, M, E = part_features.shape
        assert M == self.num_parts

        recon_parts = []
        means = []
        logvars = []
        for i in range(self.num_parts):
            part_feature = part_features[:, i, :].unsqueeze(1)  # B 1 E

            mean = self.mean_linears[i](part_feature)
            logvar = self.var_linears[i](part_feature)

            z = self.reparameterize(mean, logvar)

            reconstructed_part = self.decoders[i](z)  # B N 3
            recon_parts.append(reconstructed_part)
            means.append(mean)
            logvars.append(logvar)
        recon_all = torch.cat(recon_parts, dim=1)  # B (M*N) 3

        return recon_parts, recon_all, means, logvars


class Pointnet2(nn.Module):
    def __init__(self, num_classes):
        super(Pointnet2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 3, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])
        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


if __name__ == '__main__':
    import torch

    model = Pointnet2(13)
    xyz = torch.rand(32, 3, 2048)
    x, l4_points = model(xyz)
    print(l4_points.shape)
