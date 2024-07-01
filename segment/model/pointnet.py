import torch.nn.functional as F
import torch.nn as nn
import torch


class PointNetEncoder(nn.Module):
    def __init__(self):
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
        self.latent_dim = 128
        self.part_point = args.part_point

        self.fc1 = nn.Linear(self.latent_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, feature):
        x = F.relu(self.fc1(feature))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        recon_points = x.view(-1, self.part_point, 3)
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
        # B, M, E = part_features.shape
        B, M, N, E = part_features.shape

        recon_parts = []
        means = []
        logvars = []
        for i in range(self.num_parts):
            # part_feature = part_features[:, i, :].unsqueeze(1)  # B 1 E
            part_feature = part_features[:, i, :, :]  # B n E

            mean = self.mean_linears[i](part_feature)
            logvar = self.var_linears[i](part_feature)

            z = self.reparameterize(mean, logvar)

            reconstructed_part = self.decoders[i](z)  # B N 3
            recon_parts.append(reconstructed_part)
            means.append(mean)
            logvars.append(logvar)
        recon_all = torch.cat(recon_parts, dim=1)  # B (M*N) 3

        return recon_parts, recon_all, means, logvars

# if __name__ == '__main__':
#     import torch
#
#     model = Pointnet2(13)
#     xyz = torch.rand(32, 3, 2048)
#     x, l4_points = model(xyz)
#     print(l4_points.shape)
