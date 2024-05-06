import sys
from chamfer_distance import ChamferDistance as chamfer_dist

sys.path.append('..')
from script.PyTorchEMD import emd
from generate.model.pointnet import *
from generate.model.partae import *


class SegGen(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.superpoint_num = config.model.superpoint_num
        self.encoder = PointNetEncoder(config)
        self.attention_layer = nn.Sequential(
            nn.Conv1d(2048, self.superpoint_num, 1)
        )
        self.decoder = PointNetDecoder(config)

    def forward(self, points):
        p_feat = self.encoder(points)  # B N C
        sp_feat = self.attention_layer(p_feat)  # B 50(sp num) N

        recon_points = self.decoder(sp_feat)
        return recon_points, p_feat, sp_feat

    def get_loss(self, points, recon):
        # recon loss
        loss_emd = emd.earth_mover_distance(recon.transpose(2, 1), points.transpose(2, 1)).sum()
        loss_mse = F.mse_loss(points, recon, reduction='mean')

        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(points, recon)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss_emd, loss_mse, loss_cd
