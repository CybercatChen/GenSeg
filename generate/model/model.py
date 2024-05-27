from chamfer_distance import ChamferDistance as chamfer_dist
import torch.nn.init as init
import sys
sys.path.append('..')
from script.PyTorchEMD import emd
from generate.model.pointnet import *
from generate.model.partae import *


class SegGen(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.part_num = args.part_num
        self.encoder = PointNetEncoder(args)
        self.decoder = PointNetDecoder(args)
        self.sp_atten = nn.Parameter(torch.rand(args.data_point, args.part_num))
        init.xavier_uniform_(self.sp_atten)

    def forward(self, args, points):
        p_feat = self.encoder(points)
        sp_atten = F.softmax(self.sp_atten, dim=1)
        part_feat = torch.matmul(p_feat.transpose(1, 2), sp_atten).transpose(1, 2)
        recon_points = self.decoder(part_feat)
        return recon_points, p_feat, sp_atten

    def get_loss(self, points, recon, pre_label, gt_label):
        # recon loss
        loss_emd = emd.earth_mover_distance(recon.transpose(2, 1), points.transpose(2, 1)).sum()

        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(points, recon)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))

        ce = nn.CrossEntropyLoss()
        gt_label = (gt_label-1).clone().long()
        rep_pre_label = pre_label.repeat(gt_label.shape[0], 1, 1).float()
        loss_ce = ce(rep_pre_label.reshape(-1, self.part_num), gt_label.reshape(-1))

        return loss_emd, loss_ce, loss_cd
