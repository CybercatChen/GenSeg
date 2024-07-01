import sys

sys.path.append('..')
from generate.model.pointnet import *
from generate.model.partae import *
from extention.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extention.earth_movers_distance.emd import EarthMoverDistance


class SegGen(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.part_num = args.part_num
        self.encoder = PointNet2PointFeatureStable()
        self.decoder = PointNetDecoder(args)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(128, self.part_num, 1)
        )
        # init.xavier_uniform_(self.sp_atten)
        self.chd = chamfer_3DDist()
        self.emd = EarthMoverDistance()

    def forward(self, points):
        p_feat = self.encoder(points.transpose(1, 2))
        sp_atten = self.attention_layer(p_feat)  # B 50(sp num) N

        sp_atten = F.softmax(sp_atten, dim=1)
        # part_feat = torch.matmul(p_feat.transpose(1, 2), sp_atten).transpose(1, 2)
        part_feat = F.normalize(torch.bmm(sp_atten, p_feat.transpose(1, 2)), p=1, dim=2)
        recon_points = self.decoder(part_feat)
        return recon_points, p_feat, sp_atten

    def get_loss(self, points, recon):
        # recon loss
        loss_emd = torch.mean(self.emd(recon, points))
        dist1, dist2, idx1, idx2 = self.chd(points, recon)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))

        # ce = nn.CrossEntropyLoss()
        # gt_label = (gt_label - 1).clone().long()
        # rep_pre_label = pre_label.repeat(gt_label.shape[0], 1, 1).float()
        # loss_ce = ce(rep_pre_label.reshape(-1, self.part_num), gt_label.reshape(-1))

        return loss_emd, loss_cd
