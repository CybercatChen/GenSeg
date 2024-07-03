from model.pointnet import *
from model.pointnet2 import *
from utils.utils import *
from extention.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from extention.earth_movers_distance.emd import EarthMoverDistance


class SegGen(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.part_num = args.part_num
        self.temp = 10
        # self.encoder = PointNet2PointFeatureStable()
        self.encoder = PointNetEncoder()
        self.attention_layer = nn.Sequential(
            nn.Conv1d(256, self.part_num, 1)
        )
        self.decoder = PartDecoder(args)
        self.chd = chamfer_3DDist()
        self.emd = EarthMoverDistance()

    def forward(self, args, points):
        p_feat = self.encoder(points.transpose(1, 2))  # B C N
        sp_atten = self.attention_layer(p_feat)  # B M N
        sp_atten = F.softmax(sp_atten, dim=1)

        pre_label = torch.argmax(sp_atten.transpose(1, 2), axis=-1)
        part_feat = sample_point(args, labels=pre_label, points=p_feat)

        part_recon, recon_all, means, logvars = self.decoder(part_feat)
        return part_recon, recon_all, p_feat, part_feat, sp_atten, pre_label, means, logvars

    def get_loss(self, points,
                 part_points, part_recon, part_feat,
                 p_feat, sp_atten,
                 means, logvars):
        p_feat = p_feat.transpose(1, 2)
        B, N, C = p_feat.shape
        _, M, _ = sp_atten.shape
        # # ss loss
        # sp_feat_un = part_feat.unsqueeze(2)  # B M 1 C
        # p_feat_un = p_feat.unsqueeze(1)  # B 1 N C
        # feat_dist = sp_feat_un - p_feat_un  # B M N C
        # feat_dist = torch.norm(feat_dist, dim=-1)  # B M N
        # feat_dist = feat_dist * sp_atten  # B M N
        # # loss_ss = torch.sum(feat_dist) / (M * N)
        # loss_ss = torch.sum(feat_dist)  # paper
        # loc loss
        centriods = torch.bmm(F.normalize(sp_atten, p=1, dim=2), points)  # B M 3
        cent_un = centriods.unsqueeze(2)  # B M 1 3
        points_un = points.unsqueeze(1)  # B 1 N 3
        coord_dist = cent_un - points_un  # B M N 3
        coord_dist = torch.norm(coord_dist, dim=-1)  # B M N
        coord_dist = coord_dist * sp_atten  # B M N
        loss_loc = torch.sum(coord_dist)  # paper
        # sp balance loss
        sp_atten_per_sp = torch.sum(sp_atten, dim=-1)  # B M
        sp_atten_sum = torch.sum(sp_atten_per_sp, dim=-1, keepdim=True) / M  # B 1
        loss_bal = torch.sum((sp_atten_per_sp - sp_atten_sum) ** 2) / M

        # loss recon
        loss_emd = 0
        loss_cd = 0

        for i in range(len(part_recon)):
            loss_emd += torch.mean(self.emd(part_recon[i].to('cuda'),
                                            part_points[i].to('cuda')))
            dist1, dist2, idx1, idx2 = self.chd(part_points[i].to('cuda'), part_recon[i].to('cuda'))
            loss_cd += (torch.mean(dist1)) + (torch.mean(dist2))

        loss_kl = 0
        for mean, logvar in zip(means, logvars):
            loss_kl += -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # part_feat = part_feat.transpose(0, 1)
        # lowRankLoss = torch.zeros([self.part_num], dtype=torch.float).cuda()
        # for i in range(self.part_num):
        #     _, s, _ = torch.svd(part_feat[i, :, :], some=False)
        #     lowRankLoss[i] = s[1] / s[0]
        #
        # highRankLoss = torch.zeros([int((self.part_num - 1) * self.part_num / 2)], dtype=torch.float).cuda()
        # idx = 0
        # for i in range(self.part_num):
        #     for j in range(self.part_num):
        #         if j <= i:
        #             continue
        #         _, s, _ = torch.svd(torch.cat([part_feat[i, :, :], part_feat[j, :, :]], 1), some=False)
        #         highRankLoss[idx] = s[1] / s[0]
        #         idx = idx + 1
        # loss_rank = 1 + torch.max(lowRankLoss) - torch.min(highRankLoss)
        loss_rank = torch.zeros(1).cuda()
        loss_ss = torch.zeros(1).cuda()
        return loss_emd, loss_cd, loss_loc, loss_bal, loss_rank, loss_kl, loss_ss


class Gen(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.part_num = args.part_num
        self.encoder = PointNet2PointFeatureStable()
        self.decoder = PointNetDecoder(args)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(128, self.part_num, 1)
        )
        self.chd = chamfer_3DDist()
        self.emd = EarthMoverDistance()

    def forward(self, points):
        p_feat = self.encoder(points.transpose(1, 2))
        recon_points = self.decoder(p_feat.transpose(1, 2))
        return recon_points, p_feat

    def get_loss(self, points, recon):
        loss_emd = torch.mean(self.emd(recon, points))
        dist1, dist2, idx1, idx2 = self.chd(points, recon)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))
        return loss_emd, loss_cd


class Cls(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.part_num = args.part_num
        self.temp = 10
        # self.encoder = PointNet2PointFeatureStable()
        self.encoder = PointNetEncoder()

        self.attention_layer = nn.Sequential(
            nn.Conv1d(args.latent_dim, self.part_num, 1)
        )
        self.decoder = PointNetDecoder_Gen(args)

        self.chd = chamfer_3DDist()
        self.emd = EarthMoverDistance()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, points):
        p_feat = self.encoder(points.transpose(1, 2))
        sp_atten = self.attention_layer(p_feat)
        sp_atten = F.softmax(sp_atten, dim=1)

        part_feat = torch.bmm(F.normalize(sp_atten, p=1, dim=2), p_feat.transpose(1, 2))
        recon_all = self.decoder(part_feat)
        return recon_all, p_feat, part_feat, sp_atten

    def get_loss(self, points, recon_all, label, sp_atten):
        loss_emd = torch.mean(self.emd(recon_all, points))
        dist1, dist2, idx1, idx2 = self.chd(points, recon_all)
        loss_cd = (torch.mean(dist1)) + (torch.mean(dist2))

        sp_atten = sp_atten.transpose(1, 2).contiguous().view(-1, self.part_num)
        label = label.view(-1).long()
        loss_ce = self.ce_loss(sp_atten, label)
        return loss_emd, loss_cd, loss_ce
