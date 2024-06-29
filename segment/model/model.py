import sys

# from kaolin.metrics.pointcloud import chamfer_distance

sys.path.append('..')
from extention.PyTorchEMD import emd
from segment.model.pointnet import *
from segment.model.partae import *
from segment.utils.topoloss import *


class SegGen(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.part_num = args.part_num
        self.temp = 10
        self.encoder = PointNetEncoder(args)
        self.atten_encoder = PointNetEncoder(args)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(256, self.part_num, 1)
        )
        self.decoder = PartDecoder(args)
        # self.sp_atten = nn.Parameter(torch.rand(args.data_point, args.part_num))
        # init.xavier_uniform_(self.sp_atten)

    def forward(self, args, points):
        p_feat = self.encoder(points)  # B N C
        sp_atten = self.attention_layer(p_feat.transpose(2, 1))  # B 50(sp num) N
        sp_atten = F.softmax(sp_atten, dim=1)

        part_feat = torch.bmm(F.normalize(sp_atten, p=1, dim=2), p_feat)
        pre_label = torch.argmax(sp_atten.transpose(1, 2), axis=-1)

        part_recon, recon_all, means, logvars = self.decoder(part_feat)
        return part_recon, recon_all, p_feat, part_feat, sp_atten, pre_label, means, logvars

    def get_loss(self, points,
                 part_points, part_recon, part_feat,
                 p_feat, sp_atten,
                 means, logvars):
        B, N, C = p_feat.shape
        _, M, _ = sp_atten.shape

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
        # chd = chamfer_dist()

        for i in range(len(part_recon)):
            loss_emd += emd.earth_mover_distance(part_recon[i].transpose(2, 1).to('cuda'),
                                                 part_points[i].transpose(2, 1).to('cuda')).sum()
            dist = chamfer_distance(part_points[i].to('cuda'), part_recon[i].to('cuda'))
            loss_cd += (torch.mean(dist[0])) + (torch.mean(dist[1]))

            # dist1, dist2, idx1, idx2 = chd(part_points[i].to('cuda'), part_recon[i].to('cuda'))
            # loss_cd += (torch.mean(dist1)) + (torch.mean(dist2))
        loss_kl = 0
        for mean, logvar in zip(means, logvars):
            loss_kl += -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # ce = nn.CrossEntropyLoss()
        # gt_label = (gt_label - 1).clone().long()
        # loss_ce = ce(sp_atten, gt_label)

        part_feat = part_feat.transpose(0, 1)
        lowRankLoss = torch.zeros([self.part_num], dtype=torch.float).cuda()
        for i in range(self.part_num):
            _, s, _ = torch.svd(part_feat[i, :, :], some=False)
            lowRankLoss[i] = s[1] / s[0]

        highRankLoss = torch.zeros([int((self.part_num - 1) * self.part_num / 2)], dtype=torch.float).cuda()
        idx = 0
        for i in range(self.part_num):
            for j in range(self.part_num):
                if j <= i:
                    continue
                _, s, _ = torch.svd(torch.cat([part_feat[i, :, :], part_feat[j, :, :]], 1), some=False)
                highRankLoss[idx] = s[1] / s[0]
                idx = idx + 1
        loss_rank = 1 + torch.max(lowRankLoss) - torch.min(highRankLoss)

        return loss_emd, loss_cd, loss_loc, loss_bal, loss_rank, loss_kl

    def topo_loss(self, recon_all, points):
        B, N, C = recon_all.shape
        loss_topo = 0
        for i in range(B):
            loss_topo += topoloss_wasserstein_based(pt_gen=recon_all[i], pt_gt=points[i])
        return loss_topo
