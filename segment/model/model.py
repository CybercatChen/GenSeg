import sys
from chamfer_distance import ChamferDistance as chamfer_dist

sys.path.append('..')
from script.PyTorchEMD import emd
from segment.model.pointnet import *
from segment.model.partae import *


class SegGen(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.part_num = args.part_num

        self.encoder = PointNetEncoder(args)
        self.atten_encoder = PointNetEncoder(args)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(256, self.part_num, 1)
        )
        self.decoder = PartDecoder(args)

    def forward(self, points):
        p_feat = self.encoder(points)  # B N C
        sp_atten = self.attention_layer(p_feat.transpose(2, 1))  # B 50(sp num) N
        sp_atten = F.softmax(sp_atten, dim=1)  # B 50(sp num) N, softmax on superpoint dim: dim-1

        # get superpoint features S
        sp_feat = torch.bmm(F.normalize(sp_atten, p=1, dim=2),
                            p_feat)  # B 50(sp num) C, l1-norm on attention map last dim: dim-2
        label = torch.argmax(sp_atten.transpose(1, 2), axis=-1)
        recon_parts, recon_all = self.decoder(sp_feat)
        return recon_parts, recon_all, p_feat, sp_feat, sp_atten, label

    def get_loss(self, points, recon_all, part_points, recon, p_feat, sp_feat, sp_atten):
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

        # loss_inter
        similarity_matrix = torch.bmm(sp_feat, sp_feat.transpose(1, 2))  # B M M
        mask = torch.eye(M, device=sp_feat.device).unsqueeze(0).repeat(B, 1, 1)
        similarity_matrix = similarity_matrix * (1 - mask) + (-1e9) * mask
        max_similarity, _ = torch.max(similarity_matrix, dim=-1)  # B M
        loss_inter = torch.mean(max_similarity)

        # loss recon
        # part recon
        loss_emd = 0
        loss_cd = 0
        chd = chamfer_dist()

        for i in range(len(recon)):
            loss_emd += emd.earth_mover_distance(recon[i].transpose(2, 1).to('cuda'),
                                                 part_points[i].transpose(2, 1).to('cuda')).sum()
            dist1, dist2, idx1, idx2 = chd(part_points[i].to('cuda'), recon[i].to('cuda'))
            loss_cd += (torch.mean(dist1)) + (torch.mean(dist2))
        # global recon
        loss_emd += emd.earth_mover_distance(recon_all.transpose(2, 1).to('cuda'),
                                             points.transpose(2, 1).to('cuda')).sum()
        dist1, dist2, idx1, idx2 = chd(points.to('cuda'), recon_all.to('cuda'))
        loss_cd += (torch.mean(dist1)) + (torch.mean(dist2))
        return loss_emd, loss_cd, loss_ss, loss_loc, loss_sp_balance, loss_inter
