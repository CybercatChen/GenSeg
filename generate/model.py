import sys
sys.path.append('..')
from generate.partae import *
from utils.PyTorchEMD import emd


class SegGen(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.part_para = None
        self.encoder_output_dim = config.model.encoder.output_dim
        self.superpoint_num = config.model.superpoint_num
        self.param_num = config.model.param_num
        self.mlp_hidden_dim = config.model.mlp_hidden_dim

        self.encoder = PointNetEncoder(config)

        self.attention_layer = nn.Sequential(
            nn.Conv1d(2048, self.superpoint_num, 1)
        )

        self.decoder = Decoder(config)

    def forward(self, points):
        p_feat, g_feat = self.encoder(points)  # B N C

        g_feat_repeated = g_feat.repeat(1, points.shape[1], 1)
        feat = torch.concat([p_feat, g_feat_repeated], -1)
        sp_feat = self.attention_layer(feat)  # B 50(sp num) N

        recon_points = self.decoder(sp_feat.transpose(2, 1))
        return recon_points, p_feat, sp_feat

    def get_loss(self, points, recon):
        # recon loss
        loss_emd = emd.earth_mover_distance(recon.transpose(2, 1), points.transpose(2, 1)).sum()
        return loss_emd


if __name__ == '__main__':
    from torchsummary import summary
    from segment.utils.config import *
    from segment.utils import parser, utils

    args = parser.get_args()
    config = get_config(args)

    net = SegGen(config).cuda()
    points = torch.rand(1, 2048, 3).cuda()  # 1 batch, 2 points, 3 coords
    p_feat, sp_feat = net(points)
    print(p_feat.shape)
