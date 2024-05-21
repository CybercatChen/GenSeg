from torch.utils.data import DataLoader
import open3d as o3d
from segment.utils.dataset import *
from segment.utils.utils import *
from segment.utils import parser
from segment.model.model import *
from segment.utils.visualize import *


def test(args):
    test_dataset = PCDataset(data_path=args.input_data_path, raw_data=None, output_path=args.data_save_path,
                             cates=args.dataset,
                             split='val', scale_mode=args.scale_mode, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=1)

    # load model
    model = SegGen(args)
    model = model.cuda()
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    criterion = model.get_loss

    if args.vis:
        vis_dir = os.path.join(args.log_dir, 'visualize')
        os.makedirs(vis_dir, exist_ok=True)
        np.random.seed(123)
        sp_colors = np.random.rand(args.part_num, 3)
    losses = AverageMeter(['loss_emd', 'loss_cd', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'loss_inter', 'kl_loss'])
    model.eval()
    for i, data in enumerate(test_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        with torch.no_grad():
            p_feat, sp_atten, sp_feat = model(points)

        # loss
        loss_ss, loss_loc, loss_sp_balance = criterion(points, p_feat, sp_atten, sp_feat)
        loss = 1.0 * loss_ss + 1.0 * loss_loc + 0.001 * loss_sp_balance
        loss /= batch_size

        # summary
        losses.update([loss_ss.item(), loss_loc.item(), loss_sp_balance.item(), loss.item()])
        torch.cuda.empty_cache()

        # visulize
        if args.vis:
            pcd = o3d.geometry.PointCloud()
            for b in range(batch_size):
                vis_file = data['cate'][b] + '_' + str(np.array(data['id'][b])) + ".ply"
                vis_path = os.path.join(vis_dir, vis_file)
                points = points[b].cpu().numpy()
                sp_atten = sp_atten[b].cpu().numpy().T
                sp_idx = np.argmax(sp_atten, axis=-1)
                print(vis_file, np.unique(sp_idx, return_counts=True))
                colors = sp_colors[sp_idx].reshape(-1, 3)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(vis_path, pcd)

    print('[Test] Losses = %s' % (['%.8f' % l for l in losses.avg()]))


def sample_gen(args):
    model = SegGen(args)
    model = model.cuda()
    checkpoint = torch.load('../segment/logs/vessel_left/2024-05-17-16-07-38/model_3999.pth')
    model.load_state_dict(checkpoint, strict=True)

    vis_dir = os.path.join(args.log_dir, 'visualize')
    os.makedirs(vis_dir, exist_ok=True)

    part_feature_1 = torch.randn(1, 1, 256).cuda()
    part_feature_2 = torch.randn(1, 1, 256).cuda()
    part_feature_3 = torch.randn(1, 1, 256).cuda()
    part_feature_4 = torch.randn(1, 1, 256).cuda()
    part_feature = torch.cat((part_feature_1, part_feature_2, part_feature_3, part_feature_4), dim=1)
    # part_feature = torch.randn(1, 4, 256).cuda()

    model.eval()

    recon_parts, recon_all, _, _ = model.decoder(part_feature)

    part_recon = torch.stack([points[0] for points in recon_parts], dim=0)
    write_ply_with_color(os.path.join(vis_dir, args.dataset + "_sample.ply"),
                         part_recon.cpu().detach().numpy())


if __name__ == '__main__':
    args = parser.get_args()
    sample_gen(args)
