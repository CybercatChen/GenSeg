from torch.utils.data import DataLoader
import open3d as o3d
from generate.utils.dataset import *
from generate.model.model import *
from generate.utils.config import *
from generate.utils.utils import *
from generate.utils import parser

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_latent_3d(latent, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2])
    ax.set_xlabel('Latent 1')
    ax.set_ylabel('Latent 2')
    ax.set_zlabel('Latent 3')
    ax.set_title(title)
    plt.show()

# 可视化降维后的潜变量至2维空间
def visualize_latent_2d(latent, title):
    plt.scatter(latent[:, 0], latent[:, 1])
    plt.xlabel('Latent 1')
    plt.ylabel('Latent 2')
    plt.title(title)
    plt.show()


def latent_vis(args, config):
    train_dataset = PCDataset(data_path=args.input_data_path, output_path=args.data_save_path,
                              cates=args.dataset, raw_data=None,
                              split='train', scale_mode=args.scale_mode, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=1)

    model = SegGen(config)
    pre_encoder = torch.load(args.start_ckpts_encoder)
    model.encoder.load_state_dict(pre_encoder)
    model = model.cuda()
    all_p_feat = []
    for i, data in enumerate(train_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        with torch.no_grad():
            recon_points, p_feat, sp_feat = model(points)
        all_p_feat.append(p_feat.cpu().numpy())

    all_p_feat = np.concatenate(all_p_feat, axis=0)
    n_components = 2

    tsne = TSNE(n_components=n_components, perplexity=30, random_state=0)
    latent = tsne.fit_transform(all_p_feat[0])
    if n_components == 3:
        visualize_latent_3d(latent, title="t-SNE Visualization of Latent Space")
    else:
        visualize_latent_2d(latent, title="t-SNE Visualization of Latent Space")
    return latent


def test(args, config):
    test_dataset = PCDataset(data_path=args.input_data_path, raw_data=None, output_path=args.data_save_path,
                             cates=args.dataset,
                             split='val', scale_mode=args.scale_mode, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, num_workers=1)

    # load model
    model = SegGen(config)
    model = model.cuda()
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    is_best = checkpoint['is_best']

    print(f'Load checkpoint from {args.ckpt_path}, epoch: {epoch}, best_loss: {best_loss}, is_best: {is_best}')

    # Criterion
    criterion = model.get_loss

    # test
    if args.vis:
        vis_dir = os.path.join(args.log_dir, 'visualize')
        os.makedirs(vis_dir, exist_ok=True)
        np.random.seed(123)
        sp_colors = np.random.rand(config.model.superpoint_num, 3)
    losses = AverageMeter(['loss_fit', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'all_loss'])
    model.eval()
    for i, data in enumerate(test_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        with torch.no_grad():
            recon_points, p_feat, sp_feat = model(points)

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


if __name__ == '__main__':
    args = parser.get_args()
    config = get_config(args)
    latent_vis(args, config)
