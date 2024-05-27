import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import time
import sys

sys.path.append('..')
from segment.utils.dataset import *
from segment.model.model import *
from segment.utils import utils, parser
from segment.utils.visualize import *

torch.autograd.set_detect_anomaly(True)


def train(args, writer):
    train_dataset = PartDataset(data_path=args.data_path, cates=args.dataset,
                                raw_data=None,
                                split='train', scale_mode=args.scale_mode, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=1)
    model = SegGen(args)
    pre_encoder = torch.load(args.start_ckpts_encoder)
    model.encoder.load_state_dict(pre_encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False

    model = model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)

    print(repr(model))
    print(args)
    # Criterion
    criterion = model.get_loss

    for epoch in range(args.max_epoch):
        # train
        losses = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, writer)
        scheduler.step()

        writer.add_scalar('Epoch/loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Epoch/loss_cd', losses.avg(1), epoch)
        writer.add_scalar('Epoch/loss_ss', losses.avg(2), epoch)
        writer.add_scalar('Epoch/loss_loc', losses.avg(3), epoch)
        writer.add_scalar('Epoch/loss_sp_balance', losses.avg(4), epoch)
        writer.add_scalar('Epoch/loss_kl', losses.avg(5), epoch)
        writer.add_scalar('Epoch/loss_ce', losses.avg(6), epoch)

        if (epoch + 1) % args.ckpt_save_freq == 0:
            filename = os.path.join(args.log_file, f'model_{epoch}.pth')
            print(f'Saving checkpoint to: {filename}')
            torch.save(model.state_dict(), filename)


def train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, writer):
    losses = utils.AverageMeter(
        ['loss_emd', 'loss_cd', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'loss_kl', 'loss_ce'])
    n_batches = len(train_loader)
    model.train()

    for i, data in enumerate(train_loader):
        args.batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        gt_label = data['labels'].cuda()
        part_recon, recon_all, p_feat, sp_feat, sp_atten, pre_label, means, logvars = model(args, points)

        part_points = utils.sample_aprt_point(args, pre_label, points)

        # loss and backward
        loss_emd, loss_cd, loss_ss, loss_loc, loss_sp_balance, loss_inter, loss_kl, loss_ce \
            = criterion(points, part_points, part_recon,
                        gt_label, pre_label,
                        p_feat, sp_feat, sp_atten, means, logvars)
        loss = loss_emd + loss_cd + 100 * loss_ce
        loss /= args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # summary
        losses.update([loss_emd.item(), loss_cd.item(),
                       loss_ss.item(),
                       loss_loc.item(), loss_sp_balance.item(),
                       loss_kl.item(), loss_ce.item()])
        n_itr = epoch * n_batches + i
        writer.add_scalar('Batch/loss_emd', loss_emd.item(), n_itr)
        writer.add_scalar('Batch/loss_cd', loss_cd.item(), n_itr)
        writer.add_scalar('Batch/loss_ss', loss_ss.item(), n_itr)
        writer.add_scalar('Batch/loss_loc', loss_loc.item(), n_itr)
        writer.add_scalar('Batch/loss_sp_balance', loss_sp_balance.item(), n_itr)
        writer.add_scalar('Batch/loss_kl', loss_kl.item(), n_itr)
        writer.add_scalar('Batch/loss_ce', loss_ce.item(), n_itr)
        writer.add_scalar('Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        if ((i + 1) % (n_batches // 4) == 0) & (epoch % 40 == 0):
            save_path = data['cate'][0] + '_' + str(np.array(data['id'][0]))
            vis_part = torch.stack([points[0] for points in part_recon], dim=0)
            vis_recon = torch.concat([points[0] for points in part_recon], dim=0)
            write_ply_with_color(os.path.join(args.log_file, save_path + "_recon.ply"),
                                 vis_part.cpu().detach().numpy())
            vis_cate(points[0].cpu().detach().numpy(), pre_label[0].cpu().detach().numpy(), args,
                     save_path=os.path.join(args.log_file, save_path + "_cate.ply"))
            vis_cate(vis_recon.cpu().detach().numpy(), gt_label[0].cpu().detach().numpy(), args,
                     save_path=os.path.join(args.log_file, save_path + "_gt.ply"))

        torch.cuda.empty_cache()

    print('[Training] EPOCH: %d Losses = %s' % (
        epoch, [(name, '%.4f' % value) for name, value in zip(losses.items, losses.avg())]))

    return losses


if __name__ == '__main__':
    args = parser.get_args()
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.log_file = os.path.join(args.log_dir, args.dataset, f'{timestamp}')
    writer = SummaryWriter(args.log_file)
    train(args, writer)
