from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from utils.dataset import *
from model.model import *
from utils import utils
import config
from utils.visualize import *

torch.autograd.set_detect_anomaly(True)


def train(args, writer):
    train_dataset = PCDataset(data_path=args.data_path, cates=args.dataset,
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
        losses = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch)
        scheduler.step()

        writer.add_scalar('Epoch/loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Epoch/loss_cd', losses.avg(1), epoch)
        writer.add_scalar('Epoch/loss_loc', losses.avg(2), epoch)
        writer.add_scalar('Epoch/loss_bal', losses.avg(3), epoch)
        writer.add_scalar('Epoch/loss_rank', losses.avg(4), epoch)
        writer.add_scalar('Epoch/loss_kl', losses.avg(5), epoch)
        writer.add_scalar('Epoch/loss_ss', losses.avg(5), epoch)

        if (epoch + 1) % args.ckpt_save_freq == 0:
            filename = os.path.join(args.log_file, f'model_{epoch}.pth')
            print(f'Saving checkpoint to: {filename}')
            torch.save(model.state_dict(), filename)


def train_one_epoch(args, model, train_loader, optimizer, criterion, epoch):
    losses = utils.AverageMeter(
        ['loss_emd', 'loss_cd', 'loss_loc', 'loss_bal', 'loss_rank', 'loss_kl', 'loss_ss'])
    n_batches = len(train_loader)
    model.train()
    start_time = datetime.now()

    for i, data in enumerate(train_loader):
        args.batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        part_recon, recon_all, p_feat, part_feat, sp_atten, pre_label, means, logvars = model(args, points)

        part_points = utils.sample_aprt_point(args, pre_label, points, args.part_point)

        loss_emd, loss_cd, loss_loc, loss_bal, loss_rank, loss_kl, loss_ss \
            = criterion(points, part_points, part_recon,
                        p_feat, sp_atten, means, logvars)
        loss = loss_cd + loss_emd + loss_loc + loss_bal

        loss /= args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # summary
        losses.update([loss_emd.item(), loss_cd.item(),
                       loss_loc.item(), loss_bal.item(),
                       loss_rank.item(), loss_kl.item(),
                       loss_ss.item()])

        if ((i + 1) % (n_batches // 4) == 0) & (epoch % 400 == 0):
            save_path = data['cate'][0] + '_' + str(np.array(data['id'][0]))
            vis_part = torch.stack([points[0] for points in part_recon], dim=0)
            vis_recon = torch.concat([points[0] for points in part_recon], dim=0)
            write_ply_with_color(os.path.join(args.log_file, save_path + "_recon.ply"),
                                 vis_part.cpu().detach().numpy())
            vis_cate(points[0].cpu().detach().numpy(), pre_label[0].cpu().detach().numpy(), args,
                     save_path=os.path.join(args.log_file, save_path + "_cate.ply"))

        torch.cuda.empty_cache()
    end_time = datetime.now()
    epoch_time = (end_time - start_time).total_seconds()
    print(f'[Training] EPOCH:{epoch}, Time:{epoch_time:.2f}, '
          f'Losses={[(name, f"{value:.4f}") for name, value in zip(losses.items, losses.avg())]}')

    return losses


if __name__ == '__main__':
    args = config.get_args_seg()
    from datetime import datetime

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.log_file = os.path.join(args.log_dir, 'AE', args.dataset + f'_{timestamp}')
    writer = SummaryWriter(args.log_file)
    train(args, writer)
