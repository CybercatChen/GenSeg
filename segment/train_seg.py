import torch
from partdecoder import *
from dataset import *
from torch.utils.data import DataLoader
from superpoint import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from segment.utils.config import *
from segment.utils import parser, utils
from tensorboardX import SummaryWriter

torch.autograd.set_detect_anomaly(True)


def train(args, config, writer):
    train_dataset = PCDataset(data_path=args.input_data_path, output_path=args.data_save_path,
                              cates=args.dataset, raw_data=None,
                              split='train', scale_mode=args.scale_mode, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=1)

    # build GMVAE
    model = SegGen(config)
    model = model.cuda()

    # optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.kwargs.lr)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.max_epoch)

    # Criterion
    criterion = model.encoder.get_loss

    best_loss = float('inf')
    for epoch in range(config.train.max_epoch):
        # train
        losses = train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, writer)
        scheduler.step()

        is_best = False

        writer.add_scalar('Loss/Epoch/0_loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Loss/Epoch/1_loss_ss', losses.avg(1), epoch)
        writer.add_scalar('Loss/Epoch/2_oss_loc', losses.avg(2), epoch)
        writer.add_scalar('Loss/Epoch/3_loss_sp_balance', losses.avg(3), epoch)
        writer.add_scalar('Loss/Epoch/4_all_loss', losses.avg(4), epoch)

        if (epoch + 1) % config.train.ckpt_save_freq == 0:
            filename = os.path.join(args.log_file, f'model_{epoch}.pth')
            print(f'Saving checkpoint to: {filename}')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
                'is_best': is_best
            }, filename)


def train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, writer):
    losses = utils.AverageMeter(['loss_emd', 'loss_ss', 'loss_loc', 'loss_sp_balance', 'all_loss'])
    n_batches = len(train_loader)
    torch.autograd.set_detect_anomaly(True)
    model.train()
    for i, data in enumerate(train_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        recon, p_feat, sp_atten, sp_feat = model(points)

        # loss and backward
        loss_ss, loss_loc, loss_sp_balance, loss_emd, label = criterion(points, p_feat, sp_atten, sp_feat, recon)
        loss = 1.0 * loss_ss + 1.0 * loss_loc + 1.0 * loss_emd
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # summary
        losses.update([loss_emd.item(), loss_ss.item(), loss_loc.item(), loss_sp_balance.item(), loss.item()])
        n_itr = epoch * n_batches + i
        writer.add_scalar('Loss/Batch/0_loss_emd', loss_emd.item(), n_itr)
        writer.add_scalar('Loss/Batch/1_loss_ss', loss_ss.item(), n_itr)
        writer.add_scalar('Loss/Batch/2_loss_loc', loss_loc.item(), n_itr)
        writer.add_scalar('Loss/Batch/3_loss_sp_balance', loss_sp_balance.item(), n_itr)
        writer.add_scalar('Loss/Batch/4_all_loss', loss.item(), n_itr)
        writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        torch.cuda.empty_cache()

        # message output
        if (i + 1) % 1 == 0:
            print('[Epoch %d/%d][Batch %d/%d] Losses = %s lr = %.6f' %
                  (epoch, config.train.max_epoch, i + 1, n_batches,
                   ['%.8f' % l for l in losses.val()], optimizer.param_groups[0]['lr']))

    print('[Training] EPOCH: %d Losses = %s' % (epoch, ['%.8f' % l for l in losses.avg()]))

    return losses


if __name__ == '__main__':
    args = parser.get_args()
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.log_file = os.path.join(args.log_dir, f'{timestamp}')
    summarywriter = SummaryWriter(args.log_file)
    config = get_config(args)
    train(args, config, summarywriter)
