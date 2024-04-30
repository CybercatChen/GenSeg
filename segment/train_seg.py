from dataset import *
from torch.utils.data import DataLoader
from model import *
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from generate.utils.config import *
from generate.utils import parser, utils
from tensorboardX import SummaryWriter
from generate.utils.visualize import *

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
    criterion = model.get_loss

    best_loss = float('inf')
    for epoch in range(config.train.max_epoch):
        # train
        losses = train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, writer)
        scheduler.step()

        is_best = False

        writer.add_scalar('Loss/Epoch/loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Loss/Epoch/loss_cd', losses.avg(1), epoch)
        writer.add_scalar('Loss/Epoch/loss_mse', losses.avg(2), epoch)
        writer.add_scalar('Loss/Epoch/loss_ss', losses.avg(3), epoch)
        writer.add_scalar('Loss/Epoch/loss_loc', losses.avg(4), epoch)
        writer.add_scalar('Loss/Epoch/loss_sp_balance', losses.avg(5), epoch)

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
    losses = utils.AverageMeter(['loss_emd', 'loss_cd', 'loss_mse', 'loss_ss', 'loss_loc', 'loss_sp_balance'])
    n_batches = len(train_loader)
    model.train()
    for i, data in enumerate(train_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        recon, p_feat, sp_feat, sp_atten, label = model(points)

        # loss and backward
        loss_emd, loss_mse, loss_cd, loss_ss, loss_loc, loss_sp_balance \
            = criterion(points, recon, p_feat, sp_feat, sp_atten)
        loss = loss_emd + loss_ss + loss_sp_balance * 0.01
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # summary
        losses.update(
            [loss_emd.item(), loss_cd.item(), loss_mse.item(), loss_ss.item(), loss_loc.item(), loss_sp_balance.item()])
        n_itr = epoch * n_batches + i
        writer.add_scalar('Loss/Batch/loss_emd', loss_emd.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_cd', loss_cd.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_mse', loss_mse.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_mse', loss_ss.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_mse', loss_loc.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_mse', loss_sp_balance.item(), n_itr)
        writer.add_scalar('Loss/Batch/all_loss', loss.item(), n_itr)
        writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

        save_path = data['cate'][0] + '_' + str(np.array(data['id'][0]))
        write_ply(os.path.join(args.log_file, save_path + "_recon.ply"), recon[0].cpu().detach().numpy())
        vis_cate(points[0].cpu().detach().numpy(), label[0].cpu().detach().numpy(), config,
                 save_path=os.path.join(args.log_file, save_path + "_cate.ply"))

        torch.cuda.empty_cache()

    print('[Training] EPOCH: %d Losses = %s' % (
        epoch, [(name, '%.4f' % value) for name, value in zip(losses.items, losses.avg())]))

    return losses


if __name__ == '__main__':
    args = parser.get_args()
    timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    args.log_file = os.path.join(args.log_dir, f'{timestamp}')
    summarywriter = SummaryWriter(args.log_file)
    config = get_config(args)
    train(args, config, summarywriter)
