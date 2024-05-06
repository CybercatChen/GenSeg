from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import time
import sys

sys.path.append('..')
from generate.utils.config import *
from generate.utils import parser, utils
from generate.utils.dataset import *
from generate.model.model import *
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

    for epoch in range(config.train.max_epoch):
        # train
        losses = train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, writer)
        scheduler.step()

        writer.add_scalar('Loss/Epoch/loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Loss/Epoch/loss_cd', losses.avg(1), epoch)
        writer.add_scalar('Loss/Epoch/loss_mse', losses.avg(2), epoch)

        if (epoch + 1) % config.train.ckpt_save_freq == 0:
            filename_encoder = os.path.join(args.log_file, f'encoder_{epoch}.pth')
            filename_decoder = os.path.join(args.log_file, f'decoder_{epoch}.pth')
            print(f'Saving encoder checkpoint to: {filename_encoder}')
            print(f'Saving decoder checkpoint to: {filename_decoder}')

            torch.save(model.encoder.state_dict(), filename_encoder)
            torch.save(model.decoder.state_dict(), filename_decoder)


def train_one_epoch(args, config, model, train_loader, optimizer, criterion, epoch, writer):
    losses = utils.AverageMeter(['loss_emd', 'loss_cd', 'loss_mse'])
    n_batches = len(train_loader)
    model.train()
    for i, data in enumerate(train_loader):
        batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()
        recon, p_feat, sp_feat = model(points)

        # loss and backward
        loss_emd, loss_mse, loss_cd = criterion(points, recon)
        loss = loss_emd + loss_cd
        loss /= batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # summary
        losses.update([loss_emd.item(), loss_cd.item(), loss_mse.item()])
        n_itr = epoch * n_batches + i
        writer.add_scalar('Loss/Batch/loss_emd', loss_emd.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_cd', loss_cd.item(), n_itr)
        writer.add_scalar('Loss/Batch/loss_mse', loss_mse.item(), n_itr)
        writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
        if (i + 1) % 10 == 0:
            save_path = data['cate'][0] + '_' + str(np.array(data['id'][0]))
            write_ply(os.path.join(args.log_file, save_path + "_recon.ply"), recon[0].cpu().detach().numpy())
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
