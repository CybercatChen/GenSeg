from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
import config
from utils import utils
from utils.dataset import *
from model.model import Gen
from utils.visualize import *
from model.foldingnet import SkipValiationalFoldingNet

torch.autograd.set_detect_anomaly(True)


def train(args, writer):
    train_dataset = PartDataset(data_path=args.data_path, cates=args.dataset,
                                raw_data=None,
                                split='train', scale_mode=args.scale_mode, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=1)

    model = SkipValiationalFoldingNet(n_points=2048, feat_dims=512, shape='sphere')
    model = model.cuda()

    print(repr(model))
    print(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)
    criterion = model.get_loss

    for epoch in range(args.max_epoch):
        losses = train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, writer)
        scheduler.step()

        writer.add_scalar('Epoch/loss_emd', losses.avg(0), epoch)
        writer.add_scalar('Epoch/loss_cd', losses.avg(1), epoch)
        writer.add_scalar('Epoch/loss_kl', losses.avg(2), epoch)

        if (epoch + 1) % args.pretrain_ckpt_save_freq == 0:
            filename_encoder = os.path.join(args.log_file, f'encoder_{epoch}.pth')
            print(f'Saving encoder checkpoint to: {filename_encoder}')
            torch.save(model.encoder.state_dict(), filename_encoder)


def train_one_epoch(args, model, train_loader, optimizer, criterion, epoch, writer):
    losses = utils.AverageMeter(['loss_emd', 'loss_cd', 'loss_kl'])
    model.train()
    start_time = datetime.now()
    for i, data in enumerate(train_loader):
        args.batch_size = data['pointcloud'].shape[0]
        points = data['pointcloud'].cuda()

        recon, recon2, mu, sigma = model(points.transpose(1, 2))
        loss_emd, loss_cd, loss_kl = criterion(points, recon, recon2, mu, sigma)
        loss = loss_emd + 0.00001 * loss_kl

        loss /= args.batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update([loss_emd.item(), loss_cd.item(), loss_kl.item()])

        if ((i + 10) % 2 == 0) & (epoch % 20 == 0):
            save_path = data['cate'][0] + '_' + str(np.array(data['id'][0]))
            write_ply(os.path.join(args.log_file, save_path + "_recon.ply"), recon[0].cpu().detach().numpy())
            write_ply(os.path.join(args.log_file, save_path + "_recon2.ply"), recon2[0].cpu().detach().numpy())
            write_ply(os.path.join(args.log_file, save_path + "_ref.ply"), points[0].cpu().detach().numpy())
            # label = torch.argmax(sp_atten, dim=-2)
            # vis_cate(points[0].cpu().detach().numpy(), label.cpu().detach().numpy(), args,
            #          save_path=os.path.join(args.log_file, save_path + "_cate.ply"))
        torch.cuda.empty_cache()

    end_time = datetime.now()
    epoch_time = (end_time - start_time).total_seconds()
    print(f'[Training] EPOCH:{epoch}, Time:{epoch_time:.2f}, '
          f'Losses={[(name, f"{value:.4f}") for name, value in zip(losses.items, losses.avg())]}')
    return losses


if __name__ == '__main__':
    args = config.get_args()
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.log_file = os.path.join(args.log_dir, 'gen', args.dataset, f'{timestamp}')
    writer = SummaryWriter(args.log_file)
    train(args, writer)
