import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chair')
    parser.add_argument('--data_save_path', type=str, default=r'../data/')
    parser.add_argument('--input_data_path', type=str, default=r'../data/shapenet.hdf5')
    parser.add_argument('--scale_mode', type=str, default=None)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # args
    parser.add_argument('--gpu', type=int, default=0, help='the number of gpu to use')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, default='../segment/config.yaml', help='yaml config file path')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')

    # train args
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')

    # test args
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--ckpt_path', type=str, default='../segment/logs/2024-04-25-15-36-13/model_99.pth', help='test used ckpt path')
    parser.add_argument('--vis', action='store_true', default=True, help='test mode visualize superpoint result')

    args = parser.parse_args()
    return args


