import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vessel_left')
    parser.add_argument('--data_save_path', type=str, default=r'../data/')
    parser.add_argument('--input_data_path', type=str, default=r'../data/vessel_left.hdf5')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # args
    parser.add_argument('--gpu', type=int, default=0, help='the number of gpu to use')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--config', type=str, default='../generate/config.yaml', help='yaml config file path')
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')

    # train args
    parser.add_argument('--start_ckpts_encoder', type=str,
                        default='../generate/logs/2024-05-06-17-42-35/encoder_999.pth', help='reload used ckpt path')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='autoresume training (interrupted by accident)')

    # test args
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--vis', action='store_true', default=True, help='test mode visualize superpoint result')

    args = parser.parse_args()
    return args
