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

    parser.add_argument('--config', type=str, default='../segment/config.yaml', help='yaml config file path')
    parser.add_argument('--start_ckpts_encoder', type=str,
                        default='../generate/logs/2024-05-07-19-09-26/encoder_1999.pth',
                        help='reload used ckpt path')

    args = parser.parse_args()
    return args
