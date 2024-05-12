import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='vessel_left')
    parser.add_argument('--data_save_path', type=str, default='../data/')
    parser.add_argument('--input_data_path', type=str, default='../data/vessel_left.hdf5')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2048, type=int)

    # training set
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--ckpt_save_freq', default=400, type=int)
    parser.add_argument('--start_ckpts_encoder', type=str,
                        default='../generate/logs/vessel_left/2024-05-12-13-20-31/encoder_499.pth')
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Model
    parser.add_argument('--part_num', default=5, type=int)
    parser.add_argument('--part_decoder', type=bool, default=True)

    args = parser.parse_args()
    return args
