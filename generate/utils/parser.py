import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='chair')
    parser.add_argument('--data_save_path', type=str, default=r'../data/')
    parser.add_argument('--input_data_path', type=str, default=r'../data/shapenet.hdf5')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2048, type=int)

    # training set
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=2000, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--ckpt_save_freq', default=500, type=int)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--part_num', default=4, type=int)

    args = parser.parse_args()
    return args
