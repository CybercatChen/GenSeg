import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='chair')
    parser.add_argument('--data_save_path', type=str, default=r'../data/')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2048, type=int)

    # training set
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=500, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--ckpt_save_freq', default=50, type=int)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--part_num', default=4, type=int)
    parser.add_argument('--ckpt_encoder', type=str, default='./logs')

    args = parser.parse_args()
    return args
