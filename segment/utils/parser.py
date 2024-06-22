import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='vessel_left')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2024, type=int)
    parser.add_argument('--part_point', default=400, type=int)

    # training set
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--ckpt_save_freq', default=4000, type=int)
    parser.add_argument('--start_ckpts_encoder',
                        default='../generate/logs/vessel_left/2024-05-12-13-20-31/encoder_499.pth', type=str)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Model
    parser.add_argument('--part_num', default=5, type=int)
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--part_decoder', type=bool, default=True)

    args = parser.parse_args()
    return args
