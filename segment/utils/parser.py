import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='chair')
    parser.add_argument('--data_path', type=str, default='../data/')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2048, type=int)

    # training set
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=4000, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--ckpt_save_freq', default=500, type=int)
    parser.add_argument('--start_ckpts_encoder',
                        default='../generate/logs/2024-05-06-17-42-35/encoder_999.pth', type=str)
    parser.add_argument('--log_dir', type=str, default='./logs')

    # Model
    parser.add_argument('--part_num', default=4, type=int)
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--part_decoder', type=bool, default=True)

    args = parser.parse_args()
    return args
