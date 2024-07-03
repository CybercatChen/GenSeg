import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default='chair')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--scale_mode', type=str, default='global_unit')
    parser.add_argument('--data_point', default=2000, type=int)
    parser.add_argument('--pretrain_data_point', default=2048, type=int)
    parser.add_argument('--part_point', default=400, type=int)

    # training set
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--lr', default=0.0001, type=int)

    parser.add_argument('--pretrain_ckpt_save_freq', default=200, type=int)
    parser.add_argument('--ckpt_save_freq', default=4000, type=int)
    # parser.add_argument('--start_ckpts_encoder',
    #                     default='./logs/GEN/vessel_left/2024-07-01-22-17-10/encoder_5999.pth', type=str)
    parser.add_argument('--start_ckpts_encoder',
                        default='./logs/GEN/vessel_all/2024-05-11-21-03-52/encoder_129.pth', type=str)
    # Model
    parser.add_argument('--part_num', default=5, type=int)
    parser.add_argument('--latent_dim', default=256, type=int)
    parser.add_argument('--log_dir', type=str, default='./logs')

    args = parser.parse_args()
    return args
