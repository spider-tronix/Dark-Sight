import argparse
import datetime

from torch.utils.data import DataLoader

from data.loader import Precious
from data.transforms import *
from models.vanilla_unet import VanillaUNet

configurations = {
    1: dict(
        max_iteration=1000000,
        lr=1e-4,
        momentum=0.9,
        weight_decay=0.0,
        gamma=0.25,
        step_size=32300,  # "lr_policy: step"
        interval_validate=1000,
    ),
}


def main():
    timestamp = datetime()
    parser = argparse.ArgumentParser("DarkSight!")
    parser.add_argument('cmd', type=str, choices=['train', 'test'], help='train or test')
    parser.add_argument('--log_file', type=str, default='./train.log', help='log file')
    parser.add_argument('--checkpoint_dir', type=str, default='./weights/',
                        help='checkpoints directory')
    parser.add_argument('--result_dir', type=str, default='./result/Sony/',
                        help='directory where results are saved')
    parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys(),
                        help='the number of settings and hyperparameters used in training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='batch size in validation')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size in test')
    parser.add_argument('--patch_size', type=int, default=None, help='patch size')
    parser.add_argument('--save_freq', type=int, default=1, help='checkpoint save frequency')
    parser.add_argument('--print_freq', type=int, default=1, help='log print frequency')
    parser.add_argument('--upper_train', type=int, default=-1, help='max of train images(for debug)')
    parser.add_argument('--upper_valid', type=int, default=-1, help='max of valid images(for debug)')
    parser.add_argument('--upper_test', type=int, default=-1, help='max of test images(for debug)')
    parser.add_argument('--resume', type=str, default='', help='checkpoint file(for training or test)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pixel_shuffle', action='store_true',
                        help='uses pixel_shuffle in training')
    args = parser.parse_args()

    net = VanillaUNet().cuda()
    trainset = Precious(simple_transform)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

    dataset = trainset.__getitem__(140)
    print(dataset)


if __name__ == '__main__':
    main()
