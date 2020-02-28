import waitGPU
waitGPU.wait(utilization=40, available_memory=8000, interval=20)

import torch
import argparse
import os

from generative.gan import GAN
from generative.acgan import ACGAN

def parse_args():
    """parsing and configuration"""
    parser = argparse.ArgumentParser(description="Generative Models for MNIST")

    # for training generative model
    parser.add_argument('--gan-type', type=str, default='ACGAN', help='The type of GAN',
                        choices=['GAN', 'ACGAN'])
    parser.add_argument('--dataset', type=str, default='MNIST', help='The name of dataset')
    parser.add_argument('--mode', type=str, default='evaluate', help='Which function to run',
                        choices=['train', 'evaluate', 'reconstruct'])
    
    parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
    parser.add_argument('--batch-size', type=int, default=64, help='The size of batch')
    parser.add_argument('--input-size', type=int, default=28, help='The size of input image')
    parser.add_argument('--channels', type=int, default=1, help='The number of rgb channels')
    parser.add_argument('--save-dir', type=str, default='generative/models',
                        help='Directory name to save the model')
    parser.add_argument('--result-dir', type=str, default='generative/imgs',
                        help='Directory name to save the generated images')

    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gpu-mode', type=bool, default=True)
    parser.add_argument('--benchmark-mode', type=bool, default=True)

    # for calculating local lipschitz constant
    parser.add_argument('--radius', type=float, default=0.5, help='latent space ball radius')
    parser.add_argument('--n-samples', type=int, default=1000, help='number of natural samples')
    parser.add_argument('--n-neighbors', type=int, default=2000, help='number of neighboring points')

    # for reconstructing dataset
    parser.add_argument('--seed', type=int, default=0, help='manual seed number')
    parser.add_argument('--train_parts', type=int, default=6, help='number of partitions for training dataset')
    parser.add_argument('--train-size', type=int, default=10000, help='number of training samples')
    parser.add_argument('--test-size', type=int, default=10000, help='number of testing samples')
    return check_args(parser.parse_args())

def check_args(args):
    """checking arguments"""
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def main():
    # parse arguments
    args = parse_args()

    if args is None:
        exit()

    if args.benchmark_mode:
        torch.backends.cudnn.benchmark = True

    # declare instance for GAN
    if args.gan_type == 'GAN':
        gan = GAN(args)
    elif args.gan_type == 'ACGAN':
        gan = ACGAN(args)
    else:
        raise Exception("[!] There is no option for " + args.gan_type)

    if args.mode == 'train':
        gan.train()
        gan.visualize_results(args.epoch)

    elif args.mode == 'evaluate':
        print(" [*] Estimate the local Lipschitz parameter ")
        gan.get_lipschitz()

    elif args.mode == 'reconstruct':
        print(" [*] Reconstruct " + args.dataset + " dataset using " + args.gan_type)
        gan.reconstruct()

    else:
        raise Exception("[!] There is no option for " + args.mode)

if __name__ == '__main__':
    main()
