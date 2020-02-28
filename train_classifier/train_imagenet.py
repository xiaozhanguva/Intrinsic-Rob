import waitGPU
waitGPU.wait(utilization=40, available_memory=8000, interval=20)

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from problem import *
from trainer import *

def adjust_learning_rate(optimizer, epoch, lr_init):
    """decrease the learning rate"""
    lr = lr_init
    if epoch >= 50:
        lr = lr * 0.1
    if epoch >= 70:
        lr = lr * 0.01
    if epoch >= 90:
        lr = lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='ImageNet10 Adversarial Training')
    # general settings for training
    parser.add_argument('--method', type=str, default='madry', help='training method',
                        choices=['baseline', 'madry', 'trades'])
    parser.add_argument('--epsilon', type=float, default=3.0, help='perturbation strength')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--batch-size-test', type=int, default=50, help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=2e-4, help='SGD decay factor')
    # other general settings
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA ')
    parser.add_argument('--seed', default=1, help='random seed')
    parser.add_argument('--verbose', default=100, help='display interval')
    parser.add_argument('--load-dir', default='../data/ImageNet/BigGAN', 
                        help='directory for loading BigGAN-generated-ImageNet10 data')
    parser.add_argument('--model-dir', default='./checkpoint/ImageNet10', 
                        help='directory for saving intermediate models')
    parser.add_argument('--save-freq', default=5, type=int, help='saving frequency')
    
    # specific for adv training
    parser.add_argument('--num-steps', default=10, help='perturb number of steps')
    parser.add_argument('--step-size', default=0.5, help='perturb step size')
    parser.add_argument('--beta', default=6.0, help='regularization, i.e., 1/lambda in TRADES')
    args = parser.parse_args()
    print(args)

    model_dir = args.model_dir+'/'+args.method
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # load custom ImageNet10 data and setup dataloaders
    train_loader = custom_cifar_loaders(batch_size=args.batch_size,
                                        path=args.load_dir+'/train.npz',
                                        is_shuffle=True)
    test_loader = custom_cifar_loaders(batch_size=args.batch_size_test,
                                       path=args.load_dir+'/test.npz',
                                       is_shuffle=False)

    model = BigCNN().to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=args.momentum, weight_decay=args.weight_decay)

    for t in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(opt, t, args.lr)

        if args.method == 'baseline':   
            train_baseline(train_loader, model, opt, t, args.verbose)

        elif args.method == 'madry':
            train_madry(args, train_loader, model, opt, t, device)

        else:
            train_trades(args, train_loader, model, opt, t, device)

        # evaluation on natural examples
        eval_nat_acc(test_loader, model, t, args.verbose, device)

        # save intermiedate models
        if t % args.save_freq == 0:
            torch.save(model.state_dict(),
                    os.path.join(model_dir, 'model-imagenet10-epoch{}.pt'.format(t)))