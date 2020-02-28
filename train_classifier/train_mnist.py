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
    parser = argparse.ArgumentParser(description='MNIST Adversarial Training')
    # general settings for training
    parser.add_argument('--method', type=str, default='madry', help='training method',
                        choices=['baseline', 'zico', 'madry', 'trades'])
    parser.add_argument('--epsilon', type=float, default=3.0, help='perturbation strength')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--batch-size-test', type=int, default=50, help='batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')

    # other general settings
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--seed', default=141, help='random seed')
    parser.add_argument('--verbose', default=100, help='display interval')
    parser.add_argument('--load-dir', default='../data/MNIST/ACGAN', 
                        help='directory for loading ACGAN-generated-MNIST data')
    parser.add_argument('--model-dir', default='./checkpoint/MNIST', 
                        help='directory for saving intermediate models')
    parser.add_argument('--save-freq', default=5, type=int, help='saving frequency')
    
    # specific for adv training
    parser.add_argument('--num-steps', default=40, help='perturb number of steps')
    parser.add_argument('--step-size', default=0.5, help='perturb step size')
    parser.add_argument('--beta', default=6.0, help='regularization, i.e., 1/lambda in TRADES')
    
    # specific for certified robust training (zico)
    parser.add_argument("--starting_epsilon", default=0.05, help='for eps scheduling')
    parser.add_argument('--schedule_length', default=20, help='for eps scheduling')
    parser.add_argument('--proj', default=50, help='number of projections for training')
    parser.add_argument('--norm_train', default='l2_normal', help='norm used for training')
    parser.add_argument('--norm_test', default='l2', help='norm used for testing')
    args = parser.parse_args()

    model_dir = args.model_dir+'/'+args.method
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.method == 'zico':
        args.batch_size = 50
        args.batch_size_test = 10
        args.verbose = 200

    # load custom MNIST data and setup dataloaders
    train_loader = custom_mnist_loaders(batch_size=args.batch_size,
                                        path=args.load_dir+'/train.npz',
                                        is_shuffle=True)
    test_loader = custom_mnist_loaders(batch_size=args.batch_size_test,
                                       path=args.load_dir+'/test.npz',
                                       is_shuffle=False)

    ## model training
    if args.method == 'baseline':       # standard training 
        args.epochs = 60
        args.lr = 1e-3
        kwargs = {'proj' : args.proj} if args.proj is not None else {}
        print(args)

        model = mnist_model().to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)

        # learning rate decay and epsilon scheduling
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        for t in range(1, args.epochs+1):
            # adjust learning rate 
            lr_scheduler.step(epoch=max(t-args.schedule_length, 0))

            train_baseline(train_loader, model, opt, t, args.verbose)
            eval_nat_acc(test_loader, model, t, args.verbose, device)

            # save intermiedate models
            if t % args.save_freq == 0:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, 'model-mnist-epoch{}.pt'.format(t)))

    elif args.method == 'zico':       # certified robust training (zico)
        args.epsilon = 2.0
        args.epochs = 60
        args.lr = 1e-3
        kwargs = {'proj' : args.proj} if args.proj is not None else {}
        print(args)

        model = mnist_model().to(device)
        opt = optim.Adam(model.parameters(), lr=args.lr)

        # learning rate decay and epsilon scheduling
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        eps_schedule = np.linspace(args.starting_epsilon, args.epsilon, args.schedule_length)

        for t in range(1, args.epochs+1):
            if t < len(eps_schedule) and args.starting_epsilon is not None: 
                epsilon = float(eps_schedule[t])
            else:
                epsilon = args.epsilon
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))

            # certified robust training and testing
            train_zico(train_loader, model, opt, epsilon, t, args.verbose, 
                       norm_type=args.norm_train, bounded_input=True, **kwargs)
            eval_zico(test_loader, model, args.epsilon, t, args.verbose, 
                      norm_type=args.norm_test, bounded_input=True, **kwargs)

            # save intermiedate models
            if t % args.save_freq == 0:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, 'model-mnist-epoch{}.pt'.format(t)))

    else:     # adversarial training (madry, trades)
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        print(args)

        model = SmallCNN().to(device)
        opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for t in range(1, args.epochs + 1):
            # adjust learning rate
            adjust_learning_rate(opt, t, args.lr)

            # adversarial training
            if args.method == 'madry':
                train_madry(args, train_loader, model, opt, t, device)
            else:
                train_trades(args, train_loader, model, opt, t, device)

            # evaluation on natural examples
            eval_nat_acc(test_loader, model, t, args.verbose, device)

            # save intermiedate models
            if t % args.save_freq == 0:
                torch.save(model.state_dict(),
                        os.path.join(model_dir, 'model-mnist-epoch{}.pt'.format(t)))

