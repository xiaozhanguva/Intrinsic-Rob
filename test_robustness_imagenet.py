import waitGPU
waitGPU.wait(utilization=40, available_memory=8000, interval=20)

from generative.biggan import BigGAN128
from train_classifier.attack import PGDAttack, CarliniWagnerL2_indist
import train_classifier.problem as pblm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import argparse
import os

class Generator():
    def __init__(self):
        self.class_labels = [404, 609, 11, 281, 351, 153, 30, 339, 724, 717]
        self.model_path = './generative/models/ImageNet/BigGAN/biggan128-release.pt'
        self.G = BigGAN128().cuda()
        self.G.load_state_dict(torch.load(self.model_path))
        self.G.eval()

    def gen(self, zs, ys):
        labels = torch.tensor([self.class_labels[label] for label in ys]).cuda()
        imgs = self.G(zs, labels)
        imgs = 0.5 * (imgs + 1)
        imgs = torch.nn.functional.interpolate(imgs, size=(32, 32))
        return imgs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test robustness for ImageNet10')
    # general model settings    
    parser.add_argument('--method', default='madry', help='training method', 
                        choices=['baseline', 'madry', 'trades'])
    parser.add_argument('--eval-epoch', type=int, default=100, help='eval epoch')
    parser.add_argument('--batch-size', type=int, default=40, help='test batch size')
    parser.add_argument('--n-test', type=int, default=10, help='number of testing sets')
    # for generating adversarial examples
    parser.add_argument('--robust-type', default='unc', help='in-dist/unc robustness',
                        choices=['in', 'unc'])
    parser.add_argument('--epsilon', type=float, default=3.0, help='perturbation strength')
    parser.add_argument('--step-size', type=float, default=0.5, help='attack step size')
    parser.add_argument('--steps', default=100, help='perturb number of steps')
    # for finding in dist adversarial exps
    parser.add_argument('--z-dim', default=120, help='latent space dimension')
    parser.add_argument('--bin-steps', default=4, help='binary search steps')
    # other settings
    parser.add_argument('--seed', type=int, default=141, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--verbose', default=20, help='display interval')
    parser.add_argument('--save-dir', default='./result/ImageNet10', 
                        help='directory for saving all the statistics')
    args = parser.parse_args()

    save_dir = args.save_dir + '/' + args.robust_type
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(args)

    mypath = './train_classifier/checkpoint/ImageNet10/' + args.method
    prefix = 'model-imagenet10-epoch'
    checkpoint = os.path.join(mypath, prefix + str(args.eval_epoch) + '.pt')

    model = pblm.BigCNN().to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print(checkpoint)

    # nat_acc = []
    # rob_acc = []

    if args.robust_type == 'unc':
        attacker = PGDAttack(epsilon=args.epsilon, num_steps=args.steps, step_size=args.step_size)


        X, Y, Z = torch.load('./data/ImageNet/BigGAN/testset_with_z.pt')
        robust_err_total = 0
        natural_err_total = 0
        total = 0
        n = len(X)

        for i in range(n // args.batch_size):
            data = X[i * args.batch_size: min((i + 1) * args.batch_size, n)]
            labels = Y[i * args.batch_size: min((i + 1) * args.batch_size, n)].squeeze()
            zs = Z[i * args.batch_size: min((i + 1) * args.batch_size, n)]
            data, labels, zs = data.to(device), labels.to(device), zs.to(device)

            err_natural, err_robust = attacker.attack(model, data, labels)

            robust_err_total += err_robust
            natural_err_total += err_natural
            total += len(data)

            if args.verbose and i % args.verbose == 0: 
                print('Batch: [{0:3d}/{1}]\t\t' 'Error: {err:.2%}\t\t'
                      'Robust Error (unc): {rob_err:.2%}'.format(
                        i+1, n//args.batch_size, 
                        err=natural_err_total/total, 
                        rob_err=robust_err_total/total))

        nat_acc = 1 - natural_err_total/total
        rob_acc = 1 - robust_err_total/total

        f1 = open(save_dir + '/' + args.method + '_nat_acc.txt', 'w')
        print(nat_acc, file=f1)
        f2 = open(save_dir + '/' + args.method + '_eps' + str(args.epsilon) + '_rob_acc.txt', 'w')
        print(rob_acc, file=f2)

    else:
        attacker = CarliniWagnerL2_indist(generator=Generator,
                                          device=device,
                                          epsilon=args.epsilon,
                                          z_dim=args.z_dim,
                                          init_scheme='naive',
                                          z_init = None,
                                          search_steps=args.bin_steps,
                                          max_iterations=100,
                                          learning_rate=0.01,
                                          initial_const=1)

        X, Y, Z = torch.load('./data/ImageNet/BigGAN/testset_with_z.pt')
        natural_err_total = 0
        robust_err_total = 0
        total = 0
        n = len(X)

        for i in range(n // args.batch_size):
            print('==================== Batch: [{0}/{1}] ===================='.format(i+1, n//args.batch_size))
            data = X[i * args.batch_size: min((i + 1) * args.batch_size, n)]
            labels = Y[i * args.batch_size: min((i + 1) * args.batch_size, n)].squeeze()
            zs = Z[i * args.batch_size: min((i + 1) * args.batch_size, n)]
            data, labels, zs = data.to(device), labels.to(device), zs.to(device)

            err_natural, err_robust = attacker.attack(model, data, labels=labels, z_init=zs, targeted=False)

            robust_err_total += err_robust
            natural_err_total += err_natural
            total += len(data)

            print(' * Error: {err:.2%}\t\t' 'Robust Error (in): {rob_err:.2%}'.format(
                    err=natural_err_total/total, rob_err=robust_err_total/total))
            # print('')

        nat_acc = 1 - natural_err_total/total
        rob_acc = 1 - robust_err_total/total
    
        f1 = open(save_dir + '/' + args.method + '_nat_acc.txt', 'w')
        print(nat_acc, file=f1)
        f2 = open(save_dir + '/' + args.method + '_eps' + str(args.epsilon) + '_rob_acc.txt', 'w')
        print(rob_acc, file=f2)


        ## print the results
        print('========== Evaluating ('+args.robust_type+') robustness for '+args.method)
        print(' * Epsilon: {eps:.1f}\t\t' 
              'Nat. Acc. {acc:.2%}\t\t'
              'Rob. Acc. {rob_acc:.2%}'.format(
                  eps=args.epsilon, acc=nat_acc, rob_acc=rob_acc))
