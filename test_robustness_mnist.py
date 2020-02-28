import waitGPU
waitGPU.wait(utilization=40, available_memory=8000, interval=20)

from generative.acgan import generator
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
    def __init__(self, z_dim=100, class_num=10, input_size=28):
        self.model_dir = './generative/models/MNIST/ACGAN'
        self.z_dim = z_dim
        self.class_num = class_num
        self.input_size = input_size

        self.G = generator(input_dim=self.z_dim, output_dim=1, input_size=self.input_size, class_num=self.class_num)
        self.G.cuda()
        self.G.load_state_dict(torch.load(os.path.join(self.model_dir, 'ACGAN_G.pkl')))
        self.G.eval()

    def gen(self, z, label):
        return (self.G(z, label) + 1.0) / 2.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test robustness for MNIST')
    # general model settings    
    parser.add_argument('--method', default='madry', help='training method', 
                        choices=['baseline', 'zico', 'madry', 'trades'])
    parser.add_argument('--eval-epoch', type=int, default=100, help='eval epoch')
    parser.add_argument('--batch-size', type=int, default=1000, help='test batch size')
    # for generating adversarial examples
    parser.add_argument('--robust-type', default='unc', help='in-dist/unc robustness',
                        choices=['in', 'unc'])
    parser.add_argument('--epsilon', type=float, default=3.0, help='perturbation strength')
    parser.add_argument('--step-size', type=float, default=0.5, help='attack step size')
    parser.add_argument('--steps', default=100, help='perturb number of steps')
    # for finding in-dist adversarial exps
    parser.add_argument('--z-dim', default=100, help='latent space dimension')
    parser.add_argument('--bin-steps', default=4, help='binary search steps')
    # other settings
    parser.add_argument('--seed', type=int, default=141, help='random seed')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--save-dir', default='./result/MNIST', 
                        help='directory for saving all the statistics')
    args = parser.parse_args()

    save_dir = args.save_dir + '/' + args.robust_type
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(args)

    mypath = './train_classifier/checkpoint/MNIST/' + args.method
    prefix = 'model-mnist-epoch'
    checkpoint = os.path.join(mypath, prefix + str(args.eval_epoch) + '.pt')

    if args.method == 'baseline' or args.method == 'zico':
        model = pblm.mnist_model().to(device)
    else:
        model = pblm.SmallCNN().to(device)
    
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    print(checkpoint)

    # nat_acc = []
    # rob_acc = []

    # model robustness evaluation
    if args.robust_type == 'unc':
        attacker = PGDAttack(epsilon=args.epsilon, 
                             num_steps=args.steps, 
                             step_size=args.step_size)

        test_loader = pblm.custom_mnist_loaders(batch_size=args.batch_size,
                                                path='./data/MNIST/ACGAN/test.npz',
                                                is_shuffle=False)

        robust_err_total = 0
        natural_err_total = 0
        total = 0

        for i, (data,labels) in enumerate(test_loader):
            data, labels = data.to(device), labels.to(device)
            
            err_natural, err_robust = attacker.attack(model, data, labels)

            robust_err_total += err_robust
            natural_err_total += err_natural
            total += len(data)

            print('Batch: [{0:2d}/{1}]\t\t'
                  'Error: {err:.2%}\t\t'
                  'Robust Error (unc): {rob_err:.2%}'.format(
                    i+1, len(test_loader), 
                    err=natural_err_total/total, 
                    rob_err=robust_err_total/total))

        nat_acc = 1 - natural_err_total/total
        rob_acc = 1 - robust_err_total/total

        f1 = open(save_dir + '/' + args.method + '_nat_acc.txt', 'w')
        print(nat_acc, file=f1)
        f2 = open(save_dir + '/' + args.method + '_eps' + str(args.epsilon) + '_rob_acc.txt', 'w')
        print(rob_acc, file=f2)

    else:       # in-dist adversarial robustness
        attacker = CarliniWagnerL2_indist(generator=Generator,
                                          device=device,
                                          epsilon=args.epsilon,
                                          z_dim=args.z_dim,
                                          init_scheme='iter-opt',
                                          search_steps=args.bin_steps,
                                          max_iterations=100,
                                          learning_rate=0.01,
                                          initial_const=1)

        test_loader = pblm.custom_mnist_loaders(batch_size=args.batch_size,
                                                path='./data/MNIST/ACGAN/test.npz',
                                                is_shuffle=False)

        natural_err_total = 0
        robust_err_total = 0
        total = 0

        for i, (data,labels) in enumerate(test_loader):
            print('==================== Batch: [{0}/{1}] ===================='.format(i+1, len(test_loader)))
            data, labels = data.to(device), labels.to(device)

            err_natural, err_robust = attacker.attack(model, data, labels)
            
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