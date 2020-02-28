import waitGPU
waitGPU.wait(utilization=40, available_memory=8000, interval=20)

from generative.biggan import BigGAN128
import generative.utils as utils

import wget
import torch
import numpy as np
import imageio
import os
import argparse

class ImageNetGenerator():
    def __init__(self, args):
        self.dataset = args.dataset
        self.gan_type = args.gan_type
        self.save_dir = args.save_dir
        self.train_parts = args.train_parts
        self.test_parts = args.test_parts
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.seed = args.seed
        self.n_samples = args.n_samples
        self.n_neighbors = args.n_neighbors
        self.radius = args.radius
        self.n_parts = args.n_parts

        self.z_dim = 120
        self.class_labels = [404, 609, 11, 281, 351, 153, 30, 339, 724, 717]
        self.class_num = 10
        self.sample_num = self.class_num ** 2
        self.gpu_mode = True

        # download pretrained biggan128 model from url
        url = 'https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases/download/v0.0.0/biggan128-release.pt'
        self.pretrain_model = os.path.join(self.save_dir, 'biggan128-release.pt')
        if not os.path.isfile(self.pretrain_model):
            wget.download(url, self.pretrain_model)     # if not working, manually download from https://github.com/ivclab/BigGAN-Generator-Pretrained-Pytorch/releases

        self.G = BigGAN128().cuda()
        self.G.load_state_dict(torch.load(self.pretrain_model))
        self.G.eval()

        # fixed noise & condition for generating images
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i*self.class_num] = torch.randn(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i*self.class_num + j] = self.sample_z_[i*self.class_num]

        temp_y = torch.zeros(self.sample_num)
        for i in range(self.class_num):
            temp_y[i*self.class_num: (i+1)*self.class_num] = torch.tensor(self.class_labels)
        self.sample_y_ = temp_y.type(torch.LongTensor)

        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def generate_image(self, zs, ys):
        with torch.no_grad():
            imgs = self.G(zs, ys)
        imgs = 0.5 * (imgs.data + 1)
        imgs = torch.nn.functional.interpolate(imgs, size=(32, 32))
        return imgs

    def get_local_lipschitz(self, sample_z, sample_y, n_neighbors, n_parts, gpu_mode, z_dim=100, radius=1):
        log_diff = torch.zeros(n_neighbors*n_parts)
        if gpu_mode:
            log_diff = log_diff.cuda()

        # avoid gpu memory overflow (would improve efficiency if computing parallelly)
        for k in range(n_parts):
            z_repeat = sample_z.repeat(n_neighbors, 1)
            y_repeat = sample_y.repeat(n_neighbors, 1).squeeze()

            # generate uniform noise from l2 ball
            v = torch.nn.functional.normalize(torch.rand(n_neighbors, z_dim), p=2, dim=1)
            u = torch.rand(n_neighbors) + 1e-12  # avoid underflow
            unif_noise = (radius * u ** (1 / float(z_dim))).unsqueeze(1) * v

            if gpu_mode:
                unif_noise, z_repeat, y_repeat = unif_noise.cuda(), z_repeat.cuda(), y_repeat.cuda()

            G_z = self.generate_image(z_repeat, y_repeat).view(n_neighbors, -1)
            G_z_neighbors = self.generate_image(z_repeat + unif_noise, y_repeat).view(n_neighbors, -1)

            # roughly estimate lipschitz using samples
            dist_z = torch.sqrt(torch.sum(unif_noise ** 2, dim=1))
            dist_x = torch.sqrt(torch.sum((G_z_neighbors - G_z) ** 2, dim=1))
            
            inds = range(k*n_neighbors, (k+1)*n_neighbors) 
            log_diff[inds] = torch.log(dist_x) - torch.log(dist_z)

        lipschitz = torch.exp(torch.max(log_diff)).cpu().item()
        return lipschitz

    def get_lipschitz(self):
        log_dir = 'generative/models/' + self.dataset + '/' + self.gan_type 
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log = open(log_dir + '/lipschitz.txt', "w")
        L = np.zeros((self.class_num, self.n_samples))

        for i in range(self.class_num):
            for j in range(self.n_samples):

                sample_z = torch.randn((1, self.z_dim))
                sample_y = torch.tensor([self.class_labels[i]])
                L[i, j] = self.get_local_lipschitz(sample_z, sample_y, self.n_neighbors,
                                                   self.n_parts,self.gpu_mode, self.z_dim, self.radius)
                if j % 20 == 0:
                    print('Class: {0}\t\t' 'Sample: [{1}/{2}]\t\t' 'Lipschitz: {lip:.2f}'.format(
                            i, j, self.n_samples, lip=L[i, j]))

            # print the results
            for i in range(self.class_num):
                print(""" * class: %d, (95%%) lipschitz: %.2f, (99%%) lipschitz: %.2f, (99.9%%) lipschitz: %.2f, (max) lipschitz: %.2f""" 
                        % (i, np.percentile(L[i,:], q=95), np.percentile(L[i,:], q=99),
                           np.percentile(L[i,:], q=99.9), np.max(L[i,:])))
            
        for i in range(self.class_num):
            print("%d, %.2f, %.2f, %.2f, %.2f" 
                    % (i, np.percentile(L[i,:], q=95), np.percentile(L[i,:], q=99),
                       np.percentile(L[i,:], q=99.9), np.max(L[i,:])), file=log)
            log.flush()

    def reconstruct_dataset(self):
        ## generate sample images using fixed noise
        samples = self.generate_image(self.sample_z_, self.sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        utils.save_images(samples[:100, :, :, :], [10, 10], self.save_dir+'/gen_img.png')

        ## reconstruct the training and testing dataset
        # for training 
        data_dir = './data/' + self.dataset + '/' + self.gan_type
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        print('========== generate training dataset')
        torch.manual_seed(self.seed)

        for k in range(self.train_parts):       # avoid memory overflow
            sample_z = torch.randn((self.train_size, self.z_dim))
            labels = torch.randint(0, self.class_num, (self.train_size, 1)).type(torch.LongTensor)
            sample_y = torch.tensor([self.class_labels[label] for label in labels])

            if self.gpu_mode:
                sample_z, sample_y = sample_z.cuda(), sample_y.cuda()

            samples = self.generate_image(sample_z, sample_y)

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k == 0:
                labels_train = labels
                samples_train = samples
            else:
                labels_train = np.concatenate((labels_train, labels), axis=0)
                samples_train = np.concatenate((samples_train, samples), axis=0)
            print('train part ', k, ' done')

        np.savez(data_dir + '/train', sample=samples_train, label=labels_train.squeeze(1))

        # for testing 
        torch.manual_seed(self.seed+999)

        for k in range(self.test_parts):
            sample_z = torch.randn((self.train_size, self.z_dim))
            labels = torch.randint(0, self.class_num, (self.train_size, 1)).type(torch.LongTensor)
            sample_y = torch.tensor([self.class_labels[label] for label in labels])

            if self.gpu_mode:
                sample_z, sample_y = sample_z.cuda(), sample_y.cuda()

            samples = self.generate_image(sample_z, sample_y)

            if k == 0:
                labels_t = labels
                samples_t = samples
                z_t = sample_z
            else:
                labels_t = torch.cat((labels_t, labels), dim=0)
                samples_t = torch.cat((samples_t, samples), dim=0)
                z_t = torch.cat((z_t, sample_z), dim=0)

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k == 0:
                labels_test = labels
                samples_test = samples
            else:
                labels_test = np.concatenate((labels_test, labels), axis=0)
                samples_test = np.concatenate((samples_test, samples), axis=0)
            print('test part ', k, ' done')

            torch.save([samples_t, labels_t, z_t], data_dir + '/testset_with_z.pt')
            np.savez(data_dir + '/test', sample=samples_test, label=labels_test.squeeze(1))

def parse_args():
    """parsing and configuration"""
    parser = argparse.ArgumentParser(description="Generate ImageNet10 using BigGAN")

    # for training generative model
    parser.add_argument('--gan_type', type=str, default='BigGAN', help='The type of GAN')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The name of dataset')
    parser.add_argument('--mode', type=str, default='evaluate', choices=['evaluate', 'reconstruct'])
    parser.add_argument('--save_dir', type=str, default='./generative/models/ImageNet/BigGAN',
                        help='Directory name to save the model')

    # for calculating local lipschitz constant  (would more efficient if replace partition by parallel computing)
    parser.add_argument('--radius', type=float, default=0.5, help='latent space ball radius')
    parser.add_argument('--n_samples', type=int, default=500, help='number of natural samples')
    parser.add_argument('--n_neighbors', type=int, default=100, help='number of neighboring points')
    parser.add_argument('--n_parts', type=int, default=10, help='number of partitions for neighbors')

    # for reconstructing dataset
    parser.add_argument('--seed', type=int, default=141, help='manual seed number')
    parser.add_argument('--train_parts', type=int, default=500, help='number of partitions for training set')
    parser.add_argument('--test_parts', type=int, default=100, help='number of partitions for test set')
    parser.add_argument('--train_size', type=int, default=100, help='number of training samples')
    parser.add_argument('--test_size', type=int, default=100, help='number of testing samples')
    return parser.parse_args()

def main():
    args = parse_args()

    if args is None:
        exit()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    gan = ImageNetGenerator(args)

    if args.mode == 'evaluate':
        print(" [*] Compute the Lipschitz parameter")
        gan.get_lipschitz()
        print("")

    elif args.mode == 'reconstruct':
        print(" [*] Reconstruct " + args.dataset + " dataset using " + args.gan_type)
        gan.reconstruct_dataset()
    else:
        raise Exception("[!] There is no option for " + args.mode)


if __name__ == '__main__':
    main()
