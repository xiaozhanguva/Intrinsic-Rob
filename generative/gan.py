import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch, time, utils
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x


class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)
        return x

class GAN(object):
    def __init__(self, args):
        # train parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        # self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 100
        # evaluate parameters
        self.n_samples = args.n_samples
        self.n_neighbors = args.n_neighbors
        self.radius = args.radius

        # reconstruct parameters
        self.manual_seed = args.seed
        self.train_size = args.train_size
        self.test_size = args.test_size
        self.train_parts = args.train_parts

        # load dataset
        self.data_loader = mnist_loader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()

        # fixed Gaussian noise
        self.sample_z_ = torch.randn((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.randn((self.batch_size, self.z_dim))
                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)

                G_ = self.G(z_)
                D_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)
                D_fake = self.D(G_)
                G_loss = self.BCE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        self.save()

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random Gaussian noise """
            sample_z_ = torch.randn((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        # torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)
        generator_path = os.path.join(save_dir, self.model_name + '_G.pkl')
        # discriminator_path = os.path.join(save_dir, self.model_name + '_D.pkl')

        if not os.path.exists(generator_path):
            raise ValueError("generative model doesn't exist, need to train first")
        else:
            self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
            # self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

    def get_lipschitz(self):
        self.G.eval()
        self.load()

        log = open(self.save_dir+'/'+self.dataset+'/'+self.model_name+'/lipschitz.txt', "w")
        L = np.zeros(self.n_samples)

        for j in range(self.n_samples):
            sample_z = torch.randn((1, self.z_dim)).repeat(self.n_neighbors,1)

            # generate uniform noise from l2 ball
            v = f.normalize(torch.randn(self.n_neighbors, self.z_dim), p=2, dim=1)
            u = torch.rand(self.n_neighbors) + 1e-12    # avoid underflow
            unif_noise = (self.radius * u ** (1/float(self.z_dim))).unsqueeze(1)*v

            if self.gpu_mode:
                sample_z, unif_noise = sample_z.cuda(), unif_noise.cuda()

            G_z = self.G(sample_z).view(self.n_neighbors, -1)
            G_z_neighbors = self.G(sample_z+unif_noise).view(self.n_neighbors, -1)

            # roughly estimate lipschitz using samples 
            dist_z = torch.sqrt(torch.sum(unif_noise**2, dim=1))
            dist_x = torch.sqrt(torch.sum((G_z_neighbors-G_z)**2, dim=1))
            log_diff = torch.log(dist_x) - torch.log(dist_z)
            L[j] = torch.exp(torch.max(log_diff)).cpu().item()


        # print the results for class i
        print("""(95%%) lipschitz: %.2f, (99%%) lipschitz: %.2f, (99.9%%) lipschitz: %.2f, (max) lipschitz: %.2f""" % (
                np.percentile(L, q=95), np.percentile(L, q=99), 
                np.percentile(L, q=99.9), np.max(L)))

        print("%.2f, %.2f, %.2f, %.2f" % (
                np.percentile(L, q=95), np.percentile(L, q=99),
                np.percentile(L, q=99.9), np.max(L)), file=log)
        log.flush()
            
    def reconstruct(self):
        self.G.eval()
        self.load()

        data_dir = 'data/'+self.dataset+'/'+self.model_name
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # for training (partition the training data in case memory overflow)
        torch.manual_seed(self.manual_seed+999)

        for k in range(self.train_parts):
            sample_z = torch.randn((self.train_size, self.z_dim))

            if self.gpu_mode:
                sample_z = sample_z.cuda()

            samples = (self.G(sample_z) + 1) / 2

            if self.gpu_mode:
                samples = samples.cpu().data.numpy()
            else:
                samples = samples.data.numpy()

            if k==0:
                samples_train = samples 
            else:
                samples_train = np.concatenate((samples_train, samples), axis=0)

        np.save(data_dir+'/train', samples_train)

        # for testing
        torch.manual_seed(self.manual_seed+999)
        sample_z_test = torch.randn((self.test_size, self.z_dim))

        if self.gpu_mode:
            sample_z_test = sample_z_test.cuda()

        samples_test = (self.G(sample_z_test) + 1) / 2

        if self.gpu_mode:
            samples_test = samples_test.cpu().data.numpy()
        else:
            samples_test = samples_test.data.numpy()

        np.save(data_dir+'/test', samples_test)

        samples_test = samples_test.transpose(0, 2, 3, 1)
        utils.save_images(samples_test[:100, :, :, :], [10, 10],
                          self.save_dir+'/'+self.dataset+'/'+self.model_name+'/gen_img.png')

# define dataloader for MNIST 
def mnist_loader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.5], std=[0.5])]
                                    )
    data_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    return data_loader