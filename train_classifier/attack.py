import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from typing import Tuple
import os

class PGDAttack:
    def __init__(self, epsilon=3, num_steps=100, 
                 step_size=0.5, image_constraints=(0, 1)):
                 self.boxmin = image_constraints[0]
                 self.boxmax = image_constraints[1]
                 self.epsilon = epsilon
                 self.num_steps = num_steps
                 self.step_size = step_size

    def grad_proj(self, data, order):
        if order == 'inf':
            return data.sign()
        elif order == '2':
            norm = torch.norm(data.view(len(data), -1), 2, 1, keepdim=True)
            return data / (norm + 1e-32).unsqueeze(2).unsqueeze(3).expand_as(data)

    def eta_proj(self, eta, order, epsilon):
        if order == 'inf':
            return torch.clamp(eta, -epsilon, epsilon)
        elif order == '2':
            norm_eta = torch.norm(eta.view(len(eta), -1), p=2, dim=1, keepdim=True)
            norm_eta = torch.clamp(norm_eta, epsilon, np.inf).unsqueeze(2).unsqueeze(3).expand_as(eta)
            return eta * epsilon / norm_eta

    def attack(self, model, X, y):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum().item()
        X_pgd = X.clone()
        
        for i in range(self.num_steps):
            X_pgd = Variable(X_pgd, requires_grad=True)
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            X_pgd = X_pgd + self.step_size * self.grad_proj(X_pgd.grad.data, '2')
            eta = self.eta_proj(X_pgd - X, '2', self.epsilon)
            X_pgd = X + eta
            X_pgd = torch.clamp(X_pgd, self.boxmin, self.boxmax)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum().item()
        return err, err_pgd

class CarliniWagnerL2_indist:
    """
    Carlini's attack (C&W) in distribution version
    """

    def __init__(self, generator,
                 num_classes: int = 10,
                 epsilon: float = 3,
                 z_dim: int = 100,
                 init_scheme: str = 'naive',
                 z_init = None,
                 confidence: float = 0,
                 learning_rate: float = 0.001,
                 search_steps: int = 9,
                 max_iterations: int = 10000,
                 abort_early: bool = True,
                 initial_const: float = 0.001,
                 device: torch.device = torch.device('cpu'),
                 image_constraints: Tuple[float, float] = (0, 1)) -> None:

        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.repeat = self.binary_search_steps >= 10
        self.boxmin = image_constraints[0]
        self.boxmax = image_constraints[1]
        self.device = device
        self.log_interval = 10

        self.G = generator()
        self.z_dim = z_dim
        self.init_scheme = init_scheme

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, y_vec: torch.Tensor,
              z: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]

        adv_input = self.G.gen(z, y_vec)
        adv_input = torch.clamp(adv_input, self.boxmin, self.boxmax)
        l2 = (adv_input - inputs).view(batch_size, -1).pow(2).sum(1)
        logits = model(adv_input)

        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if targeted:
            # if targeted, optimize for making the other class most likely
            logit_dists = torch.clamp(other - real + self.confidence, min=0)
        else:
            # if non-targeted, optimize for making this class least likely.
            logit_dists = torch.clamp(real - other + self.confidence, min=0)

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss + l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return adv_input.detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach()

    def attack(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, z_init=None,
               targeted: bool = False) -> torch.Tensor:
        """
        Performs the attack of the model for the inputs and labels.
        """
        batch_size = inputs.shape[0]
        out = model(inputs)
        err_count = (out.data.max(1)[1] != labels.data).float().sum().item()

        # initialize the starting z   
        if self.init_scheme == 'iter-opt':    # for MNIST experiments
            y_vec = torch.eye(10)[labels].cuda()
            z_start_best = initialize(self.G, inputs, y_vec, self.z_dim, self.boxmin, self.boxmax)
            z = z_start_best.clone()
            # err_count = err_init.sum().item()
        elif self.init_scheme == 'naive' and z_init is not None:     # for ImageNet10 experiments
            y_vec = labels
            z = z_init

            # err_init = (out.data.max(1)[1] == labels.data).float()
            # err_count = (err_init * 1e10 ** 0.5 < self.epsilon).float().sum().item()
            # print('err init:', err_count)
        else:
            raise ValueError('init scheme not defined properly')

        # set the lower and upper bounds accordingly
        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.initial_const, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_attack = inputs.clone()

        # setup the target variable, we need it to be in one-hot form for the loss function
        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        for outer_step in range(self.binary_search_steps):

            # setup the z variable, this is the variable we are optimizing over
            z = Variable(z, requires_grad=True)

            # setup the optimizer
            optimizer = optim.Adam([z], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound

            prev = float('inf')
            for iter in range(self.max_iterations):
                # perform the attack
                adv, logits, l2, logit_dists, loss = self._step(model, optimizer, inputs, y_vec, z,
                                                                labels, labels_infhot, targeted, CONST)

                # print progress
                if iter % 20 == 0:
                    print('Iter [{0:2d}/{1}]\t\t' 'Logit Loss: {2:.2f}'.format(
                            iter, self.max_iterations, logit_dists.sum().item()))

                # check if we should abort search if we're getting nowhere.
                if self.abort_early and iter % (self.max_iterations // 10) == 0:
                    if loss > prev * 0.9999:
                        print('early stopping')
                        # print(loss.item(), prev.item())
                        break
                    prev = loss

                # adjust the best result found so far
                predicted_classes = (logits - labels_onehot * self.confidence).argmax(1) if targeted else \
                    (logits + labels_onehot * self.confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_attack[o_is_both] = adv[o_is_both]

            # adjust the constant as needed
            adv_found = (best_score == labels) if targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

            adv_count = (o_best_l2 ** 0.5 < self.epsilon).float().sum().item()
            print(' * Step: {0}\t\t' 'Adv Found: {1}'.format(outer_step+1, int(adv_count)))
        err_robust_count = (o_best_l2 ** 0.5 < self.epsilon).float().sum().item()
        return err_count, err_robust_count

## initization best z using iterative optimization 
def initialize(Generator, inputs, y_vec, z_dim, boxmin, boxmax):
    z_start_best = torch.randn(len(inputs), z_dim).cuda()
    dist_best = torch.ones(len(inputs)).cuda() * 1e10
    for _ in range(3):
        z_start = torch.randn(len(inputs), z_dim).cuda()
        for _ in range(1000):
            z_start = Variable(z_start, requires_grad=True)
            opt = optim.SGD([z_start], lr=1e-3, momentum=0.9)
            opt.zero_grad()

            X_gen = Generator.gen(z_start, y_vec)
            X_gen = torch.clamp(X_gen, boxmin, boxmax)
            loss = torch.sum((X_gen - inputs) ** 2)
            loss.backward()
            opt.step()

        dist = torch.sum((X_gen - inputs) ** 2, (1, 2, 3)) ** .5
        for i in range(len(inputs)):
            if dist[i] < dist_best[i]:
                z_start_best[i] = z_start[i]
                dist_best[i] = dist[i]
        print('dist init: {:.4f}'.format(torch.mean(dist_best).cpu().item()))
    return z_start_best 

