import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

## training objective for madry's adversarial training
def madry_loss(model, x_natural, y, optimizer, 
               step_size=0.003, epsilon=0.031, 
               perturb_steps=10, distance='l_inf'):

    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss_kl = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm + 1e-8).unsqueeze(2).unsqueeze(3).expand_as(x_natural)
            x_adv = x_adv.detach() + step_size * grad_normalized
            eta_x_adv = x_adv - x_natural
            norm_eta = torch.norm(eta_x_adv.view(batch_size, -1), p=2, dim=1, keepdim=True)
            norm_eta = torch.clamp(norm_eta, epsilon, np.inf).unsqueeze(2).unsqueeze(3).expand_as(x_natural)
            x_adv = x_natural + eta_x_adv * epsilon / norm_eta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    return loss

## training objective for adversarial training by TRADES
def trades_loss(model, x_natural, y, optimizer, 
                step_size=0.003, epsilon=0.031, 
                perturb_steps=10, beta=1.0, distance='l_inf'):
    model.eval()
    criterion_kl = nn.KLDivLoss(reduction='sum')
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif distance == 'l_2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]

            grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1, keepdim=True)
            grad_normalized = grad / (grad_norm + 1e-8).unsqueeze(2).unsqueeze(3).expand_as(x_natural)
            x_adv = x_adv.detach() + step_size * grad_normalized
            eta_x_adv = x_adv - x_natural
            norm_eta = torch.norm(eta_x_adv.view(batch_size, -1), p=2, dim=1, keepdim=True)
            norm_eta = torch.clamp(norm_eta, epsilon, np.inf).unsqueeze(2).unsqueeze(3).expand_as(x_natural)
            x_adv = x_natural + eta_x_adv * epsilon / norm_eta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()
