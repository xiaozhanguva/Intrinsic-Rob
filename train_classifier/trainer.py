import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import convex_adversarial
from convex_adversarial.dual_network import robust_loss, RobustBounds
from adv_loss import madry_loss, trades_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import time

DEBUG = False

## standard training
def train_baseline(loader, model, opt, epoch, verbose):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.train()
    print('======================================== standard training')

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda()
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        opt.step()

        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        if verbose and i % verbose == 0: 
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.2%} ({errors.avg:.2%})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors))

    # print(epoch, '{:.2%}'.format(errors.avg), '{:.4f}'.format(losses.avg), file=log)
    # log.flush()  


## certified robust training (zico)
def train_zico(loader, model, opt, epsilon, epoch, verbose, clip_grad=None, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.train()
    print('======================================== certified robust training (zico)')
    print('epsilon:', '{:.4f}'.format(epsilon))
    
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        # if y.dim() == 2: 
        #     y = y.squeeze(1)
        # data_time.update(time.time() - end)

        with torch.no_grad(): 
            ce = nn.CrossEntropyLoss()(model(X), y).item()
            err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        opt.zero_grad()
        robust_ce.backward()

        if clip_grad: 
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        opt.step()

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose and i % verbose == 0: 
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.2%} ({rerrors.avg:.2%})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.2%} ({errors.avg:.2%})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors, rloss = robust_losses, 
                   rerrors = robust_errors), end=endline)

        del X, y, robust_ce, ce, err, robust_err

        if DEBUG and i == 10: 
            break

    # print(epoch, '{:.2%}'.format(errors.avg), '{:.2%}'.format(robust_errors.avg), 
    #         '{:.4f}'.format(robust_losses.avg), file=log)
    # log.flush()  
    torch.cuda.empty_cache()


## adversarial training (madry)
def train_madry(args, loader, model, opt, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()

    model.train()
    print('======================================== adversarial training (madry)')
    
    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.to(device), y.to(device)

        with torch.no_grad(): 
            ce = nn.CrossEntropyLoss()(model(X), y).item()
            err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)
        
        opt.zero_grad()

        # calculate robust loss
        robust_ce = madry_loss(model=model, x_natural=X, y=y, optimizer=opt, 
                               step_size=args.step_size, epsilon=args.epsilon, 
                               perturb_steps=args.num_steps, distance='l_2')

        robust_ce.backward()
        opt.step()

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # print progress
        if args.verbose and i % args.verbose == 0: 
            endline = '\n' if i % args.verbose == 0 else '\r'
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.2%} ({errors.avg:.2%})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors, rloss = robust_losses), end=endline)

        del X, y, robust_ce, ce, err

        if DEBUG and i == 10: 
            break
    torch.cuda.empty_cache()


## adversarial training (trades)
def train_trades(args, loader, model, opt, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()

    model.train()
    print('======================================== adversarial training (trades)')

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.to(device), y.to(device)

        with torch.no_grad(): 
            ce = nn.CrossEntropyLoss()(model(X), y).item()
            err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)
        
        opt.zero_grad()

        # calculate robust loss
        robust_ce = trades_loss(model=model, x_natural=X, y=y,
                           optimizer=opt, step_size=args.step_size,
                           epsilon=args.epsilon, perturb_steps=args.num_steps,
                           beta=args.beta, distance='l_2')

        
        robust_ce.backward()                 
        opt.step()

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.detach().item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        # print progress
        if args.verbose and i % args.verbose == 0: 
            endline = '\n' if i % args.verbose == 0 else '\r'
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {errors.val:.2%} ({errors.avg:.2%})'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   loss=losses, errors=errors, rloss = robust_losses), end=endline)

        del X, y, robust_ce, ce, err

        if DEBUG and i == 10: 
            break
    torch.cuda.empty_cache()

def eval_nat_acc(loader, model, epoch, verbose, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    model.eval()
    print('======================================== evaluating (standard err) ========================================')

    end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.to(device), y.to(device)
        out = model(Variable(X))
        ce = nn.CrossEntropyLoss()(out, Variable(y))
        err = (out.max(1)[1] != y).float().sum()  / X.size(0)

        # measure accuracy and record loss
        losses.update(ce.item(), X.size(0))
        errors.update(err.item(), X.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if verbose and i % verbose == 0: 
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.2%} ({error.avg:.2%})'.format(
                      epoch, i, len(loader), batch_time=batch_time, loss=losses,
                      error=errors))


def eval_zico(loader, model, epsilon, epoch, verbose, **kwargs):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    robust_losses = AverageMeter()
    robust_errors = AverageMeter()

    model.eval()
    print('======================================== evaluating (certified robust err) ========================================')
    print('epsilon:', '{:.4f}'.format(epsilon))
    end = time.time()

    torch.set_grad_enabled(False)
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        robust_ce, robust_err = robust_loss(model, epsilon, X, y, **kwargs)

        ce = nn.CrossEntropyLoss()(model(X), y).item()
        err = (model(X).max(1)[1] != y).float().sum().item() / X.size(0)

        # _,pgd_err = _pgd(model, Variable(X), Variable(y), epsilon)

        # measure accuracy and record loss
        losses.update(ce, X.size(0))
        errors.update(err, X.size(0))
        robust_losses.update(robust_ce.item(), X.size(0))
        robust_errors.update(robust_err, X.size(0))

        # measure elapsed time
        batch_time.update(time.time()-end)
        end = time.time()

        if verbose and i % verbose == 0: 
            # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
            endline = '\n' if i % verbose == 0 else '\r'
            print('Epoch: [{:2d}][{:3d}/{}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Robust loss {rloss.val:.4f} ({rloss.avg:.4f})\t'
                  'Robust error {rerrors.val:.2%} ({rerrors.avg:.2%})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Error {error.val:.2%} ({error.avg:.2%})'.format(
                      epoch, i, len(loader), batch_time=batch_time, 
                      loss=losses, error=errors, rloss = robust_losses, 
                      rerrors = robust_errors), end=endline)
        
        del X, y, robust_ce, ce, err, robust_err

        if DEBUG and i == 10: 
            break
            
    print(' * Error: {error.avg:.2%}\n'
          ' * Robust error: {rerror.avg:.2%}'.format(
              error=errors, rerror=robust_errors))
    
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

