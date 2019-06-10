import torch
import shutil
from .meter import AverageMeter
from .logger import Logger
from .epoch_lr import EpochBaseLR
import os
import scipy.io as sio
import re

def save_checkpoint(state, is_best, exp_name, root='../../snapshot/'):

    path = os.path.join(root, str(exp_name))
    if not os.path.exists(path):
        os.makedirs(path)

    if (state['epoch'] % 10) == 0:
        torch.save(state, path + '/' + str(state['epoch']) + '.pth')
    filename = os.path.join(path, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint.pth', 'best.pth'))




def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_feature(part, exp_name, data, features, labels, paths, cams, root='../../feature'):
    path = os.path.join(root, str(exp_name))
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(path, data +'_'+ part+'.mat'),
                {'feature':features, 'label':labels, 'path':paths, 'cam':cams})

def parse_parameters(model, keywords):
    parameters = []
    for name, param in model.named_parameters():
        if keywords in name:
            parameters.append(param)
    return parameters

def remote_debug():
    import pydevd
    pydevd.settrace('172.0.0.1', port=60000, stdoutToServer=True, stderrToServer=True)
