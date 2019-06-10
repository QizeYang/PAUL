import numpy
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import sys

sys.path.append("..")

import models
import dataset
from utils import *
import evaluate
import loss


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=60, type=int, metavar='4N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=48, type=int,
                    metavar='N', help='mini-batch size (default: 48)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 500)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='G',
                    help='set the ID of GPU')
parser.add_argument('-exp', '--exp-name', type=str, default='0000', help='the name of the experiment')
parser.add_argument('--debug', action='store_true', help='use remote debug', default=False)


parser.add_argument('--net', type=str, default='PatchNet', choices=models.__all__, help='nets: ' +' | '.join(models.__all__) +
                                                                                   ' (default: PatchNet)')

## data setting
parser.add_argument('--data', choices=dataset.__all__, help='dataset: ' +' | '.join(dataset.__all__) +
                                                            ' (default: MSMT17Extra)', default='MSMT17Extra')
parser.add_argument('--height', type=int, default=384)
parser.add_argument('--width', type=int, default=128)
## loss setting

parser.add_argument('--tloss', default=0.1, type=float, help='the weight of the triplet margin loss')
parser.add_argument('--margin', default=1, type=float, help='the margin of the triplet margin loss')


best_prec1 = 0
args = parser.parse_args()
sys.stdout = Logger(os.path.join('../../snapshot', args.exp_name))
print(args)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
net = models.__dict__[args.net]

def main():

    global args, best_prec1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    data = dataset.__dict__[args.data](root=args.data_dir, part='train', size=(args.height, args.width),
                                       require_path=True, true_pair=True)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)



    # create models

    model = net(class_num=data.class_num)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    patch_parameters = parse_parameters(model, 'patch')
    new_parameters = parse_parameters(model, 'new')
    old_parameters = parse_parameters(model, 'backbone')
    criterion = nn.CrossEntropyLoss().cuda()
    tl_criterion = loss.TripletHard(margin=args.margin, norm=True).cuda()
    optimizer = [torch.optim.SGD(patch_parameters, args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay),
                 torch.optim.SGD(new_parameters, args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay),
                 torch.optim.SGD(old_parameters, args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)]

    lr_scheduler = [EpochBaseLR(optimizer[0], [10, 25, 40], [0, 0, 1e-5, 0], last_epoch=-1), # 35
                    EpochBaseLR(optimizer[1], [40], [0.1, 0.01], last_epoch=-1),
                    EpochBaseLR(optimizer[2], [40], [0.01, 0.001], last_epoch=-1)]
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print(model)


    for epoch in range(args.start_epoch, args.epochs):

        for scheduler in lr_scheduler:
            scheduler.step(epoch)
        print(optimizer)

        # train for one epoch
        train(train_loader, model, criterion, tl_criterion, optimizer, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, exp_name=args.exp_name, is_best=True)



def train(train_loader, model, criterion, tl_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    TLlosses = AverageMeter()
    top1 = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _, cams, positive, cams2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        positive = positive.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.repeat(2)

        input = torch.cat([input, positive], dim=0)


        # compute output
        logits_list, feat_list, embedding_list = model(input)
        loss = torch.sum(torch.stack([criterion(logits, target) for logits in logits_list], dim=0))

        all_feat = torch.cat(embedding_list, dim=1)

        if args.tloss == 0:
            tl_loss = torch.tensor([0], dtype=torch.float).cuda()
        else:
            tl_loss =  args.tloss*tl_criterion(all_feat, target)

        total_loss = loss + tl_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_list[0], target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        TLlosses.update(tl_loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))


        # compute gradient and do SGD step
        for o in optimizer:
            o.zero_grad()
        total_loss.backward()
        for o in optimizer:
            o.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'TLLoss {TLloss.val:.4f} ({TLloss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, TLloss=TLlosses, top1=top1))



if __name__ == '__main__':
    if args.debug:
        remote_debug()
    main()
