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
sys.path.append("../../config/pycharm-debug-py3k.egg")
import models
import dataset
from utils import *
import evaluate
import loss


parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--step', default=50, type=int, metavar='N',
                    help=' period of learning rate decay.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate models on validation set')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('-g', '--gpu', type=str, default='0', metavar='G',
                    help='set the ID of GPU')
parser.add_argument('-exp', '--exp-name', type=str, default='0000', help='the name of experiment')
parser.add_argument('--debug', action='store_true', help='use remote debug', default=False)

## data setting
parser.add_argument('--data', choices=dataset.__all__, help='dataset: ' +' | '.join(dataset.__all__) +
                        ' (default: MARKET)', default='MARKET')
parser.add_argument('--num', type=int, default=2, help='the total number of images for each sample')

parser.add_argument('--height', type=int, default=384)
parser.add_argument('--width', type=int, default=128)

## network
parser.add_argument('--net', type=str, default='PatchNetUn', choices=models.__all__, help='nets: ' +' | '.join(models.__all__) +
                        ' (default: PatchNet)')


## loss setting
parser.add_argument('--scale', type=float, default=15)
parser.add_argument('--mm', type=float, default=0.1, help=' the momentum of the memory')
parser.add_argument('--ploss', default=2, type=float, help='the weight of the PEDAL loss')
parser.add_argument('--iloss', default=1, type=float, help='the weight of the IFPL loss')
parser.add_argument('--margin', default=2, type=float, help='the margin of local consistence loss')
## pretrained model
parser.add_argument('--pre-name', default=None, type=str, help='use which pretrained model to initialize the model')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
net = models.__dict__[args.net]
sys.stdout = Logger(os.path.join('../../snapshot', args.exp_name))
def main():

    global args

    print(args)
    if args.evaluate:
        extract()
        evaluate.eval_result(exp_name=args.exp_name, data=args.data)


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
    data = dataset.__dict__[args.data](part='train', size=(args.height, args.width), require_path=True, pseudo_pair=args.num)
    train_loader = torch.utils.data.DataLoader(
        data,
        shuffle=True, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)


    # create models
    if args.pre_name is not None:
        path = os.path.join('../../snapshot', args.pre_name, 'checkpoint.pth')
        pretrained_model = torch.load(path)['state_dict']
        class_num = pretrained_model['module.new.fc_list.1.weight'].size(0)
        print('the class number of pretrained model is {}'.format(class_num))
    else:
        raise RuntimeError('require a pretrained model to initialize the weight')
    model = net(class_num=class_num)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(pretrained_model)

    # evaluate the directly transfer result
    if not os.path.isfile(os.path.join('../snapshot', args.exp_name, 'checkpoint.pth')):
        save_checkpoint({
            'epoch': 0,
            'state_dict': model.state_dict(),
        }, exp_name=args.exp_name, is_best=True)
    #
    extract()
    evaluate.eval_result(exp_name=args.exp_name, data=args.data)

    # define loss function (criterion) and optimizer
    parameters = model.module.get_param(args.lr)
    centers = loss.SmoothingForImage(momentum=args.mm, num=args.num)
    patch_centers = loss.PatchMemory(momentum=args.mm, num=args.num)
    pv_criterion = loss.Ipfl(margin=args.margin, num=args.num).cuda()
    pc_criterion = loss.Pedal(scale=args.scale).cuda()

    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step, gamma=0.1, last_epoch=-1)


    cudnn.benchmark = True

    print(model)
    print(optimizer)

    print('initialize the centers')
    model.train()
    for i, (input, target, path, cams) in enumerate(train_loader):
        # measure data loading time
        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            input = input.view(-1, input.size(2), input.size(3), input.size(4))

            # compute output
            feat_list, embedding_list = model(input)

            patch_centers.get_soft_label(path, feat_list)
    print('initialization done')

    # print(args.start_epoch, args.epochs)
    for epoch in range(args.start_epoch, args.epochs):

        lr_scheduler.step()

        # train for one epoch
        train(train_loader, model, pv_criterion, pc_criterion, optimizer, centers, patch_centers, epoch)

        # save checkpoint

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, exp_name=args.exp_name, is_best=True)

        if (epoch > 10 and epoch % 5 == 0) or epoch in [0, 3, 6, 9, args.epochs - 1]:
            extract()
            evaluate.eval_result(exp_name=args.exp_name, data=args.data)



def train(train_loader, model, pv_criterion, pc_criterion, optimizer, centers, patch_centers, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    i_losses = AverageMeter()
    p_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, path, cams) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        input = input.view(-1, input.size(2), input.size(3), input.size(4))
        feat_list, embedding_list = model(input)

        agent = centers.get_soft_label(path, embedding_list)
        patch_agent, position = patch_centers.get_soft_label(path, feat_list)


        if args.ploss != 0:
            feat = torch.stack(feat_list, dim=0)
            feat = feat[:,::args.num,:]
            ploss = pc_criterion(feat, patch_agent, position)
        else:
            ploss = torch.tensor([0.]).cuda()

        if args.iloss != 0:
            all_embedding = torch.cat(embedding_list, dim=1)
            iloss = pv_criterion(all_embedding, agent)
        else:
            iloss = torch.tensor([0.]).cuda()


        total_loss =  args.iloss*iloss + args.ploss*ploss


        i_losses.update(iloss.item(), input.size(0))
        p_losses.update(ploss.item(), input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'pedal {pedal.val:.4f} ({pedal.avg:.4f})\t'
                  'ipfl {ipfl.val:.4f} ({ipfl.avg:.4f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, ipfl=i_losses, pedal=p_losses))


def extract():

    model = net(is_for_test=True)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(os.path.join('../../snapshot', args.exp_name, 'checkpoint.pth'))
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(args.start_epoch)
    # switch to evaluate mode
    model.eval()
    part = ['query', 'gallery']

    for p in part:
        val_loader = torch.utils.data.DataLoader(
            dataset.__dict__[args.data](part=p,  require_path=True, size=(args.height, args.width)),
            batch_size=args.batch_size*args.num, shuffle=False,
            num_workers=args.workers, pin_memory=True)

        with torch.no_grad():
            paths = []
            for i, (input, target, path, cam) in enumerate(val_loader):

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                feat_list = model(input)

                feat = torch.cat(feat_list, dim=1)

                feature = feat.cpu()
                target = target.cpu()

                nd_label = target.numpy()
                nd_feature = feature.numpy()
                if i == 0:
                    all_feature = nd_feature
                    all_label = nd_label
                    all_cam = cam.numpy()
                else:
                    all_feature = numpy.vstack((all_feature, nd_feature))
                    all_label = numpy.concatenate((all_label, nd_label))
                    all_cam = numpy.concatenate((all_cam, cam.numpy()))
                paths.extend(path)
            all_label.shape = (all_label.size, 1)
            all_cam.shape = (all_cam.size, 1)
            print(all_feature.shape, all_label.shape, all_cam.shape)
            save_feature(p, args.exp_name, args.data, all_feature, all_label, paths, all_cam)


if __name__ == '__main__':
    if args.debug:
        remote_debug()
    main()
