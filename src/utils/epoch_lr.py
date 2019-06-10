import torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

class EpochBaseLR(_LRScheduler):
    def __init__(self, optimizer, milestones, lrs, last_epoch=-1, ):
        if len(milestones)+1 != len(lrs):
            raise ValueError('The length of milestones must equal to the '
                             ' length of lr + 1. Got {} and {} separately', len(milestones)+1, len(lrs))
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)

        self.milestones = milestones
        self.lrs = lrs
        super(EpochBaseLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.lrs[bisect_right(self.milestones, self.last_epoch)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()

        for g in self.optimizer.param_groups:
            g['lr'] = lr



def test():

    net  = nn.Linear(12,4)
    param_groups = [{'params':net.parameters(), 'lr_mult': 0.1}]

    optimizer = torch.optim.SGD(param_groups, lr=0.01,
                                momentum=0.1,
                                weight_decay=0.9,
                                nesterov=True)
    scheduler = EpochBaseLR(optimizer, milestones=[5, 30, 80], lrs=[0.01, 0.1, 0.01, 0.001])
    for epoch in range(100):
        scheduler.step()


if __name__ == '__main__':
    test()