import torch
import torch.nn as nn


class Pedal(nn.Module):

    def __init__(self, scale=10, k=10):
        super(Pedal, self).__init__()
        self.scale =scale
        self.k = k


    def forward(self, feature, centers, position):

        loss = 0
        for p in range(feature.size(0)):
            part_feat = feature[p, :, :]
            part_centers = centers[p, :, :]
            m, n = part_feat.size(0), part_centers.size(0)
            dist_map = part_feat.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                       part_centers.pow(2).sum(dim=1, keepdim=True).expand(n, m).t()
            dist_map.addmm_(1, -2, part_feat, part_centers.t())

            trick = torch.arange(dist_map.size(1)).cuda().expand_as(dist_map)
            neg, _ = dist_map[trick!=position.unsqueeze(dim=1).expand_as(dist_map)].view(dist_map.size(0), -1).sort(dim=1)

            x = ((-1 * self.scale * neg[:, :self.k]).exp().sum(dim=1)).log()
            y = ((-1 * self.scale * neg).exp().sum(dim=1)).log()

            loss += (-x + y).sum().div(feature.size(1))
        loss = loss.div(feature.size(0))

        return loss



class Ipfl(nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-6, max_iter=15, nearest=3, num=2, swap=False):

        super(Ipfl, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.max_iter = max_iter
        self.num = num
        self.nearest = nearest


    def forward(self, feature, centers):

        image_label = torch.arange(feature.size(0) // self.num).repeat(self.num, 1).transpose(0, 1).contiguous().view(-1)
        center_label = torch.arange(feature.size(0) // self.num)
        loss = 0
        size = 0

        for i in range(0, feature.size(0), 1):
            label = image_label[i]
            diff = (feature[i, :].expand_as(centers) - centers).pow(self.p).sum(dim=1)
            diff = torch.sqrt(diff)

            same = diff[center_label == label]
            sorted, index = diff[center_label != label].sort()
            trust_diff_label = []
            trust_diff = []

            # cycle ranking
            max_iter = self.max_iter if self.max_iter < index.size(0) else index.size(0)
            for j in range(max_iter):
                s = centers[center_label != label, :][index[j]]
                l = center_label[center_label != label][index[j]]

                sout = (s.expand_as(centers) - centers).pow(self.p).sum(dim=1)
                sout = sout.pow(1. / self.p)

                ssorted, sindex = torch.sort(sout)
                near = center_label[sindex[:self.nearest]]
                if (label not in near):  # view as different identity
                    trust_diff.append(sorted[j])
                    trust_diff_label.append(l)
                    break

            if len(trust_diff) == 0:
                trust_diff.append(torch.tensor([0.]).cuda())

            min_diff = torch.stack(trust_diff, dim=0).min()

            dist_hinge = torch.clamp(self.margin + same.mean() - min_diff, min=0.0)

            size += 1
            loss += dist_hinge

        loss = loss / size
        return loss


class TripletHard(nn.Module):
    def __init__(self, margin=1.0, p=2, eps=1e-5, swap=False, norm=False):
        super(TripletHard, self).__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.norm = norm
        self.sigma = 3


    def forward(self, feature, label):

        if self.norm:
            feature = feature.div(feature.norm(dim=1).unsqueeze(1))
        loss = 0

        m, n = feature.size(0), feature.size(0)
        dist_map = feature.pow(2).sum(dim=1, keepdim=True).expand(m, n) + \
                   feature.pow(2).sum(dim=1, keepdim=True).expand(n, m).t() + self.eps
        dist_map.addmm_(1, -2, feature, feature.t()).sqrt_()

        sorted, index = dist_map.sort(dim=1)

        for i in range(feature.size(0)):

            same = sorted[i, :][label[index[i, :]] == label[i]]
            diff = sorted[i, :][label[index[i, :]] != label[i]]
            dist_hinge = torch.clamp(self.margin + same[1] - diff.min(), min=0.0)
            loss += dist_hinge

        loss = loss / (feature.size(0))
        return loss
