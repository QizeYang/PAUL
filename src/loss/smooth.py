import torch


class PatchMemory(object):

    def __init__(self, momentum=0.1, num=1):

        self.name = []
        self.agent = []
        self.momentum = momentum
        self.num = num


    def get_soft_label(self, path, feat_list):

        feat = torch.stack(feat_list, dim=0)
        feat = feat[:, ::self.num, :]
        position = []


        # update the agent
        for j,p in enumerate(path):

            current_soft_feat = feat[:, j, :].detach()
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()
            key  = p
            if key not in self.name:
                self.name.append(key)
                self.agent.append(current_soft_feat)
                ind = self.name.index(key)
                position.append(ind)

            else:
                ind = self.name.index(key)
                tmp = self.agent.pop(ind)
                tmp = tmp*(1-self.momentum) + self.momentum*current_soft_feat
                self.agent.insert(ind, tmp)
                position.append(ind)

        if len(position) != 0:
            position = torch.tensor(position).cuda()

        agent = torch.stack(self.agent, dim=1).cuda()
        return agent, position


class SmoothingForImage(object):
    def __init__(self, momentum=0.1, num=1):

        self.map = dict()
        self.momentum = momentum
        self.num = num


    def get_soft_label(self, path, feature):

        feature = torch.cat(feature, dim=1)
        soft_label = []

        for j,p in enumerate(path):

            current_soft_feat = feature[j*self.num:(j+1)*self.num, :].detach().mean(dim=0)
            if current_soft_feat.is_cuda:
                current_soft_feat = current_soft_feat.cpu()

            key  = p
            if key not in self.map:
                self.map.setdefault(key, current_soft_feat)
                soft_label.append(self.map[key])
            else:
                self.map[key] = self.map[key]*(1-self.momentum) + self.momentum*current_soft_feat
                soft_label.append(self.map[key])
        soft_label = torch.stack(soft_label, dim=0).cuda()
        return soft_label



