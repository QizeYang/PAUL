import torch
import scipy.spatial.distance as distance
def cosine(query, gallery):
    query = torch.from_numpy(query)
    gallery = torch.from_numpy(gallery)

    m, n = query.size(0), gallery.size(0)
    dist = 1 - torch.mm(query, gallery.t())/((torch.norm(query, 2, dim=1, keepdim=True).expand(m, n)
                                              *torch.norm(gallery, 2, dim=1, keepdim=True).expand(n, m).t()))
    return dist.numpy()

def euclidean(query, gallery):
    query = torch.from_numpy(query)
    gallery = torch.from_numpy(gallery)

    m, n = query.size(0), gallery.size(0)
    dist = torch.pow(query, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(gallery, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, query, gallery.t())
    return dist.numpy()



if __name__ == '__main__':
    a = torch.rand((3, 3))
    b = torch.rand((4, 3))
    print(a,b)
    print(euclidean(a.numpy(),b.numpy()))
    import scipy.spatial.distance as distance
    print(distance.cdist(a.numpy(), b.numpy(), 'euclidean'))

