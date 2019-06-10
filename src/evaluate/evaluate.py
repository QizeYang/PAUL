import sys
import os
import numpy as np
sys.path.append("..")
sys.path.append("../../config/pycharm-debug-py3k.egg")
import utils.measure
from  scipy import io
import utils.metric
from collections import defaultdict

def eval_result(exp_name, data, path='../../result'):

    gallery = '../../feature/' +exp_name + '/'+ data +'_gallery.mat'
    query = '../../feature/'+ exp_name + '/'+ data+'_query.mat'
    gallery = io.loadmat(gallery)
    query = io.loadmat(query)

    dist = utils.metric.cosine(query['feature'], gallery['feature'])

    mAP = utils.measure.mean_ap(dist, query['label'].squeeze(), gallery['label'].squeeze(),
                                query['cam'].squeeze(), gallery['cam'].squeeze())

    print('Mean AP: {:4.2%}'.format(mAP))


    # Compute all kinds of CMC scores
    cmc_configs = {
        'first_match': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)

    }
    cmc_scores = {name: utils.measure.cmc(dist, query['label'].squeeze(), gallery['label'].squeeze(),
                            query['cam'].squeeze(), gallery['cam'].squeeze(), **params)
                  for name, params in cmc_configs.items()}

    if not os.path.exists(os.path.join(path, exp_name)):
        os.makedirs(os.path.join(path, exp_name))

    io.savemat(os.path.join(path, exp_name, data + '_cmc.mat'), cmc_scores)

    cmc_topk = (1, 5, 10, 15, 20)

    print('CMC Scores{:>12}'
          .format('first_match'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.2%}'
              .format(k,cmc_scores['first_match'][k - 1]))


