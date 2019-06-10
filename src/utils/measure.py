from collections import defaultdict
import numpy as np
from sklearn.metrics.base import _average_binary_score
from sklearn.metrics import precision_recall_curve, auc

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

def average_precision_score(y_true, y_score, average="macro",
                            sample_weight=None):
    def _binary_average_precision(y_true, y_score, sample_weight=None):
        precision, recall, thresholds = precision_recall_curve(
            y_true, y_score, sample_weight=sample_weight)
        return auc(recall, precision)

    return _average_binary_score(_binary_average_precision, y_true, y_score,
                                 average, sample_weight=sample_weight)

def cmc(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, topk=100,
                separate_camera_set=False, single_gallery_shot=False, first_match_break=False,
                average=True):
    """
    Args:
            distmat: numpy array with shape [num_query, num_gallery], the
                    pairwise distance between query and gallery samples
            query_ids: numpy array with shape [num_query]
            gallery_ids: numpy array with shape [num_gallery]
            query_cams: numpy array with shape [num_query]
            gallery_cams: numpy array with shape [num_gallery]
            average: whether to average the results across queries
    Returns:
            If `average` is `False`:
                ret: numpy array with shape [num_query, topk]
                is_valid_query: numpy array with shape [num_query], containing 0's and
                1's, whether each query is valid or not
            If `average` is `True`:
                numpy array with shape [topk]
    """

    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros([m, topk])
    is_valid_query = np.zeros(m)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i])|(gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        is_valid_query[i] = 1
        if single_gallery_shot:
            repeat = 100
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[i, k - j] += 1
                    break
                ret[i, k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    ret = ret.cumsum(axis=1)
    if average:
        return np.sum(ret, axis=0) / num_valid_queries
    return ret, is_valid_query


def mean_ap(distmat, query_ids=None, gallery_ids=None, query_cams=None, gallery_cams=None, average=True):
    """
    Args:
            distmat: numpy array with shape [num_query, num_gallery], the
                    pairwise distance between query and gallery samples
            query_ids: numpy array with shape [num_query]
            gallery_ids: numpy array with shape [num_gallery]
            query_cams: numpy array with shape [num_query]
            gallery_cams: numpy array with shape [num_gallery]
            average: whether to average the results across queries
    Returns:
            If `average` is `False`:
                    ret: numpy array with shape [num_query]
                    is_valid_query: numpy array with shape [num_query], containing 0's and
                            1's, whether each query is valid or not
            If `average` is `True`:
                    a scalar
    """

    # -------------------------------------------------------------------------


    m, n = distmat.shape

    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = np.zeros(m)
    is_valid_query = np.zeros(m)
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i])|(gallery_cams[indices[i]] != query_cams[i]))
        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        is_valid_query[i] = 1
        aps[i] = average_precision_score(y_true, y_score)
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    if average:
        return float(np.sum(aps)) / np.sum(is_valid_query)
    return aps, is_valid_query