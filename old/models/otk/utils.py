# -*- coding: utf-8 -*-
import torch
import math
import random
import numpy as np

EPS = 1e-6


def normalize(x, p=2, dim=-1, inplace=False):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    if inplace:
        x.div_(norm.clamp(min=EPS))
    else:
        x = x / norm.clamp(min=EPS)
    return x

def spherical_kmeans(x, n_clusters, max_iters=100, block_size=None, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x kmer_size x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    if x.ndim == 3:
        n_samples, kmer_size, n_features = x.size()
    else:
        n_samples, n_features = x.size()
    if init is None:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf
    tmp = x.new_empty(n_samples)
    assign = x.new_empty(n_samples, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = x.shape[0]

    for n_iter in range(max_iters):
        for i in range(0, n_samples, block_size):
            end_i = min(i + block_size, n_samples)
            cos_sim = x[i: end_i].view(end_i - i, -1).mm(clusters.view(n_clusters, -1).t())
            tmp[i: end_i], assign[i: end_i] = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm(dim=-1, keepdim=True).clamp(min=EPS)

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters
