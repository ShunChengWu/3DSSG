# -*- coding: utf-8 -*-
import math
import torch
from .utils import spherical_kmeans,normalize

def sinkhorn(dot, mask=None, eps=1e-03, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    n, in_size, out_size = dot.shape
    if return_kernel:
        K = torch.exp(dot / eps)
    else:
        K = dot
    # K: n x in_size x out_size
    u = K.new_ones((n, in_size))
    v = K.new_ones((n, out_size))
    a = float(out_size / in_size)
    if mask is not None:
        mask = mask.float()
        a = out_size / mask.sum(1, keepdim=True)
    for _ in range(max_iter):
        u = a / torch.bmm(K, v.view(n, out_size, 1)).view(n, in_size)
        if mask is not None:
            u = u * mask
        v = 1. / torch.bmm(u.view(n, 1, in_size), K).view(n, out_size)
    K = u.view(n, in_size, 1) * (K * v.view(n, 1, out_size))
    if return_kernel:
        K = K / out_size
        return (K * dot).sum(dim=[1, 2])
    return K

def log_sinkhorn(K, mask=None, eps=1.0, return_kernel=False, max_iter=100):
    """
    dot: n x in_size x out_size
    mask: n x in_size
    output: n x in_size x out_size
    """
    batch_size, in_size, out_size = K.shape
    def min_eps(u, v, dim):
        Z = (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps
        return -torch.logsumexp(Z, dim=dim)
    # K: batch_size x in_size x out_size
    u = K.new_zeros((batch_size, in_size))
    v = K.new_zeros((batch_size, out_size))
    a = torch.ones_like(u).fill_(out_size / in_size)
    if mask is not None:
        a = out_size / mask.float().sum(1, keepdim=True)
    a = torch.log(a)
    for _ in range(max_iter):
        u = eps * (a + min_eps(u, v, dim=-1)) + u
        if mask is not None:
            u = u.masked_fill(~mask, -1e8)
        v = eps * min_eps(u, v, dim=1) + v
    if return_kernel:
        output = torch.exp(
            (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
        output = output / out_size
        return (output * K).sum(dim=[1, 2])
    K = torch.exp(
        (K + u.view(batch_size, in_size, 1) + v.view(batch_size, 1, out_size)) / eps)
    return K

def multihead_attn(input, weight, mask=None, eps=1.0, return_kernel=False,
                   max_iter=100, log_domain=False, position_filter=None):
    """Comput the attention weight using Sinkhorn OT
    input: n x in_size x in_dim
    mask: n x in_size
    weight: m x out_size x in_dim (m: number of heads/ref)
    output: n x out_size x m x in_size
    """
    n, in_size, in_dim = input.shape
    m, out_size = weight.shape[:-1]
    # K.shape = [n, in_size, m, out_size]
    K = torch.tensordot(normalize(input), normalize(weight), dims=[[-1], [-1]]) #multiply and sum the last dim of two inputs.
    # K.shape = [n, m, in_size, out_size]
    K = K.permute(0, 2, 1, 3)
    if position_filter is not None:
        K = position_filter * K
    # K: [n x m, in_size, out_size]
    K = K.reshape(-1, in_size, out_size)
    # K = K-K.min()
    # K = K/K.max()
    # K: nm x in_size x out_size
    if mask is not None:
        mask = mask.repeat_interleave(m, dim=0)
    if log_domain:
        K = log_sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    else:
        if not return_kernel:
            K = torch.exp(K / eps)
        K = sinkhorn(K, mask, eps, return_kernel=return_kernel, max_iter=max_iter)
    # K: nm x in_size x out_size
    if return_kernel:
        return K.reshape(n, m)
    K = K.reshape(n, m, in_size, out_size)
    if position_filter is not None:
        K = position_filter * K
    # K [n, out_size, m, in_size]
    K = K.permute(0, 3, 1, 2).contiguous()
    return K

def wasserstein_barycenter(x, c, eps=1.0, max_iter=100, sinkhorn_iter=50, log_domain=False):
    """
    x: n x in_size x in_dim
    c: out_size x in_dim
    """
    prev_c = c
    for i in range(max_iter):
        T = attn(x, c, eps=eps, log_domain=log_domain, max_iter=sinkhorn_iter)
        # T: n x out_size x in_size
        c = 0.5*c + 0.5*torch.bmm(T, x).mean(dim=0) / math.sqrt(c.shape[0])
        c /= c.norm(dim=-1, keepdim=True).clamp(min=1e-06)
        if ((c - prev_c) ** 2).sum() < 1e-06:
            break
        prev_c = c
    return c

def wasserstein_kmeans(x, n_clusters, out_size, eps=1.0, block_size=None, max_iter=100,
                       sinkhorn_iter=50, wb=False, verbose=True, log_domain=False, use_cuda=False):
    """
    x: n x in_size x in_dim
    output: n_clusters x out_size x in_dim
    out_size <= in_size
    """
    n, in_size, in_dim = x.shape
    if n_clusters == 1:
        if use_cuda:
            x = x.cuda()
        clusters = spherical_kmeans(x.view(-1, in_dim), out_size, block_size=block_size)
        if wb:
            clusters = wasserstein_barycenter(x, clusters, eps=0.1, log_domain=False)
        clusters = clusters.unsqueeze_(0)
        return clusters
    ## intialization
    indices = torch.randperm(n)[:n_clusters]
    clusters = x[indices, :out_size, :].clone()
    if use_cuda:
        clusters = clusters.cuda()

    wass_sim = x.new_empty(n)
    assign = x.new_empty(n, dtype=torch.long)
    if block_size is None or block_size == 0:
        block_size = n

    prev_sim = float('inf')
    for n_iter in range(max_iter):
        for i in range(0, n, block_size):
            end_i = min(i + block_size, n)
            x_batch = x[i: end_i]
            if use_cuda:
                x_batch = x_batch.cuda()
            tmp_sim = multihead_attn(x_batch, clusters, eps=eps, return_kernel=True, max_iter=sinkhorn_iter, log_domain=log_domain)
            tmp_sim = tmp_sim.cpu()
            wass_sim[i : end_i], assign[i: end_i] = tmp_sim.max(dim=-1)
        del x_batch
        sim = wass_sim.mean()
        if verbose and (n_iter + 1) % 10 == 0:
            print("Wasserstein spherical kmeans iter {}, objective value {}".format(
                  n_iter + 1, sim))

        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                idx = wass_sim.argmin()
                clusters[j].copy_(x[idx, :out_size, :])
                wass_sim[idx] = 1
            else:
                xj = x[index]
                if use_cuda:
                    xj = xj.cuda()
                c = spherical_kmeans(xj.view(-1, in_dim), out_size, block_size=block_size, verbose=False)
                if wb:
                    c = wasserstein_barycenter(xj, c, eps=0.001, log_domain=True, sinkhorn_iter=50)
                clusters[j] = c
        if torch.abs(prev_sim - sim) / sim.clamp(min=1e-10) < 1e-6:
            break
        prev_sim = sim
    return clusters
