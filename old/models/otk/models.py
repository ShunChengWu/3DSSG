# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import OTKernel, Linear
from ckn.layers import BioEmbedding
from ckn.models import CKNSequential


class SeqAttention(nn.Module):
    def __init__(self, in_channels, nclass, hidden_sizes, filter_sizes,
                 subsamplings, kernel_args=None, eps=0.1, heads=1,
                 out_size=1, max_iter=50, alpha=0., fit_bias=True,
                 mask_zeros=True):
        super().__init__()
        self.embed_layer = BioEmbedding(
            in_channels, False, mask_zeros=True, no_embed=True)
        self.ckn_model = CKNSequential(
            in_channels, hidden_sizes, filter_sizes,
            subsamplings, kernel_args_list=kernel_args)
        self.attention = OTKernel(hidden_sizes[-1], out_size, heads=heads,
                                  eps=eps, max_iter=max_iter)
        self.out_features = out_size * heads * hidden_sizes[-1]
        self.nclass = nclass

        self.classifier = Linear(self.out_features, nclass, bias=fit_bias)
        self.alpha = alpha
        self.mask_zeros = mask_zeros

    def feature_parameters(self):
        import itertools
        return itertools.chain(self.ckn_model.parameters(),
                               self.attention.parameters())

    def normalize_(self):
        self.ckn_model.normalize_()

    def ckn_representation_at(self, input, n=0):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model.representation(output, n)
        mask = self.ckn_model.compute_mask(mask, n)
        return output, mask

    def ckn_representation(self, input):
        output = self.embed_layer(input)
        output = self.ckn_model(output).permute(0, 2, 1).contiguous()
        return output

    def representation(self, input):
        output = self.embed_layer(input)
        mask = self.embed_layer.compute_mask(input)
        output = self.ckn_model(output).permute(0, 2, 1).contiguous()
        mask = self.ckn_model.compute_mask(mask)
        if not self.mask_zeros:
            mask = None
        output = self.attention(output, mask).reshape(output.shape[0], -1)
        return output

    def forward(self, input):
        output = self.representation(input)
        return self.classifier(output)

    def predict(self, data_loader, only_repr=False, use_cuda=False):
        n_samples = len(data_loader.dataset)
        target_output = torch.LongTensor(n_samples)
        batch_start = 0
        for i, (data, target) in enumerate(data_loader):
            batch_size = data.shape[0]
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                if only_repr:
                    batch_out = self.representation(data).data.cpu()
                else:
                    batch_out = self(data).data.cpu()
            if i == 0:
                output = batch_out.new_empty([n_samples] +
                                             list(batch_out.shape[1:]))
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target
            batch_start += batch_size
        return output, target_output

    def train_classifier(self, data_loader, criterion=None, epochs=100,
                         optimizer=None, use_cuda=False):
        encoded_train, encoded_target = self.predict(
            data_loader, only_repr=True, use_cuda=use_cuda)
        self.classifier.fit(encoded_train, encoded_target, criterion,
                            reg=self.alpha, epochs=epochs, optimizer=optimizer,
                            use_cuda=use_cuda)

    def unsup_train(self, data_loader, n_sampling_patches=300000,
                    n_samples=5000, wb=False, use_cuda=False):
        self.eval()
        if use_cuda:
            self.cuda()

        for i, ckn_layer in enumerate(self.ckn_model):
            print("Training ckn layer {}".format(i))
            n_patches = 0
            try:
                n_patches_per_batch = (
                    n_sampling_patches + len(data_loader) - 1
                    ) // len(data_loader)
            except:
                n_patches_per_batch = 1000
            patches = torch.Tensor(n_sampling_patches, ckn_layer.patch_dim)
            if use_cuda:
                patches = patches.cuda()

            for data, _ in data_loader:
                if n_patches >= n_sampling_patches:
                    continue
                if use_cuda:
                    data = data.cuda()
                with torch.no_grad():
                    data, mask = self.ckn_representation_at(data, i)
                    data_patches = ckn_layer.sample_patches(
                        data, mask, n_patches_per_batch)
                size = data_patches.size(0)
                if n_patches + size > n_sampling_patches:
                    size = n_sampling_patches - n_patches
                    data_patches = data_patches[:size]
                patches[n_patches: n_patches + size] = data_patches
                n_patches += size

            print("total number of patches: {}".format(n_patches))
            patches = patches[:n_patches]
            ckn_layer.unsup_train(patches, init=None)

        n_samples = min(n_samples, len(data_loader.dataset))
        cur_samples = 0
        print("Training attention layer")
        for i, (data, _) in enumerate(data_loader):
            if cur_samples >= n_samples:
                continue
            if use_cuda:
                data = data.cuda()
            with torch.no_grad():
                data = self.ckn_representation(data)

            if i == 0:
                patches = torch.empty([n_samples]+list(data.shape[1:]))

            size = data.shape[0]
            if cur_samples + size > n_samples:
                size = n_samples - cur_samples
                data = data[:size]
            patches[cur_samples: cur_samples + size] = data
            cur_samples += size
        print(patches.shape)
        self.attention.unsup_train(patches, wb=wb, use_cuda=use_cuda)
