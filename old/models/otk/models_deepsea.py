# -*- coding: utf-8 -*-
import torch
from torch import nn
from .layers import OTKernel


class OTLayer(nn.Module):
    def __init__(self, in_dim, out_size, heads=1, eps=0.1, max_iter=10,
                 position_encoding=None, position_sigma=0.1, out_dim=None,
                 dropout=0.4):
        super().__init__()
        self.out_size = out_size
        self.heads = heads
        if out_dim is None:
            out_dim = in_dim

        self.layer = nn.Sequential(
            OTKernel(in_dim, out_size, heads, eps, max_iter, log_domain=True,
                     position_encoding=position_encoding, position_sigma=position_sigma),
            nn.Linear(heads * in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
            )
        nn.init.xavier_uniform_(self.layer[0].weight)
        nn.init.xavier_uniform_(self.layer[1].weight)

    def forward(self, input):
        output = self.layer(input)
        return output

class SeqAttention(nn.Module):
    def __init__(self, nclass, hidden_size, filter_size,
                 n_attn_layers, eps=0.1, heads=1,
                 out_size=1, max_iter=10, hidden_layer=False,
                 position_encoding=None, position_sigma=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(4, hidden_size, kernel_size=filter_size),
            nn.ReLU(inplace=True),
            )

        attn_layers = [OTLayer(
            hidden_size, out_size, heads, eps, max_iter,
            position_encoding, position_sigma=position_sigma)] + [OTLayer(
            hidden_size, out_size, heads, eps, max_iter, position_encoding, position_sigma=position_sigma
            ) for _ in range(n_attn_layers - 1)]
        self.attn_layers = nn.Sequential(*attn_layers)

        self.out_features = out_size * hidden_size
        self.nclass = nclass

        if hidden_layer:
            self.classifier = nn.Sequential(
                nn.Linear(self.out_features, nclass),
                nn.ReLU(inplace=True),
                nn.Linear(nclass, nclass))
        else:
            self.classifier = nn.Linear(self.out_features, nclass)

    def representation(self, input):
        output = self.embed(input).transpose(1, 2).contiguous()
        output = self.attn_layers(output)
        output = output.reshape(output.shape[0], -1)
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
                output = batch_out.new_empty([n_samples] + list(batch_out.shape[1:]))
            output[batch_start:batch_start + batch_size] = batch_out
            target_output[batch_start:batch_start + batch_size] = target
            batch_start += batch_size
        return output, target_output
