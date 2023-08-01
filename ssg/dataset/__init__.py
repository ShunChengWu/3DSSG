from .dataloader_SGFN import SGFNDataset
from . import dataloader_SGFN_seq
dataset_dict = {
  'sgfn': SGFNDataset,
   'sgfn_seq': dataloader_SGFN_seq.Dataset,
}

# __all__ = ['dataset_dict']

