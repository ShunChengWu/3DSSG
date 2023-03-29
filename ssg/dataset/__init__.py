# from .dataloader_3RScan import Sequence_Loader
# from .dataloader_detection import Graph_Loader
from .dataloader_SGFN import SGFNDataset
# from .dataloader_SGPN import SGPNDataset
# from .dataloader_SGFN_incre import SGFNIDataset
from . import dataloader_SGFN_seq
# from .dataloader_mv_scannet import MultiViewImageLoader
# from .dataloader_mv_roi import MultiViewROIImageLoader
# from .dataloader_sv_roi import SingleViewROIImageLoader
dataset_dict = {
#   '3RScan': Sequence_Loader,
#   'graph': Graph_Loader,
  'sgfn': SGFNDataset,
#   "sgpn": SGPNDataset,
#   # 'mv': MultiViewImageLoader,
#    'mv_roi': MultiViewROIImageLoader,
#    'sgfn_incre': SGFNIDataset,
   'sgfn_seq': dataloader_SGFN_seq.Dataset,
   # 'sv_roi': SingleViewROIImageLoader,
}

# __all__ = ['dataset_dict']

