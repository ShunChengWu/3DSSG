import torch.nn as nn

# class SSG2D(nn.Module):
#     def __init__ (self, cfg):
#         super().__init__()
#         self.cfg = cfg
        
#     def forward(self, inputs):
#         ''' compute node feature '''
        
#         ''' compute edge feature '''
        
#         ''' update node & edge with GNN '''
        
#         ''' classification '''
        
#         pass
    
#         return 
    
    
#     def compute_node_feature(self, data):
#         '''
#         # geometric description 
#         use bounding box and centroid to compute a simple descriptor. 
#         '''
        
#         '''
#         # aggregate image features
#         get image feature of each object within the detected bounding box or mask region.
#         '''
#         patch = []
#         for bdb in boxes['bdb2D_pos']:
#             img = image.crop((bdb[0], bdb[1], bdb[2], bdb[3]))
#             img = data_transforms(img)
#             patch.append(img)
        