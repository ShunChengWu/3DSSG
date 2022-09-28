# from .ssg2d import SSG2D
# __all__ = ['SSG2D']
from .node_encoder import node_encoder_list
from . import edge_encoder
from .classifier import classifider_list
from .network_GNN import GraphEdgeAttenNetworkLayers, JointGNN,TripletGCNModel,TripletIMP,TripletVGfM

edge_encoder_list = {
    #'basic': edge_encoder.EdgeEncoder,
    'sgfn': edge_encoder.EdgeEncoder_SGFN,
    'sgpn': edge_encoder.EdgeEncoder_SGPN,
    '2dssg': edge_encoder.EdgeEncoder_2DSSG,
    '2dssg_1':edge_encoder.EdgeEncoder_2DSSG_1,
    'vgfm':edge_encoder.EdgeEncoder_VGfM,
}
gnn_list = {
    'fan': GraphEdgeAttenNetworkLayers,
    'triplet': TripletGCNModel,
    'imp': TripletIMP,
    'vgfm': TripletVGfM,
    'jointgnn': JointGNN
}
