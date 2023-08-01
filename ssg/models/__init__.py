from . import edge_encoder
from .node_encoder import node_encoder_list
from .classifier import classifier_list
from .network_GNN import GraphEdgeAttenNetworkLayers,FAN_GRU,FAN_GRU_2, JointGNN,TripletGCNModel,TripletIMP,TripletVGfM
from .network_GNN import *

edge_encoder_list = {
    'sgfn': edge_encoder.EdgeEncoder_SGFN,
    'sgpn': edge_encoder.EdgeEncoder_SGPN,
    '2dssg': edge_encoder.EdgeEncoder_2DSSG,
    '2dssg_1':edge_encoder.EdgeEncoder_2DSSG_1,
    'vgfm':edge_encoder.EdgeEncoder_VGfM,
}
gnn_list = {
    'fan': GraphEdgeAttenNetworkLayers,
    'fan_gru': FAN_GRU,
    'fan_gru_2': FAN_GRU_2,
    'triplet': TripletGCNModel,
    'imp': TripletIMP,
    'vgfm': TripletVGfM,
    'jointgnn': JointGNN
}
