import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing

def generate_data(num_nodes=5,num_images=10,dim_nodes=128,dim_images=256,num_edge_img2node=10,num_edge_node2node=10):
    data = HeteroData()
    data['node'].x = torch.rand([num_nodes,dim_nodes])
    data['image'].x = torch.rand([num_images,dim_images])

    # build image 2 node
    img2node_0 = torch.randint(0,num_images-1,[num_edge_img2node])
    img2node_1 = torch.randint(0,num_nodes-1,[num_edge_img2node])
    img2node = torch.stack([img2node_0,img2node_1])
    data['image','observe','node'].edge_index = img2node

    # build node 2 node
    node2node_0 = torch.randint(0,num_nodes-1,[num_edge_node2node])
    node2node_1 = torch.randint(0,num_nodes-1,[num_edge_node2node])
    node2node = torch.stack([node2node_0,node2node_1])
    data['node','nearby','node'].edge_index = node2node

    return data

# class HG_MessagePassing(MessagePassing):
#     def __init__(self):
#         super().__init__()
#     def forward(self,x,edge_index):
#         return self.propagate(edge_index,x=x)
#     def message(self,x_i,x_j):
#         print('')
#         return None

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        
        # self.msg = HG_MessagePassing()
        self.conv1 = torch_geometric.nn.SAGEConv((-1, -1), hidden_channels)
        self.conv2 = torch_geometric.nn.SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        
        # self.msg(x,edge_index)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

data = generate_data()

model = GNN(hidden_channels=64, out_channels=20)
# model(data.x_dict,data.edge_index_dict)
print(model)
model = torch_geometric.nn.to_hetero(model, data.metadata(), aggr='sum')

print(model)
print('===')

model(data.x_dict,data.edge_index_dict)