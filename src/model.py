import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn.norm import BatchNorm

# Definición del modelo HeteroGNN mejorado
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                ('subject', 'has_region', 'region'): gnn.GraphConv((-1, -1), hidden_channels),
                ('region', 'rev_has_region', 'subject'): gnn.GraphConv((-1, -1), hidden_channels),
                ('region', 'is_functionally_connected', 'region'): gnn.GATConv(-1, hidden_channels, heads=2)
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))

        self.lin_subject = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_region = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict['subject'] = norm(x_dict['subject'])  # Normalización de nodos 'subject'
            x_dict['region'] = norm(x_dict['region'])  # Normalización de nodos 'region'
            x_dict['subject'] = F.relu(x_dict['subject'])
            x_dict['region'] = F.relu(x_dict['region'])
            x_dict['subject'] = F.dropout(x_dict['subject'], p=self.dropout, training=self.training)
            x_dict['region'] = F.dropout(x_dict['region'], p=self.dropout, training=self.training)
            
        return {
            'subject': self.lin_subject(x_dict['subject']),
            'region': self.lin_region(x_dict['region'])
        }
