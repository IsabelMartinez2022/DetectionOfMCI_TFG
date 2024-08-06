import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import RandomNodeSplit

# Create a mapping from node IDs to consecutive indices
node_id_map = {node_id: idx for idx, node_id in enumerate(nodes_df['id'])}

# Prepare edge_index
edges_df['source'] = edges_df['source'].map(node_id_map)
edges_df['target'] = edges_df['target'].map(node_id_map)
edge_index = torch.tensor(edges_df[['source', 'target']].values.T, dtype=torch.long).contiguous()

# Prepare node features and labels
features = torch.tensor(nodes_df[['features']].values.tolist(), dtype=torch.float)  # Adjust depending on feature format
labels = torch.tensor(nodes_df['label'].values, dtype=torch.long)

# Create PyG Data object
data = Data(x=features, edge_index=edge_index, y=labels)

# Optionally split the data into train/test sets
transform = RandomNodeSplit(num_test=0.1, num_val=0.1)
data = transform(data)

print(data)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)