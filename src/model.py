import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv 

# Defines the GCN model 
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.0, act='relu'):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers

        # Input layer
        self.layers.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.out_layer = GCNConv(hidden_channels, out_channels)

        # Activation function
        self.act = getattr(F, act) if isinstance(act, str) else act
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_layer(x, edge_index)
        return F.log_softmax(x, dim=1)

    def training(self, data, optimizer, num_epochs=200):
        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = self(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

    def evaluation(self, data):
        self.eval()
        with torch.no_grad():
            pred = self(data).argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')