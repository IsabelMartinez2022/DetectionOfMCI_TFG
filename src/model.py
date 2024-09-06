import torch
import torch.nn.functional as F 
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear

# custom neural network module inheriting from torch.nn.Module.
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        # For each layer, define different GNN layers for different edge types
        for _ in range(num_layers):
            conv = HeteroConv({
                # Message passing from subjects to regions via 'has_region' edge type
                ('subject', 'has_region', 'region'): SAGEConv((-1, -1), hidden_channels),
                # Message passing between regions via 'is_functionally_connected' edge type
                ('region', 'is_functionally_connected', 'region'): GCNConv(-1, hidden_channels),
            }, aggr='sum')  # Aggregation strategy (sum, mean, etc.)
            self.convs.append(conv)

        # Final linear layer to produce output (classification for subjects)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            # Perform message passing and relu activation for each layer
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Return final logits for the 'subject' node type
        return self.lin(x_dict['subject'])

def training_model(model, data, optimizer, num_epochs=200): #200 iterations over the training set to update weights
    model.train()
    for epoch in range(num_epochs):
        # Setting gradient to 0 for initialization
        optimizer.zero_grad()
        # Forward pass is called implicitly to train subject nodes
        out = model(data['subject'].x, data[('subject', 'has_region', 'region')].edge_index, data[('region', 'is_functionally_connected', 'region')].edge_index)
         # To get log probabilities from the output of the last linear layer
        probs = F.log_softmax(out, dim=1)
        # Error between predicted and true values in the training nodes using Negative Log Likelihood Loss
        loss = F.nll_loss(probs[data['subject'].train_mask], data['subject'].y[data['subject'].train_mask])
        
        # Backpropagation: computing the gradient of the loss function with respect to each model parameter
        loss.backward()
        # Model update based on gradients
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

def evaluation_model(model, data):
    model.eval()
    # gradient no needed for evaluation
    with torch.no_grad():
        # Maximum value along dimension 1= prediction class
        out = model(data['subject'].x, data[('subject', 'has_region', 'region')].edge_index,  data[('region', 'is_functionally_connected', 'region')].edge_index)
        probs = F.log_softmax(out, dim=1)
        # The predicted classes
        pred = probs.argmax(dim=1)

        # Count of correct predictions in the test set
        correct_pred = (pred[data['subject'].test_mask] == data['subject'].y[data['subject'].test_mask]).sum()
        acc = int(correct_pred) / int(data['subject'].test_mask.sum())
        print(f'Accuracy: {acc:.4f}')