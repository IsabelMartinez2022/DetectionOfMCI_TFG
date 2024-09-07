import torch
import torch.nn.functional as F 
import torch_geometric.nn as gnn

# custom neural network module inheriting from torch.nn.Module
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        # Store the layers in a ModuleList for num_layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                # Message passing from subjects to regions via 'has_region' edge type
                ('region', 'has_region', 'subject'): gnn.SAGEConv((-1, -1), hidden_channels),
                # Message passing between regions via 'is_functionally_connected' edge type
                ('region', 'is_functionally_connected', 'region'): gnn.GCNConv(-1, hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)
        
        # Final linear layer to produce output (classification for subjects)
        self.lin = gnn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for conv in self.convs:
            out_dict = {}
            for edge_type, conv_layer in conv.convs.items():
                # Check if the layer supports edge_attr by looking at its argument list
                if hasattr(conv_layer, 'edge_attr') and edge_type in edge_attr_dict:
                    out_dict[edge_type[2]] = conv_layer(
                        x_dict[edge_type[0]],
                        edge_index_dict[edge_type],
                        edge_attr=edge_attr_dict[edge_type]
                    )
                else:
                    # Do not pass edge_attr if the layer doesn't support it
                    out_dict[edge_type[2]] = conv_layer(
                        x_dict[edge_type[0]],
                        edge_index_dict[edge_type]
                    )
            x_dict = out_dict
        # Final output layer applied to 'subject' node type
        return self.lin(x_dict['subject'])


def training_model(model, data, optimizer, num_epochs=200): #200 iterations over the training set to update weights
    model.train()
    
    for epoch in range(num_epochs):
        # Setting gradient to 0 for initialization
        optimizer.zero_grad()
        # Forward pass is called implicitly to train subject nodes
        print(data[('region', 'has_region', 'subject')].edge_attr)
        print(data[('region', 'is_functionally_connected', 'region')].edge_attr)

        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        
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
        out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        probs = F.log_softmax(out, dim=1)
        # The predicted classes
        pred = probs.argmax(dim=1)

        # Count of correct predictions in the test set
        correct_pred = (pred[data['subject'].test_mask] == data['subject'].y[data['subject'].test_mask]).sum()
        acc = int(correct_pred) / int(data['subject'].test_mask.sum())
        print(f'Accuracy: {acc:.4f}')