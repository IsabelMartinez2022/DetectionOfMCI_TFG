import torch
import torch.nn.functional as F 
import torch_geometric.nn as gnn
from torch.cuda.amp import GradScaler, autocast

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
            for edge_type, conv in conv.convs.items():

                # Gives error all the time
                out_dict = conv(x_dict, edge_index_dict, edge_attr_dict)

                # Extract edge index and edge attributes
                edge_index = edge_index_dict[edge_type]
                edge_attr = edge_attr_dict.get(edge_type)  # edge_attr does not exist for GCNconv

                # If edge attributes exist, concatenate them with node features
                # if edge_attr is not None:
                    #source_nodes, _ = edge_index
                    #node_features = x_dict[edge_type[0]]  # Source node features
                
                    # Concatenating edge attributes to the node features to include them during convolution
                    #concatenated_input = torch.cat([node_features[source_nodes], edge_attr], dim=-1)
                    #out_dict[edge_type[2]] = conv_layer(concatenated_input, edge_index)
                # else:
                    # If no edge attributes, just use the node features as usual
                    #out_dict[edge_type[2]] = conv_layer(x_dict[edge_type[0]], edge_index)

            x_dict = out_dict

        # Final output layer for 'subject' nodes
        return self.lin(x_dict['subject'])
        

def training_model(model, data, optimizer, num_epochs=200): #200 iterations over the training set to update weights
    scaler = GradScaler()  # Automatic mixed precision
    model.train()
    
    for epoch in range(num_epochs):
        # Setting gradient to 0 for initialization
        optimizer.zero_grad()

        print(data.x_dict)
        print(data.edge_index_dict)
        print(data.edge_attr_dict)
        print(data.y_dict)

        with autocast():
            # Forward pass is called implicitly to train subject nodes
            out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict) #tensor output
            print(f'Tensor shape: {out.shape}')
            print(f'Tensor content (first 10 rows):\n{out[:10]}')
        
            loss = F.nll_loss(out, data.y_dict['subject'][data['subject'].train_mask])
      
            
            print(f'out shape: {out.shape}')
            #print(f'train_mask shape: {train_mask.shape}')
  
        
        # Apply mask to get labels for subject nodes
        #y_train = data['subject'].y[train_mask]  # Labels for training nodes

        # Get log probabilities
        #probs = F.log_softmax(out_train, dim=1)
          
        #loss = F.nll_loss(probs, y_train)
        scaler.scale(loss).backward()
        #loss.backward()
        #optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        print(f'Epoch {epoch}, Loss: {loss.item()}')

        # Create a mask for subject nodes
        # Assuming `data['subject'].train_mask` is a boolean mask for `subject` nodes
        # num_subject_nodes = data['subject'].train_mask.sum().item()

        # Check if the `num_total_nodes` aligns with the number of nodes for which you have predictions
        #num_total_nodes = probs.shape[0]  # This should be the total number of nodes in your graph

        #if num_total_nodes != len(data['subject'].train_mask):
            #raise ValueError("Mismatch between total number of nodes and the length of the train mask")

        # Create an index tensor for subject nodes
        #subject_indices = torch.arange(num_total_nodes)[data['subject'].train_mask]

        # Filter for subject nodes
        #train_probs = probs[subject_indices]
        #train_labels = data['subject'].y[subject_indices]
        
        # Error between predicted and true values in the training nodes using Negative Log Likelihood Loss
        #loss = F.nll_loss(train_probs, train_labels)
        
        # Backpropagation: computing the gradient of the loss function with respect to each model parameter
        #loss.backward()
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