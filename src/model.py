import torch
import torch.nn.functional as F 
import torch_geometric.nn as gnn
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.nn.norm import BatchNorm

# custom neural network module inheriting from torch.nn.Module
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()

        # Store the layers in a ModuleList for num_layers
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Initial linear layer projecting 'subject' and 'region' features to the hidden_channels
        self.lin_subject = torch.nn.Linear(3, hidden_channels) 
        self.lin_region = torch.nn.Linear(hidden_channels, hidden_channels) 

        for _ in range(num_layers):
            conv = gnn.HeteroConv({
                # Message passing from subjects to regions via 'has_region' edge type
                ('region', 'has_region', 'subject'): gnn.SAGEConv((-1, -1), hidden_channels),
                # Message passing between regions via 'is_functionally_connected' edge type
                ('region', 'is_functionally_connected', 'region'): gnn.GCNConv(-1, hidden_channels)
            }, aggr='sum')
            self.convs.append(conv)
            self.norms.append(BatchNorm(hidden_channels))
        
        # Final linear layer to produce output (classification for subjects)
        self.lin_output_subject = torch.nn.Linear(hidden_channels, out_channels)
        self.lin_output_region = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
        
    
def forward(self, x_dict, edge_index_dict, edge_attr_dict):
    
    for conv in self.convs:

        x_dict['subject'] = self.lin_subject(x_dict['subject'])
        x_dict['region'] = self.lin_region(x_dict['region'])
        
        for conv, norm in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict['subject'] = norm(x_dict['subject'])  # Normalización de nodos 'subject'
            x_dict['region'] = norm(x_dict['region'])  # Normalización de nodos 'region'
            x_dict['subject'] = F.relu(x_dict['subject'])
            x_dict['region'] = F.relu(x_dict['region'])
            x_dict['subject'] = F.dropout(x_dict['subject'], p=self.dropout, training=self.training)
            x_dict['region'] = F.dropout(x_dict['region'], p=self.dropout, training=self.training)
            
    return {
        'subject': self.lin_output_subject(x_dict['subject']),
        'region': self.lin_output_region(x_dict['region'])
    }
            
            # Code that tries to concatante edge attributes to the node feature matrix
            #out_dict = {}
            #for edge_type, conv_layer in conv.convs.items():
                # Extract edge index and edge attributes
                #edge_index = edge_index_dict[edge_type]
                #edge_attr = edge_attr_dict.get(edge_type)  # edge_attr may not exist for all edges

                # If edge attributes exist, concatenate them with node features
                #if edge_attr is not None:
                    #source_nodes, _ = edge_index
                    #node_features = x_dict[edge_type[0]]  # Source node features
                
                    # Concatenate edge attributes to the node features
                    #concatenated_input = torch.cat([node_features[source_nodes], edge_attr], dim=-1)
                
                    #out_dict[edge_type[2]] = conv_layer(concatenated_input, edge_index)
                #else:
                    # If no edge attributes, just use the node features as usual
                    #out_dict[edge_type[2]] = conv_layer(x_dict[edge_type[0]], edge_index)

            #x_dict = out_dict

        # Final output layer for 'subject' nodes
        #return self.lin(x_dict['subject'])
        

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
        loss.backward()
        # Model update based on gradients
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')


def evaluation_model(model, data):
    model.eval()
    
    # gradient not needed for evaluation
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