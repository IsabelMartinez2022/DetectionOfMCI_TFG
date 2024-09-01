import torch
import torch.nn.functional as F

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