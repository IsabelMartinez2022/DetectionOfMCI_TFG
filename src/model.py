import torch
import torch.nn.functional as F

def training_model(model, data, optimizer, num_epochs=200): #200 iterations over the training set to update weights
    model.train()
    for epoch in range(num_epochs):
        # Setting gradient to 0 for initialization
        optimizer.zero_grad()
        # Forward pass is called implicitly 
        out = model(data.x, data.edge_index)
        # Error between predicted and true values in the training nodes using Negative Log Likelihood Loss
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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
        pred = model(data.x, data.edge_index).argmax(dim=1)
        # Count of correct predictions in the test set
        correct_pred = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct_pred) / int(data.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')