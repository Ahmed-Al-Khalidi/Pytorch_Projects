import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Input data (4 data points)
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Shape: [4, 1]
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Shape: [4, 1]

# Create a TensorDataset and DataLoader for mini-batching
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Batch size = 2

# Initialize weights and bias
w = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)  # Weight
b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)    # Bias

# Define the loss function (Mean Squared Error)
loss_fn = torch.nn.MSELoss()

# Define the optimizer (e.g., SGD)
optimizer = optim.SGD([w, b], lr=0.01)  # Learning rate = 0.01

# Number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    # Batch loop
    for x_batch, y_batch in dataloader:  # Iterate over mini-batches
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass: Compute predicted y for the mini-batch
        y_pred = torch.matmul(x_batch, w) + b  # Shape: [batch_size, 1]
        
        # Compute the loss (average loss over the mini-batch)
        loss = loss_fn(y_pred, y_batch)  # Scalar value
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update weights and bias using the optimizer
        optimizer.step()
    
    # Print the loss after each epoch
    print(f"Epoch {epoch + 1}: Loss = {loss.item()}, w = {w.item()}, b = {b.item()}")

# Final values of w and b
print(f"Final values: w = {w.item()}, b = {b.item()}")
