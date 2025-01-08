import torch
import torch.optim as optim

# Input data (x) and target output (y)
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Input features
y_data = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Target outputs

# Initialize weights and bias
w = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)  # Weight
b = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)    # Bias

# Define the loss function (Mean Squared Error)
loss_fn = torch.nn.MSELoss()

# Define the optimizer (e.g., SGD)
optimizer = optim.SGD([w, b], lr=0.01)  # Learning rate = 0.01

# Number of iterations
num_iters = 1000

# Training loop
for i in range(num_iters):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Forward pass: Compute predicted y
    y_pred = torch.matmul(x_data, w) + b  # y = w * x + b
    
    # Compute the loss using the predefined loss function
    loss = loss_fn(y_pred, y_data)
    
    # Backward pass: Compute gradients
    loss.backward()
    
    # Update weights and bias using the optimizer
    optimizer.step()
    
    # Print the loss every 100 iterations
    if i % 100 == 0:
        print(f"Iteration {i}: Loss = {loss.item()}, w = {w.item()}, b = {b.item()}")

# Final values of w and b
print(f"Final values: w = {w.item()}, b = {b.item()}")
