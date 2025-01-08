# 1. Import Dependencies
#----------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
#----------------------------------------------------------------------------------------------------#

# 2. Load and Process Dataset
#----------------------------------------------------------------------------------------------------#
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#----------------------------------------------------------------------------------------------------#

# 3. Split Dataset
#----------------------------------------------------------------------------------------------------#
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#----------------------------------------------------------------------------------------------------#

# 4. Convert Dataset to Tensors
#----------------------------------------------------------------------------------------------------#
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#----------------------------------------------------------------------------------------------------#

# 5. Build Neural Network Model
#----------------------------------------------------------------------------------------------------#
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
#----------------------------------------------------------------------------------------------------#

# 6. Create an Instance of the Model
#----------------------------------------------------------------------------------------------------#
model = NeuralNet()
#----------------------------------------------------------------------------------------------------#

# 7. Define Learning Rate, Loss Function, and Optimizer
#----------------------------------------------------------------------------------------------------#
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#----------------------------------------------------------------------------------------------------#

# 8. Define Number of Epochs
#----------------------------------------------------------------------------------------------------#
epochs = 5
#----------------------------------------------------------------------------------------------------#

# 9. Build the Training Loop
#----------------------------------------------------------------------------------------------------#
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            # 9.1 Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # 9.2 Zero the gradients
            optimizer.zero_grad()
            # 9.3 Forward pass
            outputs = model(inputs)
            # 9.4 Compute loss
            loss = criterion(outputs, labels)
            # 9.5 Backward pass
            loss.backward()
            # 9.6 Optimize the weights
            optimizer.step()
            # 9.7 Update running loss
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}')
#----------------------------------------------------------------------------------------------------#










# 10. Build the Testing Loop
#----------------------------------------------------------------------------------------------------#
def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # 10.1 Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            # 10.2 Forward pass
            outputs = model(inputs)
            # 10.3 Compute loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # 10.4 Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # 10.5 Calculate average loss and accuracy
    avg_loss = test_loss / len(dataloader)
    accuracy = correct / total
print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
#----------------------------------------------------------------------------------------------------#

# 11. Device Configuration
#----------------------------------------------------------------------------------------------------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#----------------------------------------------------------------------------------------------------#

# 12. Train and Test the Model
#----------------------------------------------------------------------------------------------------#
train(model, train_loader, criterion, optimizer, device)
test(model, test_loader, criterion, device)
#----------------------------------------------------------------------------------------------------#

# 13. Save the trained model 
#----------------------------------------------------------------------------------------------------#
torch.save(model.state_dict(), 'model.pth')
#----------------------------------------------------------------------------------------------------#
