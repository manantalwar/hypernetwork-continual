import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hypnettorch.hnets.mlp_hnet import HMLP
from hypnettorch.mnets.simple_rnn import SimpleRNN

####FORWARD PASS IS NOT WORKING. FIX IT####


# Hyperparameters
batch_size = 64
learning_rate = 0.01
epochs = 10

# Load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class MLP_SimpleRNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP_SimpleRNN, self).__init__()

        # Hypernetwork
        hnet = HMLP(target_shapes=[(128, 3, 32, 32)], layers=(128, 128), activation_fn=nn.ReLU())

        # Main network
        rnn = SimpleRNN(n_in=128, rnn_layers=(128, 128), fc_layers=(num_classes,), activation=nn.ReLU())

        # Register hypernetwork and main network
        self.hnet = nn.Sequential(hnet)
        self.rnn = nn.Sequential(rnn)

    def forward(self, uncond_input):
    # Forward pass through hypernetwork
        hnet_out = self.hnet(uncond_input)

    # Set weights in main network
        self.rnn.set_weights(hnet_out)

    # Forward pass through main network
        out = self.rnn(uncond_input)

        return out


# Initialize model
model = MLP_SimpleRNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print training statistics
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

# Test model on a few samples from CIFAR-10 dataset
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images, labels)
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted labels:', predicted)
