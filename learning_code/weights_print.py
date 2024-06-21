import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

# Example model
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(32 * 14 * 14, 10)
)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Loss function
criterion = nn.CrossEntropyLoss()

# Training constants
batch_size = 64
num_epochs = 10
shuffle = True

# Data transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Open a file to write the outputs
with open("model_details.txt", "w") as file:
    file.write("Model parameters (weights and biases):\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            file.write(f"{name}: {param.data}\n")

    file.write("\nOptimizer parameters:\n")
    for group in optimizer.param_groups:
        file.write(f"{group}\n")

    file.write("\nLoss function:\n")
    file.write(f"{criterion}\n")

    file.write("\nTraining constants:\n")
    file.write(f"Batch size: {batch_size}\n")
    file.write(f"Number of epochs: {num_epochs}\n")
    file.write(f"Shuffle: {shuffle}\n")

    file.write("\nData transformations:\n")
    file.write(f"{transform}\n")

    file.write("\nLearning rate scheduler:\n")
    file.write(f"{scheduler}\n")

    file.write("\nDevice:\n")
    file.write(f"{device}\n")
