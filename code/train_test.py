import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def store_loss(epoch_loss, loss_values):
    """Stores the loss value for the current epoch."""
    loss_values.append(epoch_loss)

def plot_loss(loss_values, num_epochs):
    """Plots the loss values over epochs."""
    plt.figure()
    plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

def train_model(dataloader, model, criterion, optimizer, num_epochs=10):
    loss_values = []  # List to store loss values for visualization
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        store_loss(epoch_loss, loss_values)  # Store the epoch loss
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    plot_loss(loss_values, num_epochs)  # Plot the loss values

def test_model(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
