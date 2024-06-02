import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def async_save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """Saves the model and optimizer state asynchronously using the new zipfile-based serialization format."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    def save():
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f, _use_new_zipfile_serialization=True)
        # print(f"Checkpoint saved at {filename}")

    thread = threading.Thread(target=save)
    thread.start()


def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """Saves the model and optimizer state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")



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
    plot_path = os.path.join('..', 'saved_models', 'training_loss_plot.png')
    plt.savefig(plot_path)
    print(f"Training loss plot saved at {plot_path}")


def load_checkpoint(model_name, checkpoint_num, model, optimizer):
    """Loads the model and optimizer state from a specified checkpoint."""
    filename = os.path.join('..', 'saved_models', model_name, f'checkpoint_epoch_{checkpoint_num}.pth')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train_model(dataloader, model, criterion, optimizer, model_name, num_epochs=10, save_every=2, checkpoint_num=None):
    """Trains the model and saves checkpoints regularly."""
    loss_values = []  # List to store loss values for visualization
    start_epoch = 0

    # Create directory for saving checkpoints if it doesn't exist
    save_dir = os.path.join('..', 'saved_models', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint if provided
    if checkpoint_num:
        model, optimizer, start_epoch, _ = load_checkpoint(model_name, checkpoint_num, model, optimizer)
        print(f"Resumed training from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader):
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
        
        # Save the model checkpoint after every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            # save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_filename)
            async_save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_filename)

    plot_loss(loss_values, num_epochs)  # Plot the loss values

def test_model(dataloader, model):
    """Tests the model performance on the test dataset."""
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
