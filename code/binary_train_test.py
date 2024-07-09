import os
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("../saved_models/output.log"),
    logging.StreamHandler()
])
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def calculate_metrics(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    weighted_accuracy = ((tp * 0.5) + (tn * 0.5)) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
        'f1': f1,
        'weighted_accuracy': weighted_accuracy
    }

def async_save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """Saves the model and optimizer state asynchronously using the new zipfile-based serialization format."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    def save():
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f, _use_new_zipfile_serialization=True)
        with open('../saved_models/output.txt', 'w') as f:
            f.write(f"Async Checkpoint saved at {filename}\n")
        
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
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    logger.info(f"Checkpoint saved at {filename}")

def store_loss(epoch_loss, loss_values):
    """Stores the loss value for the current epoch."""
    loss_values.append(epoch_loss)

def plot_loss(train_losses,test_losses, model_name):
    """Plots the loss values over epochs."""
    num_epochs=len(train_losses)
    plt.figure()
    plt.plot(range(1, num_epochs + 1),train_losses, 'b-',label='Training Loss')
    plt.plot(range(1,num_epochs +1), test_losses, 'r-', label='Test Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    save_dir = os.path.join('..', 'saved_models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'training_test_loss_plot_{model_name}.png')
    plt.savefig(plot_path)

    logger.info(f"Training and test loss plot saved at {plot_path}")

def plot_metrics(train_losses, test_losses, train_accuracies,test_accuracies,model_name):
    """Plots the loss and accuracy values over epochs."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 6))
    
    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_dir = os.path.join('..', 'saved_models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, f'training_test_metrics_plot_{model_name}.png')
    plt.savefig(plot_path)
    
    logger.info(f"Training and test metrics plot saved at {plot_path}")


def load_checkpoint(model_name, checkpoint_num, model, optimizer):
    """Loads the model and optimizer state from a specified checkpoint."""
    filename = os.path.join('..', 'saved_models', model_name, f'checkpoint_epoch_{checkpoint_num}.pth')
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the given dataset without performing updates."""
    model.eval()
    running_loss = 0.0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float()  # BCE loss function expects both the actual and predicted labels to be in float
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))  # BCE expects the dimensions to be same as well
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    metrics = calculate_metrics(tp, fp, tn, fn)
    test_loss = running_loss / len(dataloader)
    return test_loss, metrics

def train_model(train_loader, test_loader, model, criterion, optimizer, model_name, num_epochs=10, save_every=2, checkpoint_num=None):
    """Trains the model and saves checkpoints regularly."""
    train_losses = []  # List to store loss values for visualization
    test_losses = []
    train_metrics = []
    test_metrics = []
    start_epoch = 0

    # Create directory for saving checkpoints if it doesn't exist
    save_dir = os.path.join('..', 'saved_models', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Load checkpoint if provided
    if checkpoint_num:
        model, optimizer, start_epoch, _ = load_checkpoint(model_name, checkpoint_num, model, optimizer)
        logger.info(f"Resumed training from epoch {start_epoch + 1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        tp = fp = tn = fn = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()  # Threshold for binary classification

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

            print("Batch Loss: {}, Running Accuracy: {}".format(loss.item(), (tp + tn) / (tp + tn + fp + fn)))
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_metrics.append(calculate_metrics(tp, fp, tn, fn))

        test_loss, test_metric = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_metrics.append(test_metric)
        
        logger.info(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Train Metrics: {train_metrics[-1]}, Test Metrics: {test_metric}')
        
        # Write metrics to file
        metrics_path = os.path.join(save_dir, f'training_test_metrics_{model_name}.txt')
        with open(metrics_path, 'a') as f:
            f.write(f'Epoch {epoch + 1}: Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Train Metrics: {train_metrics[-1]}, Test Metrics: {test_metric}\n')

        # Save the metrics to a separate file
        metrics_path_detailed = os.path.join(save_dir, f'training_test_metrics_detailed_{model_name}.txt')
        with open(metrics_path_detailed, 'a') as f:
            f.write(f'Epoch {epoch + 1}: {train_metrics[-1]}, Test Metrics: {test_metric}\n')

        # Save the model checkpoint after every 'save_every' epochs
        if (epoch + 1) % save_every == 0:
            checkpoint_filename = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            async_save_checkpoint(model, optimizer, epoch + 1, epoch_loss, checkpoint_filename)
        
        # Plot metrics after each epoch
        plot_metrics(train_losses, test_losses, [m['accuracy'] for m in train_metrics], [m['accuracy'] for m in test_metrics], model_name)

    plot_metrics(train_losses, test_losses, [m['accuracy'] for m in train_metrics], [m['accuracy'] for m in test_metrics], model_name)  # Plot the loss values

def save_predictions_to_csv(dataloader, model, output_file):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            predicted = (outputs > 0.5).float()  # Threshold for binary classification
            predictions.extend(predicted.cpu().numpy())
            
    df = pd.DataFrame({'Predictions': predictions})
    df.to_csv(output_file, index=False)
    logger.info(f"Predictions saved to {output_file}")
