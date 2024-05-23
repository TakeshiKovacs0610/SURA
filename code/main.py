import torch as nn
import torch.optim as optim
from dataloader import get_dataloader
from model import SimpleCNN
from train_test import train_model, test_model

def main():
    train_csv = '../data/BigOne/trainLabels.csv'
    train_dir = '../data/BigOne/sample/'

    batch_size = 32
    num_workers = 4
    num_epochs = 10

    # Get dataloaders
    train_loader = get_dataloader(csv_file=train_csv, root_dir=train_dir, batch_size=batch_size, num_workers=num_workers)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)

    # Optionally, test the model (if you have a test set and loader)
    # test_loader = get_dataloader(test_csv, test_dir, batch_size=batch_size, num_workers=num_workers)
    # test_model(test_loader, model)

if __name__ == "__main__":
    main()
