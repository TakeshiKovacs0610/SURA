import json
from dataloader import get_dataloader
from model import SimpleCNN
from train_test import train_model, test_model
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

def main():
    train_csv = '../data/train_labels.csv'
    train_dir = '../data/fundus_757/'

    # Load configurations from JSON file
    with open('configs.json', 'r') as f:
        configs = json.load(f)

    for config in configs:
        print(f"Running with configuration: {config}")

        batch_size = config['batch_size']
        num_workers = config['num_workers']
        num_epochs = config['num_epochs']
        learning_rate = config['learning_rate']
        input_size = config.get('input_size', 256)  # Default to 32 if not specified

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Get dataloaders
        train_loader = get_dataloader(csv_file=train_csv, root_dir=train_dir, batch_size=batch_size, num_workers=num_workers, transforms=transform)

        # Initialize model, loss function, and optimizer
        model = SimpleCNN(input_size=input_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)

        # Optionally, test the model (if you have a test set and loader)
        # test_loader = get_dataloader(test_csv, test_dir, batch_size=batch_size, num_workers=num_workers)
        # test_model(test_loader, model)

if __name__ == "__main__":
    main()
