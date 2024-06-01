# import torch 
# import torch.nn as nn
# import torch.optim as optim
# from dataloader import get_dataloader
# from model import SimpleCNN
# from train_test import train_model, test_model

# def main():
#     #device for computation
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")


#     train_csv = '../data/train_labels.csv'
#     train_dir = '../data/fundus_757/'

#     batch_size = 32
#     num_workers = 4
#     num_epochs = 10

#     # Get dataloaders
#     train_loader = get_dataloader(csv_file=train_csv, root_dir=train_dir, batch_size=batch_size, num_workers=num_workers)

#     # Initialize model, loss function, and optimizer
#     model = SimpleCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Train the model
#     train_model(train_loader, model, criterion, optimizer,device=device,num_epochs=num_epochs)

#     # Optionally, test the model (if you have a test set and loader)
#     # test_loader = get_dataloader(test_csv, test_dir, batch_size=batch_size, num_workers=num_workers)
#     # test_model(test_loader, model)

# if __name__ == "__main__":
#     main()


import json
from dataloader import get_dataloader
from model import * 
from train_test import train_model, test_model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_csv = '../data/trainLabels.csv'
    train_dir = '../data/kaggle_data/train/'

    # Load configurations from JSON file
    with open('configs.json', 'r') as f:
        configs = json.load(f)

    for config in configs:
        print(f"Running with configuration: {config}")
        print("Device in use: ", device)
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
        # model = SimpleCNN(input_size=input_size,num_classes=5).to(device)
        model= models.resnet50(pretrained = True)
        num_classes = 5
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(train_loader, model, criterion, optimizer, num_epochs=num_epochs)

        # Optionally, test the model (if you have a test set and loader)
        # test_loader = get_dataloader(test_csv, test_dir, batch_size=batch_size, num_workers=num_workers)
        # test_model(test_loader, model)

if __name__ == "__main__":
    main()
    