# Description: Main script to run the training and testing of the model
import os
import json
from dataloader import get_dataloader
from hdf5_dataloader import get_hdf5_dataloader
from model import * 
from train_test import train_model, save_predictions_to_csv
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    train_dir = '../data/kaggle_data/train/'
    test_dir = '../data/kaggle_data/test/'
    root_dir='../data/kaggle_data/'

    cur_model='resnet_50_pretrained_full_data'

    # Load configurations from JSON file
    with open('configs.json', 'r') as f:
        configs = json.load(f)

    # Define DEBUG flag
    DEBUG = False  # Set to False for full dataset training

    for config in configs:
        print(f"Running with configuration: {config}")
        print("Device in use: ", device)
        batch_size = config['batch_size']
        num_workers = config['num_workers']
        num_epochs = config['num_epochs']
        learning_rate = config['learning_rate']
        # input_size = config.get('input_size', 256)  # Default to 256 if not specified
        input_size = 299

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #just see if these are the correct mean and standard deivations.
            transforms.Normalize(mean=[0.3203331149411, 0.224459668745, 0.1610336857647], std=[0.3024821581568, 0.2185284505098, 0.1741767781568])
        ])

        train_hdf5_path = os.path.join(train_dir,'train_dataset.h5')
        test_hdf5_path = os.path.join(test_dir,'test_dataset.h5')

        # Get dataloaders
        # train_loader = get_dataloader(csv_file=train_csv, root_dir=train_dir, batch_size=batch_size, num_workers=num_workers, transforms=transform)
        train_loader = get_hdf5_dataloader(train_hdf5_path, batch_size=batch_size, num_workers=num_workers, transforms=transform)
        test_loader = get_hdf5_dataloader(test_hdf5_path, batch_size=batch_size, num_workers=num_workers, transforms=transform)
        
        if DEBUG:
            # Reduce dataset sizes for debugging
            train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_loader.dataset, list(range(128))), batch_size=batch_size, num_workers=num_workers, shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_loader.dataset, list(range(32))), batch_size=batch_size, num_workers=num_workers, shuffle=False)
            num_epochs = 30  # Reduce number of epochs for debugging


        # Initialize model, loss function, and optimizer
        # model = SimpleCNN(input_size=input_size,num_classes=5).to(device)
        model = models.inception_v3(pretrained=True, aux_logits=False)
        num_classes = 5
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop( model.parameters() , lr=0.001 , weight_decay = 0.00004 )

        # Train the model
        train_model(train_loader,test_loader, model, criterion, optimizer,model_name=cur_model, num_epochs=num_epochs,save_every=2,checkpoint_num=None)

        result_csv_path = os.path.join(root_dir,'predictions.csv')
        save_predictions_to_csv(test_loader,model,result_csv_path)

        # Optionally, test the model (if you have a test set and loader)
        # test_loader = get_dataloader(test_csv, test_dir, batch_size=batch_size, num_workers=num_workers)
        # test_model(test_loader, model)

if __name__ == "__main__":
    main()
    