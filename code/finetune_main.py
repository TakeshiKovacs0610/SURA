# Description: Main script to run the training and testing of the model
import os
import json
from One_Time_Work.dataloader import get_dataloader
from hdf5_dataloader import get_hdf5_dataloader
from model import * 
from binary_train_test import train_model, save_predictions_to_csv
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    # V add path here
    train_dir = 'path/to/train/'
    test_dir = 'path/to/test/'
    root_dir='path to main directory where we can save the remaining information'
    
    # V the model name
    cur_model='finetuning_resnet_50_attempt_1'
    
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
        input_size = config.get('input_size', 256)  # Default to 32 if not specified

        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.3203331149411, 0.224459668745, 0.1610336857647], std=[0.3024821581568, 0.2185284505098, 0.1741767781568])
        ])

        
        # V add the path to the hdf5 files
        train_hdf5_path = os.path.join(train_dir,'train_dataset.h5')
        test_hdf5_path = os.path.join(test_dir,'test_dataset.h5')
        
        # Get dataloaders
        train_loader = get_hdf5_dataloader(train_hdf5_path, batch_size=batch_size, num_workers=num_workers, transforms=transform)
        test_loader = get_hdf5_dataloader(test_hdf5_path, batch_size=batch_size, num_workers=num_workers, transforms=transform)
        
        if DEBUG:
            # Reduce dataset sizes for debugging
            train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_loader.dataset, list(range(128))), batch_size=batch_size, num_workers=num_workers, shuffle=True)
            test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_loader.dataset, list(range(32))), batch_size=batch_size, num_workers=num_workers, shuffle=False)
            num_epochs = 30  # Reduce number of epochs for debugging
            
    
    # Initialize the CustomResNet model
    # Still Need to make changes to the model
        model = CustomResNet(num_classes=1)
        model = model.to(device)
    
    # V add the path to the checkpoint -> "training_test_metrics_resnet_50_pretrained_full_data_pre_processed" -> Epoch 38 
    checkpoint_path = 'path/to/checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)
    
    # V if you get error here then let me know I might know why it is happening
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()  
    # This is for binary classification
    # Eariler we had multiclass classification so we used cross entropy loss
    
    
    # Train the model
    train_model(train_loader, test_loader, model, criterion, optimizer, model_name=cur_model, num_epochs=num_epochs, save_every=2, checkpoint_num=None)

    result_csv_path = os.path.join(root_dir, 'predictions_finetuned.csv')
    save_predictions_to_csv(test_loader, model, result_csv_path)
    
    


if __name__ == '__main__':
    main()