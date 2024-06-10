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

def rename_checkpoint_keys(checkpoint,prefix="model."):
    new_state_dict={}
    for k,v in checkpoint.items():
        new_key = prefix+k
        new_state_dict[new_key]=v
    return new_state_dict



def main():
    
    # V add path here
    train_dir = '../data/Fine_tuning_sample/train_processed/'
    test_dir = '../data/Fine_tuning_sample/test_processed/'
    root_dir='../data/Fine_tuning_sample'
    
    # V the model name
    cur_model='finetuning_resnet_50_attempt_1_partially_trainable'
    
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
            transforms.Normalize(mean=[0.5030136145939631, 0.5020357954878051, 0.501814967034372], std=[0.051652922458824094, 0.04804644590105783, 0.030665797935070693])
        ])

        
        # V add the path to the hdf5 files
        train_hdf5_path = os.path.join(train_dir,'train_processed_dataset.h5')
        test_hdf5_path = os.path.join(test_dir,'test_processed_dataset.h5')
        
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
    checkpoint_path = '../saved_models/resnet_50_pretrained_full_data_pre_processed/checkpoint_epoch_38.pth'
    checkpoint = torch.load(checkpoint_path)
    #print("Keys in checkpoint['model_state_dict']:", checkpoint['model_state_dict'].keys())


    # renaming the loaded checkpoints to match the custom model layer names
    adjusted_state_dict = rename_checkpoint_keys(checkpoint['model_state_dict'])


    # V if you get error here then let me know I might know why it is happening
    model.load_state_dict(adjusted_state_dict, strict=False)
    

 # -------------------------------------------------------------------------------
    # debudding to check the trainable weights and loading of pretrainded weights 
    # print("Adjusted keys in state_dict:")
    # for key in adjusted_state_dict.keys():
    #     print(key)

    # # verifying the loading of the model
    # for name, param in model.named_parameters():
    #         if name in adjusted_state_dict:
    #             checkpoint_param = adjusted_state_dict[name]
    #             if not torch.equal(param.data, checkpoint_param.data):
    #                 print(f"Warning: The parameter {name} does not match the checkpoint parameter.")
    #             else:
    #                 print(f"The parameter {name} is successfully loaded from the checkpoint.")
    #         else:
    #             print(f"The parameter {name} is not in the checkpoint.")
# -------------------------------------------------------------------------------------------


    #-----------Parameter freezing----------------
    for name,param in model.named_parameters():
        print(f"Layer: {name} | Trainable: {param.requires_grad}")

    print("-----------------------------------------------------------------------------")
    for name,param in model.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = False

    for name,param in model.named_parameters():
        print(f"Layer: {name} | Trainable: {param.requires_grad}")



    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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