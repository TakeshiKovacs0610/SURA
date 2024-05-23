import os  # used for directory operations
import pandas as pd  # reading csv files and data manipulation
from PIL import Image  # for image manipulation

import matplotlib
import matplotlib.pyplot as plt # for plotting images
matplotlib.use('Agg')

import torch  # pytorch
from torch.utils.data import Dataset  # dataset class which we will inherit
import torchvision.transforms as transforms  # for image transformations


def get_dataloader(csv_file, root_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = MyDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

def visualize_images(dataloader):
    images, labels = next(iter(dataloader))
    images = images.numpy()

    fig, axes = plt.subplots(1, len(images), figsize=(12, 12))
    if len(images) == 1:
        axes = [axes]
        
    for idx, ax in enumerate(axes):
        img = images[idx].transpose((1, 2, 0))  # Convert from CHW to HWC
        img = (img * 0.5) + 0.5  # Unnormalize
        ax.imshow(img)
        ax.set_title(f'Label: {labels[idx].item()}')
        ax.axis('off')
    plt.savefig("sample.png")
    print("DOne")
    # matplotlib.show()

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images if needed


        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # Create dataset

    dataset = MyDataset(csv_file='../data/train_labels.csv', root_dir='../data/fundus_757/', transform=transform)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # Example usage
    visualize_images(dataloader)
    # for images, labels in dataloader:
    #     print(images.shape, labels.shape)



if __name__ == "__main__":
    main()
       
