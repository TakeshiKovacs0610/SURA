# Description: This file contains the code for creating a custom dataset class and dataloader for our dataset.

import os  # used for directory operations
import pandas as pd  # reading csv files and data manipulation
from PIL import Image  # for image manipulation

import matplotlib
import matplotlib.pyplot as plt # for plotting images
matplotlib.use('Agg')

import torch  # pytorch
from torch.utils.data import Dataset  # dataset class which we will inherit
import torchvision.transforms as transforms  # for image transformations

#Progress bar
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        # csv_file -> Path to the csv file.
        # root_dir -> directory containing all the images
        # store the transforms
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.valid_indices = self._get_valid_indices()
        

    def _get_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.labels)):
            img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0]+'.jpeg')
            if os.path.isfile(img_name):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)  # returns total number of valid image samples


# retrieves an image and its label by index, applies transformations, and returns the image and label as a tuple

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = os.path.join(self.root_dir, self.labels.iloc[actual_idx, 0]+'.jpeg')
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[actual_idx, 1]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloader(csv_file, root_dir, batch_size=32, num_workers=4,transforms = None):
    transform = transforms
    
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


