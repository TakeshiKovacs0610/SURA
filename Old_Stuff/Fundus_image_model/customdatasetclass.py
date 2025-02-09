import os  # used for directory operations
import pandas as pd  # reading csv files and data manipulation
from PIL import Image  # for image manipulation

import torch  # pytorch
from torch.utils.data import Dataset  # dataset class which we will inherit
import torchvision.transforms as transforms  # for image transformations


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
            img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
            if os.path.isfile(img_name):
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)  # returns total number of valid image samples

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        img_name = os.path.join(self.root_dir, self.labels.iloc[actual_idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[actual_idx, 1]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    # Define transforms
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # Resize images if needed
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
    ])

    # Create dataset
    dataset = MyDataset(csv_file='../data/BigOne/trainLabels.csv', root_dir='../data/BigOne/sample/', transform=transform)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # Example usage
    for images, labels in dataloader:
        print(images.shape, labels.shape)
        # Perform your training or evaluation steps here


if __name__ == "__main__":
    main()
       
