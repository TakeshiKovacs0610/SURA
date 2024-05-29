import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class HDF5Dataset(Dataset):
    def __init__(self, h5_file, transform=None):
        self.h5_file = h5_file
        self.transform = transform
        self.hf = h5py.File(h5_file, 'r')
        self.images = self.hf['images']
        self.labels = self.hf['labels']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        return image, label

def get_hdf5_dataloader(h5_file, batch_size=32, num_workers=4, transforms=None):
    dataset = HDF5Dataset(h5_file=h5_file, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader
