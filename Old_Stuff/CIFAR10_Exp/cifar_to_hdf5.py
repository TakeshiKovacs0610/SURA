import h5py
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Define the transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# Convert dataset to numpy arrays
train_data = np.array([trainset[i][0].numpy() for i in range(len(trainset))])
train_labels = np.array([trainset[i][1] for i in range(len(trainset))])

test_data = np.array([testset[i][0].numpy() for i in range(len(testset))])
test_labels = np.array([testset[i][1] for i in range(len(testset))])

# Create HDF5 file
with h5py.File('../data/cifar10.hdf5', 'w') as f:
    f.create_dataset('train_data', data=train_data)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('test_data', data=test_data)
    f.create_dataset('test_labels', data=test_labels)

print("HDF5 dataset created successfully.")
