import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import h5py
from torch.utils.data import Dataset, DataLoader

# MPS -> Metal Performance Shading for Apple Silicon
# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data loading work
# My transform
transform = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])  # Normalize ((mean), (std))

# Custom dataset to load data from HDF5 file
class CIFAR10HDF5(Dataset):
    def __init__(self, hdf5_file, train=True, transform=None):
        self.hdf5_file = hdf5_file
        self.train = train
        self.transform = transform
        with h5py.File(self.hdf5_file, 'r') as f:
            if self.train:
                self.data = f['train_data'][:]
                self.labels = f['train_labels'][:]
            else:
                self.data = f['test_data'][:]
                self.labels = f['test_labels'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        img = torch.tensor(img)
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    # Load the HDF5 dataset
    hdf5_file = '../data/cifar10.hdf5'
    trainset = CIFAR10HDF5(hdf5_file, train=True, transform=transform)
    testset = CIFAR10HDF5(hdf5_file, train=False, transform=transform)

    # Data loaders
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model definition work
    import torch.nn as nn
    import torch.nn.functional as F

    class myCNN(nn.Module):
        def __init__(self):
            super(myCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)  # will be used twice after conv1 and conv2
            self.fc1 = nn.Linear(32 * 8 * 8, 120)  # before this a linearisation will occur to ensure that the correct data is transmitted forward.
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # move to device
    test_net = myCNN().to(device)

    # Training loop setup
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(test_net.parameters(), lr=0.001, momentum=0.9)

    with open("training_log.txt", "w") as log_file, open("layer_weights.txt", "w") as weights_file:
        for epoch in range(10):
            running_loss = 0.0

            # enumerate takes the iterable trainloader and the starting index 0
            for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}"), 0):
                # get the inputs, data is a list of [inputs, labels]
                # data is a list with 2 elements both are tensors, input is a tensor containing input images and labels is a tensor containing labels
                inputs, labels = data
                
                # move to the device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = test_net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # calculate the gradient loss with respect to the model parameters using backprop
                optimizer.step()  # update the model parameters using optimization algo SGD

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 200 mini-batches
                    log_message = f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}'
                    print(log_message)
                    log_file.write(log_message + '\n')
                    running_loss = 0.0

            # After each epoch, calculate the sum of weights for each layer and write to file
            weights_file.write(f"Epoch {epoch + 1}\n")
            for name, param in test_net.named_parameters():
                if param.requires_grad:
                    weights_file.write(f"{name}: {param.data.sum().item()}\n")

        log_file.write('Finished Training\n')

        # Testing part
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(testloader, desc="Testing"):
                images, labels = data
                
                # moved to device
                images, labels = images.to(device), labels.to(device)
                
                outputs = test_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy_message = f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%'
        print(accuracy_message)
        log_file.write(accuracy_message + '\n')

if __name__ == '__main__':
    main()
