import torch
import torchvision
import torchvision.transforms as transforms

# MPS -> Metal Performance Shading for Apple Silicon
# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data loading work
# My transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def main():
    # Loading the data
    # Does the work of downloading the data if not present and applying the transforms
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)

    # This basically creates an iterable dataset, creating batches and giving you the option to enable shuffle after every epoch.
    # num_workers -> number of subprocesses to use for dataloading. More workers can speed up process.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model definition work
    import torch.nn as nn
    import torch.nn.functional as F

    class ImprovedCNN(nn.Module):
        
        def __init__(self):
            super(ImprovedCNN, self).__init__()
            # First convolutional block
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout1 = nn.Dropout(0.25)

            # Second convolutional block
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout2 = nn.Dropout(0.25)

            # Third convolutional block
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout3 = nn.Dropout(0.25)

            # Fully connected layers
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.bn_fc1 = nn.BatchNorm1d(512)
            self.dropout_fc1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, 256)
            self.bn_fc2 = nn.BatchNorm1d(256)
            self.dropout_fc2 = nn.Dropout(0.5)
            self.fc3 = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.pool1(x)
            x = self.dropout1(x)

            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = self.pool2(x)
            x = self.dropout2(x)

            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
            x = self.pool3(x)
            x = self.dropout3(x)

            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.bn_fc1(self.fc1(x)))
            x = self.dropout_fc1(x)
            x = F.relu(self.bn_fc2(self.fc2(x)))
            x = self.dropout_fc2(x)
            x = self.fc3(x)
            return x

    # move to device
    test_net = ImprovedCNN().to(device)

    # Training loop setup
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(test_net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(10):
        running_loss = 0.0

        # enumerate takes the iterable trainloader and the starting index 0
        for i, data in enumerate(trainloader, 0):
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
                # .3f ensures that the printed loss is rounded and displayed to 3 decimal places f -> fixed point
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
        
        scheduler.step()

    print('Finished Training')

    # Testing part
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            # moved to device
            images, labels = images.to(device), labels.to(device)
            
            outputs = test_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()
