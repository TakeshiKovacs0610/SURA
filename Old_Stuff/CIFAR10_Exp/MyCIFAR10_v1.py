import torch
import torchvision
import torchvision.transforms as transforms

# MPS -> Metal Performance Shading for Apple Silicon
# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Data loading work
# My transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])  # Normalize ((mean), (std))

def main():
    # Loading the data
    # Does the work of downloading the data if not present and applying the transforms
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)

    # This basically creates an iterable dataset, creating batches and giving you the option to enable shuffle after every epoch.
    # num_workers -> number of subprocesses to use for dataloading. More workers can speed up process.
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)

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
