import torchvision.transforms as transform_module
import torchvision.datasets as datasets_module
import torch.utils.data # for dataloader -> creating an iteratble 


transform1 = transform_module.Compose(
    [
        transform_module.RandomHorizontalFlip(),
        transform_module.RandomCrop(32,padding = 4),
        transform_module.ToTensor(),
        transform_module.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

transform2 = transform_module.Compose(
    [
        transform_module.ToTensor(),
        transform_module.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)


def get_data_loaders(data_file_path,batch_size,apply_augment=False):
    
    if apply_augment:
        transform = transform1
    else:
        transform = transform2
        
    trainset = datasets_module.CIFAR10(root=data_file_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets_module.CIFAR10(root=data_file_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader
