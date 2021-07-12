import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def load_cifar10(path, batch_size, is_cuda=False) -> (DataLoader, DataLoader, tuple):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset_kwargs = {'batch_size': batch_size}

    if not is_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        dataset_kwargs.update(cuda_kwargs)

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, **dataset_kwargs)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, **dataset_kwargs)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes
