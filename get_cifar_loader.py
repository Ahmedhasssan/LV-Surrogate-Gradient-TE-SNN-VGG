
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def build_cifar(args, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    aug.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root='./data/',
                            train=True, download=download, transform=transform_train)
    val_dataset = CIFAR10(root='./data/',
                            train=False, download=download, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            )
    return train_loader, val_loader