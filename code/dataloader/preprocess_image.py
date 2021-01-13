import math
import os
from collections import defaultdict

import PIL
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from utils import *

from .vision import VisionDataset

image_types = ['cifar10', 'cifar100', 'imagenet']
image_size = [224, 224]
delimiter = '/'

def preprocess_images(args):
    if args.datasets == 'imagenet':
        return load_imagenet(args)
    elif args.datasets == 'cifar10':
        return load_cifar10(args)
    else:
        return load_cifar100(args)


def load_imagenet(args):
    traindir = os.path.join(args.image_path, 'train')
    valdir = os.path.join(args.image_path, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    trainloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers, pin_memory=True)

    valloader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers, pin_memory=True)
    print('preparing imagenet dataset completed')
    return trainloader, valloader

def load_cifar10(args):

    dataloader = datasets.CIFAR10

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

    trainset = dataloader(root=args.image_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers)

    testset = dataloader(root=args.image_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers)

    print('preparing dataset cifar10 completed')

    return trainloader, testloader

def load_cifar100(args):
    
    dataloader = datasets.CIFAR100
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

    trainset = dataloader(root=args.image_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_sizes[0], shuffle=args.shuffle[0], num_workers=args.num_workers)

    testset = dataloader(root=args.image_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_sizes[1], shuffle=args.shuffle[1], num_workers=args.num_workers)
    print('preparing dataset cifar100 completed')

    return trainloader, testloader

