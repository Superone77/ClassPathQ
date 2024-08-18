## /etc/datasets
import torch
import torchvision
import torchvision.transforms as transforms



def prepare_datasets(dataset_kind, batch_size, num_workers = 4, stat = False):
  if colab_flag:
    if dataset_kind == "CIFAR10":
        trainset_size = 40000
        valset_size = 10000

        transform_training = transforms.Compose(
        [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        transform_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        transform_val = transform_test

        index = torch.randperm(trainset_size + valset_size)
        index_training = index[:trainset_size]
        index_val = index[trainset_size:]

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_training)
        if stat == True:
            trainset = torch.utils.data.Subset(trainset, index_training)
        valset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_val)
        valset = torch.utils.data.Subset(valset, index_val)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers = num_workers)

    elif dataset_kind == "CIFAR100":
        trainset_size = 40000
        valset_size = 10000

        transform_training = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        transform_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])

        transform_val = transform_test

        index = torch.randperm(trainset_size + valset_size)
        index_training = index[:trainset_size]
        index_val = index[trainset_size:]

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_training)
        if stat == True:
            trainset = torch.utils.data.Subset(trainset, index_training)

        valset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_val)
        valset = torch.utils.data.Subset(valset, index_val)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers = num_workers)

    else:
        print("error, no such dataset")
        exit()

    return trainloader, valloader, testloader
  else:

    local_path = '/srv/data/image_dataset/'

    CIFAR10_root = local_path + 'CIFAR10/'
    CIFAR100_root = local_path + 'CIFAR100/'


    if dataset_kind == "CIFAR10":
        trainset_size = 40000
        valset_size = 10000

        transform_training = transforms.Compose(
        [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

        transform_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        transform_val = transform_test

        index = torch.randperm(trainset_size + valset_size)
        index_training = index[:trainset_size]
        index_val = index[trainset_size:]

        trainset = torchvision.datasets.CIFAR10(root=CIFAR10_root, train=True,
                                                download=False, transform=transform_training)
        if stat == True:
            trainset = torch.utils.data.Subset(trainset, index_training)
        valset = torchvision.datasets.CIFAR10(root=CIFAR10_root, train=True,
                                                download=False, transform=transform_val)
        valset = torch.utils.data.Subset(valset, index_val)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        testset = torchvision.datasets.CIFAR10(root=CIFAR10_root, train=False,
                                            download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers = num_workers)

    elif dataset_kind == "CIFAR100":
        trainset_size = 40000
        valset_size = 10000

        transform_training = transforms.Compose(
            [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        transform_test = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])

        transform_val = transform_test

        index = torch.randperm(trainset_size + valset_size)
        index_training = index[:trainset_size]
        index_val = index[trainset_size:]

        trainset = torchvision.datasets.CIFAR100(root=CIFAR100_root, train=True,
                                                download=False, transform=transform_training)
        if stat == True:
            trainset = torch.utils.data.Subset(trainset, index_training)

        valset = torchvision.datasets.CIFAR100(root=CIFAR100_root, train=True,
                                                download=False, transform=transform_val)
        valset = torch.utils.data.Subset(valset, index_val)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                                shuffle=True, num_workers = num_workers)

        testset = torchvision.datasets.CIFAR100(root=CIFAR100_root, train=False,
                                            download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers = num_workers)

    else:
        print("error, no such dataset")
        exit()

    return trainloader, valloader, testloader