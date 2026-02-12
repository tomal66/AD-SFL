
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

def get_mnist_datasets(root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def get_cifar10_datasets(root='./data'):
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
    train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset

class PoisonedDataset(Dataset):
    def __init__(self, dataset, target_label=0, poison_fraction=0.1, mode='train', dataset_name='mnist'):
        self.dataset = dataset
        self.target_label = target_label
        self.poison_fraction = poison_fraction
        self.mode = mode
        self.poison_indices = []
        self.dataset_name = dataset_name
        
        if mode == 'train':
            # Randomly select indices to poison
            num_poison = int(len(dataset) * poison_fraction)
            self.poison_indices = np.random.choice(len(dataset), num_poison, replace=False)
        elif mode == 'test':
             # For test, we might want to poison all to measure ASR
             self.poison_indices = np.arange(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if idx in self.poison_indices:
            # Apply trigger
            img = img.clone()
            if self.dataset_name == 'mnist':
                # 4x4 white square at bottom right
                img[:, 24:28, 24:28] = 2.8 
            elif self.dataset_name == 'cifar10':
                # 4x4 white square at bottom right, for all channels
                # Max value normalized approx (assuming ~2.0-2.5 max depending on channel)
                # Setting to a high value like 2.5 is sufficient to be visible/distinct
                img[:, 28:32, 28:32] = 2.5
                
            label = self.target_label
            
        return img, label

def split_data(dataset, num_clients, distribution='iid', alpha=0.5):
    """
    Split dataset into num_clients partitions.
    distribution: 'iid' or 'non_iid'
    alpha: Dirichlet concentration parameter (for non_iid)
    """
    data_len = len(dataset)
    indices = np.arange(data_len)

    if distribution == 'iid':
        # Simple IID split
        split_size = data_len // num_clients
        # Random shuffle for IID
        np.random.shuffle(indices)
        
        client_datasets = []
        for i in range(num_clients):
            subset_indices = indices[i*split_size : (i+1)*split_size]
            client_datasets.append(Subset(dataset, subset_indices))
        return client_datasets

    elif distribution == 'non_iid':
        # Dirichlet split
        # Only works for datasets with .targets or .labels (like MNIST/CIFAR)
        try:
            labels = np.array(dataset.targets)
        except AttributeError:
             try:
                 labels = np.array(dataset.labels)
             except AttributeError:
                 raise ValueError("Dataset does not have .targets or .labels attribute for non-IID split.")

        num_classes = len(np.unique(labels))
        min_size = 0
        
        # Retry until all clients have at least min_size samples
        while min_size < 10:
            client_idcs = [[] for _ in range(num_clients)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                
                # Sample proportions from Dirichlet
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # Balance: Scale proportions to match label count
                # Start indices for this label
                proportions = np.array([p * (len(idx_k) < num_clients and 1/num_clients or 1) for p in proportions])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                
                client_splits = np.split(idx_k, proportions)
                for i in range(num_clients):
                    client_idcs[i] += client_splits[i].tolist()

            min_size = min([len(c) for c in client_idcs])

        client_datasets = []
        for i in range(num_clients):
            client_datasets.append(Subset(dataset, client_idcs[i]))
        
        return client_datasets

    else:
        raise ValueError(f"Unknown distribution: {distribution}")
