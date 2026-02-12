
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
    def __init__(self, dataset, target_label=0, poison_fraction=0.1, mode='train', dataset_name='mnist', attack_type='backdoor', source_labels=None):
        self.dataset = dataset
        self.target_label = target_label
        self.poison_fraction = poison_fraction
        self.mode = mode
        self.poison_indices = []
        self.dataset_name = dataset_name
        self.attack_type = attack_type
        self.source_labels = source_labels  # List of labels to poison (backdoor only)

        # Pre-compute label map for label flipping if needed
        self.label_flip_map = {}
        if self.attack_type == 'label_flip':
            if self.dataset_name == 'mnist':
                # (1, 8), (2, 7), (3, 6)
                pairs = [(1, 8), (2, 7), (3, 6)]
                for a, b in pairs:
                    self.label_flip_map[a] = b
                    self.label_flip_map[b] = a
            elif self.dataset_name == 'cifar10':
                # automobile(1) <-> ship(8)
                # bird(2)       <-> horse(7)
                # cat(3)        <-> dog(5)
                pairs = [(1, 8), (2, 7), (3, 5)]
                for a, b in pairs:
                    self.label_flip_map[a] = b
                    self.label_flip_map[b] = a
        
        if mode == 'train':
            # Identify candidate indices for poisoning
            candidate_indices = []
            
            if self.attack_type == 'backdoor' and self.source_labels is not None:
                # Filter by source labels
                # We need to iterate to find them (or use targets if available)
                targets = None
                if hasattr(dataset, 'targets'):
                    targets = dataset.targets
                elif hasattr(dataset, 'labels'):
                    targets = dataset.labels
                
                if targets is not None:
                    # Convert to numpy if tensor or list
                    if not isinstance(targets, np.ndarray):
                         targets = np.array(targets)
                    
                    # Find indices where label is in source_labels
                    mask = np.isin(targets, self.source_labels)
                    candidate_indices = np.where(mask)[0]
                else:
                    # Fallback if no targets attribute (slower)
                    candidate_indices = [i for i in range(len(dataset)) if dataset[i][1] in self.source_labels]
            else:
                # All indices are candidates (for label_flip or backdoor with no source constraint)
                candidate_indices = np.arange(len(dataset))

            # Randomly select from candidates
            if len(candidate_indices) > 0:
                num_poison = int(len(candidate_indices) * poison_fraction)
                # If we want poison_fraction of the *entire* dataset or just the candidates? 
                # "poison_fraction" usually implies fraction of total data, or fraction of *malicious client's* data?
                # User says "clients implant ... into a subset of their local inputs". 
                # Usually poison_ratio is relative to the client's dataset size.
                # However, if source_labels restricts us, we can only poison those. 
                # Let's assume poison_fraction is "fraction of dataset to be poisoned". 
                # If we don't have enough candidates, we clamp.
                
                # RE-READING: "For each malicious client, samples drawn from a set of source classes are modified... 
                # while all remaining samples... are left unchanged."
                # Typically `poison_fraction` is the ratio of poisoned samples in the batch/dataset.
                
                cnt = int(len(dataset) * poison_fraction)
                if cnt > len(candidate_indices):
                    cnt = len(candidate_indices)
                
                self.poison_indices = np.random.choice(candidate_indices, cnt, replace=False)

        elif mode == 'test':
             # For test, we might want to poison all to measure ASR
             # But only valid source labels should be triggered for Backdoor ASR?
             # Standard ASR usually implies "inputs from source classes + trigger -> target class".
             # If source_labels is defined, we should only poison those for ASR measurement.
             if self.attack_type == 'backdoor' and self.source_labels is not None:
                 targets = None
                 if hasattr(dataset, 'targets'):
                     targets = dataset.targets
                 elif hasattr(dataset, 'labels'):
                     targets = dataset.labels
                
                 if targets is not None:
                     if not isinstance(targets, np.ndarray):
                          targets = np.array(targets)
                     mask = np.isin(targets, self.source_labels)
                     self.poison_indices = np.where(mask)[0]
                 else:
                     self.poison_indices = [i for i in range(len(dataset)) if dataset[i][1] in self.source_labels]
             else:
                self.poison_indices = np.arange(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # label_flip logic: check ALL samples if they match the pair
        # "Within each local mini-batch... all samples whose labels belong to these pairs are relabelled"
        # This implies it's deterministic for ALL matching samples, not just a random subset?
        # "malicious clients flip labels symmetrically... Within each local mini-batch ... all samples whose labels belong to these pairs are relabelled"
        # This sounds like 100% of susceptible samples are flipped if the client is malicious (poison_fraction=1.0 conceptually for those classes?)
        # But usually `poison_fraction` controls how "malicious" the client is or how many samples are affected.
        # IF attack_type is label_flip, standard practice might be to flip satisfying labels.
        # Let's interpret "poison_fraction" as "percentage of total data that is poisoned". 
        # But for label flipping, the constraint is usually class-based. 
        # "all samples whose labels belong to these pairs are relabelled" -> implies if you have the label, you get flipped.
        # But maybe we only do this for `poison_indices`? 
        # Text says: "Within each local mini-batch of a malicious client, all samples... are relabelled"
        # This typically means if the client is malicious, it does this to EVERYTHING it sees that matches the criteria.
        # So for label_flip, maybe we ignore `poison_fraction` and just flip everything?
        # Or maybe the user meant "malicious clients" are fully malicious.
        # But `Client` has `poison_fraction`.
        # I will stick to `poison_indices` for consistency. If poison_fraction=1.0, it affects all.
        # Wait, if poison_fraction=0.5, does it mean 50% of the dataset is subject to flipping?
        # The prompt says "malicious clients ... all samples ... are relabelled".
        # This suggests a malicious client does it to ALL their data (of those classes).
        # So if I am a malicious client, I should probably ignore `poison_indices` for label_flip and just check the label map?
        # Use `poison_fraction` only for Backdoor?
        # Let's check `simulation.py`: `p_frac = 1.0 if is_malicious else 0.0`. 
        # Ah! `simulation.py` sets poison_fraction to 1.0 for malicious clients.
        # So iterating `poison_indices` which covers the whole dataset (if 1.0) is correct.
        
        if idx in self.poison_indices:
            if self.attack_type == 'backdoor':
                # Apply trigger
                img = img.clone()
                if self.dataset_name == 'mnist':
                    # 4x4 white square at bottom right
                    img[:, 24:28, 24:28] = 2.8 
                elif self.dataset_name == 'cifar10':
                    # 4x4 white square at bottom right
                    img[:, 28:32, 28:32] = 2.5
                    
                label = self.target_label
            
            elif self.attack_type == 'label_flip':
                # Flip label if in map
                # Ensure we check the value properly (handle tensors)
                lbl_val = label.item() if hasattr(label, 'item') else label
                
                if lbl_val in self.label_flip_map:
                    # Update label. If original was tensor, do we keep it tensor? 
                    # Usually datasets return matching types.
                    # But if we change the value, we might just return the new value.
                    # Standard pytorch collation handles mixed types if needed, but usually we want consistent types.
                    # If original was tensor, let's try to return tensor, else int.
                    new_val = self.label_flip_map[lbl_val]
                    if hasattr(label, 'item') and not isinstance(label, int): # Check if it's a tensor-like
                         # Create a new 0-d tensor with the new value
                         label = torch.tensor(new_val, dtype=label.dtype, device=label.device if hasattr(label, 'device') else 'cpu')
                    else:
                         label = new_val
            
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
