import numpy as np
from torch.utils.data import Subset

def partition_data_iid(dataset, num_clients):
    """
    Partitions the dataset in an IID fashion among num_clients.
    Returns a list of Subsets, one for each client.
    """
    num_items = int(len(dataset) / num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    client_datasets = []
    
    for i in range(num_clients):
        # Sample num_items from all_idxs without replacement
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # Remove those from all_idxs
        all_idxs = list(set(all_idxs) - dict_users[i])
        
        # Create Subset for this client
        client_subset = Subset(dataset, list(dict_users[i]))
        client_datasets.append(client_subset)
        
    return client_datasets

def partition_data_dirichlet(dataset, num_clients, alpha=0.5, num_classes=10):
    """
    Partitions the dataset in a non-IID fashion using a Dirichlet distribution.
    alpha: concentration parameter of the Dirichlet distribution. 
           Lower alpha -> higher data heterogeneity (more non-IID).
    Returns a list of Subsets, one for each client.
    """
    try:
        targets = np.array(dataset.targets)
    except AttributeError:
        # Fallback if targets is not an attribute
        targets = np.array([y for _, y in dataset])
    
    client_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        idx_k = np.where(targets == c)[0]
        np.random.shuffle(idx_k)
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Convert proportions to counts
        counts = (proportions * len(idx_k)).astype(int)
        
        # Handle rounding errors by giving the remainder to the clients randomly
        diff = len(idx_k) - counts.sum()
        for i in range(diff):
            counts[np.random.randint(0, num_clients)] += 1
            
        # Split idx_k according to counts
        splits = np.split(idx_k, np.cumsum(counts)[:-1])
        
        for i in range(num_clients):
            client_indices[i].extend(splits[i].tolist())
            
    client_datasets = []
    for indices in client_indices:
        np.random.shuffle(indices)
        client_datasets.append(Subset(dataset, indices))
        
    return client_datasets
