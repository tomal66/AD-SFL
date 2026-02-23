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

def partition_data_noniid(dataset, num_clients, num_classes=10):
    """
    A simple non-IID partitioning function (e.g. sorting by label and splitting).
    """
    # For a full non-iid implementation, you'd sort the dataset by labels, 
    # divide it into shards, and assign shards to clients.
    # We will raise NotImplementedError here as a placeholder for advanced logic.
    raise NotImplementedError("Non-IID partitioning not implemented in this demo.")
