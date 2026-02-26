import torch
from torch.utils.data import DataLoader
import numpy as np

class SplitFedClient:
    """
    Simulates a client in a Split Federated Learning setup.
    """
    def __init__(self, client_id, model, dataset, batch_size=32, lr=0.01, device='cpu', 
                 is_malicious=False):
        self.id = client_id
        self.model = model.to(device)
        self.dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True
        )
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.device = device
        
        # Attack identifier (logging only, poisoning happens natively in self.dataloader)
        self.is_malicious = is_malicious
        
        # Iterator to fetch batches across simulation epochs
        self.data_iterator = iter(self.dataloader)
        self.current_activations = None
        self.current_labels = None

    def reset_iterator(self):
        """Resets the data iterator for a new epoch/round."""
        self.data_iterator = iter(self.dataloader)

    def get_next_batch(self):
        try:
            data, target = next(self.data_iterator)
        except StopIteration:
            return None, None
        return data.to(self.device), target.to(self.device)



    def forward_pass(self, global_round=0):
        """
        Runs the forward pass up to the cut layer.
        Returns the smashed data/activations and labels, or (None, None) if exhausted.
        """
        data, target = self.get_next_batch()
        if data is None:
            return None, None

        self.model.train()
        self.optimizer.zero_grad()
        
        activations = self.model(data)
        
        self.current_activations = activations
        self.current_labels = target
        
        # Clone and require gradient to simulate sending over network
        smashed_data = activations.clone().detach().requires_grad_(True)
        return smashed_data, target

    def backward_pass(self, grad_from_server):
        """
        Receives the gradient of the loss with respect to the smashed data from the server,
        and computes the rest of the backward pass locally.
        """
        self.current_activations.backward(grad_from_server)
        self.optimizer.step()

    def get_weights(self):
        """Returns the client model's parameters."""
        return self.model.state_dict()

    def set_weights(self, weights):
        """Sets the client model's parameters (e.g., from server aggregation)."""
        self.model.load_state_dict(weights)
