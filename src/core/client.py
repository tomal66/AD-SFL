import torch
from torch.utils.data import DataLoader
import numpy as np
from src.data.attacks import (
    apply_trigger_batch,
    apply_targeted_label_attack_tensor,
    apply_backdoor_attack_tensor,
    apply_label_flipping_attack_multiple_pairs_tensor
)

class SplitFedClient:
    """
    Simulates a client in a Split Federated Learning setup.
    """
    def __init__(self, client_id, model, dataset, batch_size=32, lr=0.01, device='cpu', 
                 is_malicious=False, attack_type="none", attack_kwargs=None, dataset_name="MNIST"):
        self.id = client_id
        self.model = model.to(device)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.device = device
        
        # Attack Configurations
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        self.dataset_name = dataset_name
        self.attack_kwargs = attack_kwargs or {}
        
        # Dynamic Attack State
        self.dynamic_attack = self.attack_kwargs.get("dynamic_attack", False)
        self.attack_pattern = self.attack_kwargs.get("attack_pattern", "always")
        self.malicious_flip_prob = self.attack_kwargs.get("malicious_flip_prob", 1.0)
        self.cyclic_period = self.attack_kwargs.get("cyclic_attack_period", 5)
        self.periodic_interval = self.attack_kwargs.get("periodic_attack_interval", 3)
        
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

    def _should_attack_this_round(self, current_round: int) -> bool:
        if not self.is_malicious or self.attack_type == "none":
            return False
            
        if not self.dynamic_attack:
            return True
            
        if self.attack_pattern == "always":
            return True
        elif self.attack_pattern == "random":
            return np.random.rand() < self.malicious_flip_prob
        elif self.attack_pattern == "cyclic":
            cycle_length = 2 * self.cyclic_period
            pos_in_cycle = current_round % cycle_length
            return pos_in_cycle < self.cyclic_period
        elif self.attack_pattern == "periodic":
            return (current_round % self.periodic_interval) == 0
        else:
            return True

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
        
        if self._should_attack_this_round(global_round):
            if self.attack_type == "pair_flip":
                label_pairs = self.attack_kwargs.get("label_pairs_to_flip", [])
                flip_fraction = self.attack_kwargs.get("flip_fraction", 1.0)
                target = apply_label_flipping_attack_multiple_pairs_tensor(target, flip_fraction, label_pairs)
            
            elif self.attack_type == "targeted":
                poison_fraction = self.attack_kwargs.get("targeted_poison_fraction", 0.3)
                source_labels = self.attack_kwargs.get("targeted_source_labels", [])
                target_label = self.attack_kwargs.get("targeted_target_label", 0)
                target = apply_targeted_label_attack_tensor(target, poison_fraction, source_labels, target_label)
                
            elif self.attack_type == "backdoor":
                poison_fraction = self.attack_kwargs.get("backdoor_poison_fraction", 1.0)
                source_labels = self.attack_kwargs.get("backdoor_source_labels", [])
                target_label = self.attack_kwargs.get("backdoor_target_label", 0)
                trigger_size = self.attack_kwargs.get("trigger_size", 3)
                trigger_value = self.attack_kwargs.get("trigger_value_raw", 1.0)
                trigger_pos = self.attack_kwargs.get("trigger_pos", "br")
                
                data, target = apply_backdoor_attack_tensor(
                    data, target,
                    dataset_name=self.dataset_name,
                    poison_fraction=poison_fraction,
                    source_labels=source_labels,
                    target_label=target_label,
                    trigger_size=trigger_size,
                    trigger_value_raw=trigger_value,
                    trigger_location=trigger_pos
                )
        
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
