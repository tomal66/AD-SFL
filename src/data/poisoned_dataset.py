import torch
import numpy as np
from torch.utils.data import Dataset
from src.data.attacks import stamp_trigger_chw

class PoisonedDataset(Dataset):
    """
    Statically poisons a dataset partition during initialization so that
    compromised items are injected at the __getitem__ level.
    """
    def __init__(self, dataset: Dataset, attack_type: str, attack_kwargs: dict, dataset_name: str = "MNIST", seed: int = 42):
        self.dataset = dataset
        self.attack_type = attack_type
        self.attack_kwargs = attack_kwargs
        self.dataset_name = dataset_name
        self.rng = np.random.default_rng(seed)
        
        self.poisoned_indices = set()
        self.target_labels_map = {}
        
        self._prepare_poisoning()

    def _prepare_poisoning(self):
        if self.attack_type in ["backdoor", "targeted"]:
            fraction = self.attack_kwargs.get(f"{self.attack_type}_poison_fraction", 1.0)
            src_labels = self.attack_kwargs.get(f"{self.attack_type}_source_labels", [])
            tgt_label = self.attack_kwargs.get(f"{self.attack_type}_target_label", 0)
            
            eligible = []
            for i in range(len(self.dataset)):
                _, y = self.dataset[i]
                if int(y) in src_labels:
                    eligible.append(i)
                    
            n_poison = int(len(eligible) * fraction)
            if n_poison > 0:
                chosen = self.rng.choice(eligible, n_poison, replace=False)
                for idx in chosen:
                    self.poisoned_indices.add(int(idx))
                    self.target_labels_map[int(idx)] = int(tgt_label)
                    
        elif self.attack_type == "pair_flip":
            fraction = self.attack_kwargs.get("flip_fraction", 1.0)
            label_pairs = self.attack_kwargs.get("label_pairs_to_flip", [])
            
            # Map source to target for all pairs identically to tensor logic
            pair_map = {}
            for src, tgt in label_pairs:
                pair_map[int(src)] = int(tgt)
                pair_map[int(tgt)] = int(src)
                
            eligible_by_class = {k: [] for k in pair_map.keys()}
            
            for i in range(len(self.dataset)):
                _, y = self.dataset[i]
                if int(y) in eligible_by_class:
                    eligible_by_class[int(y)].append(i)
                    
            for src_cls, indices in eligible_by_class.items():
                n_poison = int(len(indices) * fraction)
                if n_poison > 0:
                    chosen = self.rng.choice(indices, n_poison, replace=False)
                    for idx in chosen:
                        self.poisoned_indices.add(int(idx))
                        self.target_labels_map[int(idx)] = pair_map[src_cls]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        if idx in self.poisoned_indices:
            y = self.target_labels_map[idx]
            
            # Apply trigger if it's a backdoor attack
            if self.attack_type == "backdoor":
                trigger_size = self.attack_kwargs.get("trigger_size", 3)
                trigger_value = self.attack_kwargs.get("trigger_value_raw", 1.0)
                trigger_pos = self.attack_kwargs.get("trigger_pos", "br")
                
                x = stamp_trigger_chw(
                    x, 
                    dataset_name=self.dataset_name, 
                    trigger_size=trigger_size, 
                    trigger_value_raw=trigger_value,
                    location=trigger_pos
                )
        return x, y
