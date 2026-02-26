import torch
import numpy as np
from typing import List, Tuple

def _trigger_value_normalized(trigger_value_raw: float, dataset_name: str = "MNIST"):
    """
    Convert raw [0,1] pixel value to normalized value used by your dataset transform.
    Returns either a float (MNIST) or a torch tensor [C] (CIFAR).
    """
    if dataset_name.upper() == "CIFAR10" or dataset_name.upper() == "CIFAR100":
        mean = torch.tensor((0.4914, 0.4822, 0.4465), dtype=torch.float32)
        std  = torch.tensor((0.2023, 0.1994, 0.2010), dtype=torch.float32)
        v = (trigger_value_raw - mean) / std
        return v  # shape [3]
    elif dataset_name.upper() == "IMAGENET":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        v = (trigger_value_raw - mean) / std
        return v
    else:  # mnist
        mean = 0.1307
        std  = 0.3081
        return (float(trigger_value_raw) - mean) / std

def stamp_trigger_chw(x_img_chw: torch.Tensor, dataset_name: str = "MNIST", trigger_size=3, trigger_value_raw=1.0, location="br"):
    """
    x_img_chw: torch tensor [C,H,W] already NORMALIZED (because your pipeline normalizes).
    Stamps a constant trigger patch using the trigger value converted to normalized space.
    """
    x = x_img_chw.clone()
    C, H, W = x.shape

    if location == "br":
        r0 = H - trigger_size
        c0 = W - trigger_size
    elif isinstance(location, (tuple, list)) and len(location) == 2:
        r0, c0 = int(location[0]), int(location[1])
    else:
        raise ValueError("Trigger location must be 'br' or (row, col)")

    r0 = max(0, min(r0, H - trigger_size))
    c0 = max(0, min(c0, W - trigger_size))

    tv = _trigger_value_normalized(trigger_value_raw, dataset_name=dataset_name)

    if dataset_name.upper() in ["CIFAR10", "CIFAR100", "IMAGENET"]:
        # tv is [3]; broadcast into [3, trigger_size, trigger_size]
        tv = tv.to(x.dtype).view(C, 1, 1)
        x[:, r0:r0 + trigger_size, c0:c0 + trigger_size] = tv
    else:
        # tv is scalar
        x[:, r0:r0 + trigger_size, c0:c0 + trigger_size] = float(tv)

    return x

def apply_trigger_batch(x_batch: torch.Tensor, dataset_name: str = "MNIST", trigger_size=3, trigger_value_raw=1.0, trigger_location="br"):
    """
    x_batch: [B,C,H,W] already NORMALIZED.
    """
    x_bd = x_batch.clone()
    for i in range(x_bd.size(0)):
        x_bd[i] = stamp_trigger_chw(
            x_bd[i],
            dataset_name=dataset_name,
            trigger_size=trigger_size,
            trigger_value_raw=trigger_value_raw,
            location=trigger_location,
        )
    return x_bd

def apply_targeted_label_attack_tensor(
    y_data: torch.Tensor,
    poison_fraction=0.3,
    source_labels=None,
    target_label=0,
    seed=None
):
    """
    Label-only targeted: pick poison_fraction of samples with y in source_labels -> set to target_label.
    """
    if source_labels is None or len(source_labels) == 0:
        return y_data

    rng = np.random.default_rng(seed)
    y = y_data.clone()

    src = torch.tensor(source_labels, device=y.device, dtype=y.dtype)
    eligible = torch.where(torch.isin(y, src))[0].cpu().numpy()
    if len(eligible) == 0:
        return y

    n_poison = int(len(eligible) * poison_fraction)
    if n_poison <= 0:
        return y

    poison_idx = rng.choice(eligible, size=n_poison, replace=False)
    y[torch.from_numpy(poison_idx).to(y.device)] = int(target_label)
    return y

def apply_backdoor_attack_tensor(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    dataset_name: str = "MNIST",
    poison_fraction=1.0,
    source_labels=None,
    target_label=0,
    trigger_size=3,
    trigger_value_raw=1.0,
    trigger_location="br",
    seed=None,
):
    """
    Backdoor: pick poison_fraction of samples with y in source_labels,
    stamp trigger on x, relabel to target_label.
    x_data is already normalized in your pipeline.
    """
    if source_labels is None or len(source_labels) == 0:
        return x_data, y_data

    rng = np.random.default_rng(seed)
    x = x_data.clone()
    y = y_data.clone()

    src = torch.tensor(source_labels, device=y.device, dtype=y.dtype)
    eligible = torch.where(torch.isin(y, src))[0].cpu().numpy()
    if len(eligible) == 0:
        return x, y

    n_poison = int(len(eligible) * poison_fraction)
    if n_poison <= 0:
        return x, y

    poison_idx = rng.choice(eligible, size=n_poison, replace=False)
    poison_idx_t = torch.from_numpy(poison_idx).to(x.device)

    # stamp + relabel
    for idx in poison_idx_t:
        x[idx] = stamp_trigger_chw(
            x[idx],
            dataset_name=dataset_name,
            trigger_size=trigger_size,
            trigger_value_raw=trigger_value_raw,
            location=trigger_location,
        )
        y[idx] = int(target_label)

    return x, y

def apply_label_flipping_attack_multiple_pairs_tensor(y_data: torch.Tensor, flip_fraction: float, label_pairs: List[Tuple[int,int]]):
    y = y_data.clone()
    for a, b in label_pairs:
        idx_a = (y == a).nonzero(as_tuple=False).squeeze()
        idx_b = (y == b).nonzero(as_tuple=False).squeeze()
        
        # If scalar tensor (0-dim), .squeeze() leaves it 0-dim. .numel() handles this correctly.
        if idx_a.numel() == 0 and idx_b.numel() == 0:
            continue
            
        # Ensure they are 1D arrays for slicing
        if idx_a.numel() == 1 and idx_a.dim() == 0:
            idx_a = idx_a.unsqueeze(0)
        if idx_b.numel() == 1 and idx_b.dim() == 0:
            idx_b = idx_b.unsqueeze(0)
            
        if idx_a.numel() > 0:
            n_a = int(idx_a.numel() * flip_fraction)
            if n_a > 0:
                flip_a = idx_a[torch.randperm(idx_a.numel())[:n_a]]
                y[flip_a] = b
        if idx_b.numel() > 0:
            n_b = int(idx_b.numel() * flip_fraction)
            if n_b > 0:
                flip_b = idx_b[torch.randperm(idx_b.numel())[:n_b]]
                y[flip_b] = a
    return y
