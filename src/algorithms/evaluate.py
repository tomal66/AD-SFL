import torch
import numpy as np

@torch.no_grad()
def evaluate_accuracy(client_model, server_model, loader, device):
    """
    Standard test accuracy over the entire loader.
    """
    client_model.eval()
    server_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        z = client_model(x)
        logits = server_model(z)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)

@torch.no_grad()
def evaluate_backdoor_asr(client_model, server_model, loader, source_labels, target_label, trigger_args, device):
    """
    Attack Success Rate for Backdoor attacks.
    ASR = P( model(trigger(x)) == target_label | y in source_labels )
    """
    from src.data.attacks import apply_trigger_batch
    
    client_model.eval()
    server_model.eval()

    src = set(int(s) for s in (source_labels or []))
    if len(src) == 0:
        return 0.0

    total = 0
    success = 0

    for x, y in loader:
        y_np = y.cpu().numpy()
        mask = np.isin(y_np, list(src))
        if not mask.any():
            continue

        x_sel = x[torch.from_numpy(mask).to(x.device)]
        if x_sel.size(0) == 0:
            continue

        # stamp trigger
        x_bd = apply_trigger_batch(
            x_sel,
            trigger_size=trigger_args.get('trigger_size', 3),
            trigger_value_raw=trigger_args.get('trigger_value_raw', 1.0),
            trigger_location=trigger_args.get('trigger_pos', 'br'),
            dataset_name=trigger_args.get('dataset_name', 'MNIST')
        )

        x_bd = x_bd.to(device)
        logits = server_model(client_model(x_bd))
        pred = logits.argmax(dim=1)

        total += pred.numel()
        success += (pred == int(target_label)).sum().item()

    return success / max(total, 1)

@torch.no_grad()
def evaluate_targeted_asr(client_model, server_model, loader, source_labels, target_label, device):
    """
    Attack Success Rate for Targeted Attacks (model weights change).
    ASR = P( model(x) == target_label | y in source_labels )
    """
    client_model.eval()
    server_model.eval()

    src = set(int(s) for s in (source_labels or []))
    if len(src) == 0:
        return 0.0

    total = 0
    success = 0

    for x, y in loader:
        y_np = y.cpu().numpy()
        mask = np.isin(y_np, list(src))
        if not mask.any():
            continue

        x_sel = x[torch.from_numpy(mask).to(x.device)]
        if x_sel.size(0) == 0:
            continue

        x_sel = x_sel.to(device)
        logits = server_model(client_model(x_sel))
        pred = logits.argmax(dim=1)

        total += pred.numel()
        success += (pred == int(target_label)).sum().item()

    return success / max(total, 1)

@torch.no_grad()
def evaluate_pair_flip_asr(client_model, server_model, loader, label_pairs, device):
    """
    Attack Success Rate for Pair Flip attacks.
    ASR = P( model(x) == target | y == source ) over all pairs (source, target)
    """
    client_model.eval()
    server_model.eval()

    if not label_pairs:
        return 0.0

    total = 0
    success = 0

    # Build a lookup for quick target mapping
    pair_map = {src: tgt for src, tgt in label_pairs}

    for x, y in loader:
        # Vectorized implementation
        y_np = y.cpu().numpy()
        mask = np.isin(y_np, list(pair_map.keys()))
        if not mask.any():
            continue
            
        x_sel = x[torch.from_numpy(mask).to(x.device)]
        y_sel = y[torch.from_numpy(mask).to(y.device)]
        
        if x_sel.size(0) == 0:
            continue
            
        x_sel = x_sel.to(device)
        logits = server_model(client_model(x_sel))
        pred = logits.argmax(dim=1).cpu()  # Bring back to CPU for mapping check
        
        for i in range(pred.size(0)):
            src_label = int(y_sel[i].item())
            target_label = pair_map[src_label]
            total += 1
            if int(pred[i].item()) == target_label:
                success += 1

    return success / max(total, 1)
