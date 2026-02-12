
# simulation.py
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from models import get_models, Head, Backbone, Tail, ResNetHead, ResNetBackbone, ResNetTail
from safesplit import SafeSplit
from data import PoisonedDataset

ClientState = Dict[str, Dict]  # {'head': state_dict, 'tail': state_dict}

class Client:
    def __init__(self, client_id, dataset, device="cpu", poison_fraction=0.0, target_label=0, dataset_name='mnist', lr=0.01, momentum=0.9, batch_size=32):
        self.client_id = client_id
        self.device = device
        self.dataset = dataset
        self.dataset_name = dataset_name # Store for poisoning logic if needed later

        # Use factory to get models. Server backbone will be created separately but must match.
        # We only need Head and Tail here.
        h, _, t = get_models(dataset_name, device)
        self.head = h
        self.tail = t

        self.optimizer_h = optim.SGD(self.head.parameters(), lr=lr, momentum=momentum)
        self.optimizer_t = optim.SGD(self.tail.parameters(), lr=lr, momentum=momentum)

        self.is_malicious = poison_fraction > 0
        if self.is_malicious:
            self.dataset = PoisonedDataset(dataset, target_label, poison_fraction, mode="train", dataset_name=dataset_name)

        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

    def train_epoch(self, server):
        self.head.train()
        self.tail.train()

        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # ---- Head forward ----
            self.optimizer_h.zero_grad()
            h_out = self.head(inputs)

            # smash data sent to server
            h_out_sent = h_out.clone().detach().requires_grad_(True)

            # ---- Backbone forward (server) ----
            b_out_sent = server.process_forward(h_out_sent)

            # ---- Tail forward + loss ----
            self.optimizer_t.zero_grad()
            outputs = self.tail(b_out_sent)

            loss = criterion(outputs, labels)
            loss.backward()

            # ---- Tail update ----
            self.optimizer_t.step()

            # gradients to server
            grad_to_server = b_out_sent.grad.clone()

            # ---- Backbone backward (server) ----
            grad_to_client = server.process_backward(grad_to_server)

            # ---- Head backward ----
            h_out.backward(grad_to_client)
            self.optimizer_h.step()

            total_loss += float(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += int(predicted.eq(labels).sum().item())

        return total_loss / max(1, len(self.dataloader)), 100.0 * correct / max(1, total)

    def get_models(self) -> ClientState:
        return {
            "head": copy.deepcopy(self.head.state_dict()),
            "tail": copy.deepcopy(self.tail.state_dict()),
        }

    def set_models(self, state_dicts: ClientState):
        self.head.load_state_dict(state_dicts["head"])
        self.tail.load_state_dict(state_dicts["tail"])


class Server:
    def __init__(self, device="cpu", defense_enabled=True, num_clients=5, low_freq_ratio=0.25, dataset_name='mnist', lr=0.01, momentum=0.9):
        self.device = device
        
        # Initialize backbone based on dataset
        _, b, _ = get_models(dataset_name, device)
        self.backbone = b
        
        self.optimizer_b = optim.SGD(self.backbone.parameters(), lr=lr, momentum=momentum)

        # SafeSplit uses N = number of clients
        self.defense = SafeSplit(N=num_clients, defense_enabled=defense_enabled, low_freq_ratio=low_freq_ratio)

        # History of full checkpoints aligned with backbone history:
        # each item is (head_state, backbone_state, tail_state)
        self.full_ckpt_history: list[Tuple[Dict, Dict, Dict]] = []

    # ---------- Split learning backbone steps ----------
    def process_forward(self, client_interm):
        self.optimizer_b.zero_grad()
        self._current_client_interm = client_interm
        self._current_server_interm = self.backbone(client_interm)
        return self._current_server_interm.clone().detach().requires_grad_(True)

    def process_backward(self, grad_from_tail):
        self._current_server_interm.backward(grad_from_tail)
        self.optimizer_b.step()
        return self._current_client_interm.grad.clone()

    # ---------- SafeSplit integration ----------
    def record_checkpoint(self, head_state: Dict, tail_state: Dict):
        """Record current (H,B,T) into FIFO-aligned histories."""
        b_state = copy.deepcopy(self.backbone.state_dict())

        # backbone FIFO (N+1 kept inside SafeSplit)
        self.defense.update_history(b_state)

        # full checkpoint FIFO: keep same length as defense.backbone_history
        self.full_ckpt_history.append((copy.deepcopy(head_state), b_state, copy.deepcopy(tail_state)))
        # Trim to at most N+1 so indices match
        while len(self.full_ckpt_history) > self.defense.N + 1:
            self.full_ckpt_history.pop(0)

    def select_benign_checkpoint(self) -> Optional[Tuple[Dict, Dict]]:
        """
        Run SafeSplit and, if rollback needed, load backbone and return (head_state, tail_state)
        for the selected benign checkpoint.

        Returns None if no rollback (use latest state).
        """
        rollback_hist_index = self.defense.check_for_rollback()
        if rollback_hist_index is None:
            return None

        # Roll back backbone to selected history checkpoint
        head_state, b_state, tail_state = self.full_ckpt_history[rollback_hist_index]
        self.backbone.load_state_dict(b_state)
        return head_state, tail_state


def run_simulation(config):
    num_clients = config.get("num_clients", 5)
    rounds = config.get("rounds", 10)
    poison_ratio = config.get("poison_ratio", 0.0)
    dataset_name = config.get("dataset", "mnist").lower() # New config param
    
    # Training Hyperparameters
    lr = config.get("lr", 0.01)
    momentum = config.get("momentum", 0.9)
    batch_size = config.get("batch_size", 32)
    test_batch_size = config.get("test_batch_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data dispatch
    from data import get_mnist_datasets, get_cifar10_datasets, split_data
    
    if dataset_name == 'mnist':
        train_data, test_data = get_mnist_datasets()
    elif dataset_name == 'cifar10':
        train_data, test_data = get_cifar10_datasets()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    client_datasets = split_data(train_data, num_clients, 
                                 distribution=config.get("distribution", "iid"),
                                 alpha=config.get("alpha", 0.5))

    # Server (SafeSplit uses N=num_clients)
    server = Server(
        device=device,
        defense_enabled=config.get("defense_enabled", True),
        num_clients=num_clients,
        low_freq_ratio=config.get("low_freq_ratio", 0.25),
        dataset_name=dataset_name, # Pass dataset name
        lr=lr,
        momentum=momentum
    )

    # Clients
    clients = []
    num_malicious = int(num_clients * poison_ratio)

    for i in range(num_clients):
        is_malicious = i < num_malicious
        p_frac = 1.0 if is_malicious else 0.0
        ds = client_datasets[i]
        clients.append(
            Client(
                client_id=i,
                dataset=ds,
                device=device,
                poison_fraction=p_frac,
                target_label=config.get("target_label", 0),
                dataset_name=dataset_name, # Pass dataset name
                lr=lr,
                momentum=momentum,
                batch_size=batch_size
            )
        )

    # Initialize global head/tail (H0,T0). Backbone is initialized in server.
    current_client_state = clients[0].get_models()

    # Record initial checkpoint (H0,B0,T0) into histories so that SafeSplit can form windows.
    server.record_checkpoint(current_client_state["head"], current_client_state["tail"])

    history = {"loss": [], "acc": [], "asr": []}
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    # Evaluation helper
    def evaluate(server_obj: Server, head_state, tail_state, loader, target_label=None):
        # Instantiate eval models based on dataset
        h_eval, _, t_eval = get_models(dataset_name, device)
        
        h_eval.load_state_dict(head_state)
        t_eval.load_state_dict(tail_state)

        h_eval.eval()
        t_eval.eval()
        server_obj.backbone.eval()

        correct = 0
        total = 0
        success = 0
        poison_total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # ASR
                if target_label is not None:
                    p_inputs = inputs.clone()
                    if dataset_name == 'mnist':
                        p_inputs[:, :, 24:28, 24:28] = 2.8  # trigger
                    elif dataset_name == 'cifar10':
                        p_inputs[:, :, 28:32, 28:32] = 2.5 # trigger

                    h_out = h_eval(p_inputs)
                    b_out = server_obj.backbone(h_out)
                    outputs = t_eval(b_out)
                    _, predicted = outputs.max(1)

                    success += int(predicted.eq(target_label).sum().item())
                    poison_total += labels.size(0)

                # Clean accuracy
                h_out = h_eval(inputs)
                b_out = server_obj.backbone(h_out)
                outputs = t_eval(b_out)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += int(predicted.eq(labels).sum().item())

        server_obj.backbone.train()
        acc = 100.0 * correct / max(1, total)
        asr = 100.0 * success / max(1, poison_total) if target_label is not None else 0.0
        return acc, asr

    print(f"Starting Training for {rounds} rounds on {dataset_name}...")

    for r in range(rounds):
        round_loss = 0.0

        for i in range(num_clients):
            client = clients[i]

            # Before client starts, SafeSplit may have rolled back after previous client.
            # current_client_state always holds the (H,T) to use next.
            client.set_models(current_client_state)

            # Train one client epoch (updates H,T locally; updates B on server)
            loss, _ = client.train_epoch(server)
            round_loss += loss

            finished_state = client.get_models()

            # Record checkpoint after this client's training: (H_t, B_t, T_t)
            server.record_checkpoint(finished_state["head"], finished_state["tail"])

            # Run SafeSplit to decide which checkpoint should be used as the starting point
            # for the next client.
            rolled = server.select_benign_checkpoint()
            if rolled is None:
                # Keep latest state
                current_client_state = finished_state
            else:
                # Rollback to benign (H,T) matching loaded backbone
                head_state, tail_state = rolled
                current_client_state = {"head": head_state, "tail": tail_state}

        # End-of-round eval
        val_acc, val_asr = evaluate(
            server,
            current_client_state["head"],
            current_client_state["tail"],
            test_loader,
            target_label=config.get("target_label", 0),
        )

        print(
            f"Round {r+1}/{rounds} - Loss: {round_loss/num_clients:.4f} "
            f"- Val Acc: {val_acc:.2f}% - ASR: {val_asr:.2f}%"
        )
        history["loss"].append(round_loss / num_clients)
        history["acc"].append(val_acc)
        history["asr"].append(val_asr)

    return history
