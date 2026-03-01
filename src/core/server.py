import torch
import torch.nn as nn
import copy

_BN_BUFFER_KEYS = ("running_mean", "running_var", "num_batches_tracked")
def _is_bn_buffer_key(k: str) -> bool:
    return any(s in k for s in _BN_BUFFER_KEYS)

class SplitFedServer:
    """
    Simulates the central server in a Split Federated Learning setup.
    """
    def __init__(self, model, num_clients, lr=0.01, device='cpu', **kwargs):
        self.device = device
        self.num_clients = num_clients
        
        # Global server model parameters
        self.model = model.to(device)
        
        # For FedAvg on server-side, we maintain a separate server model instance for each client
        self.models = [copy.deepcopy(model).to(device) for _ in range(num_clients)]
        
        # Each server model gets its own optimizer
        self.optimizers = [
            torch.optim.SGD(
                m.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True
            )
            for m in self.models
        ]
                
        # Maintain a dummy optimizer attribute for compatibility
        self.optimizer = self.optimizers[0] 
        
        self.criterion = nn.CrossEntropyLoss()
        
        # To store aggregated client models (if utilizing FedAvg on client side of split network)
        self.client_weights = []

    def train_step(self, smashed_data, labels, client_id):
        """
        Completes the forward pass from the cut layer, calculates loss,
        and returns the gradients to be sent back to the client.
        """
        model = self.models[client_id]
        optimizer = self.optimizers[client_id]
        
        model.train()
        optimizer.zero_grad()
        smashed_data = smashed_data.to(self.device)
        labels = labels.to(self.device)

        # Forward
        outputs = model(smashed_data)
        loss = self.criterion(outputs, labels)

        # Calculate Accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0

        # Backward
        loss.backward()
        optimizer.step()

        # The gradients with respect to the input of the server model (smashed data)
        grad_to_client = smashed_data.grad.clone().detach()

        return grad_to_client, loss.item(), accuracy

    def aggregate_server_models(self, active_client_indices=None, weights=None, skip_bn_buffers=True):
        if active_client_indices is None:
            active_client_indices = list(range(self.num_clients))

        if weights is None:
            weights = [1.0] * len(active_client_indices)

        total_w = float(sum(weights))
        if total_w <= 0 or len(active_client_indices) == 0:
            return

        first = active_client_indices[0]
        global_state = copy.deepcopy(self.models[first].state_dict())

        for k in global_state.keys():
            if skip_bn_buffers and _is_bn_buffer_key(k):
                continue

            v = global_state[k]
            if torch.is_floating_point(v):
                global_state[k] = v * (weights[0] / total_w)
                for idx, w in zip(active_client_indices[1:], weights[1:]):
                    global_state[k] += self.models[idx].state_dict()[k] * (w / total_w)
            else:
                global_state[k] = v  # keep

        self.model.load_state_dict(global_state, strict=False)
        for m in self.models:
            m.load_state_dict(global_state, strict=False)

    def aggregate_client_models(self, client_weights_list):
        """
        Performs Federated Averaging (FedAvg) on the client side models.
        Returns the aggregated weights to be broadcast to all clients.
        """
        aggregated_weights = copy.deepcopy(client_weights_list[0])
        for key in aggregated_weights.keys():
            for i in range(1, len(client_weights_list)):
                aggregated_weights[key] += client_weights_list[i][key]
                
            if aggregated_weights[key].dtype.is_floating_point:
                aggregated_weights[key] = torch.div(aggregated_weights[key], len(client_weights_list))
            else:
                aggregated_weights[key] = torch.div(aggregated_weights[key], len(client_weights_list), rounding_mode='trunc')
                
        return aggregated_weights

    def aggregate_client_models_weighted(self, client_payloads):
        # client_payloads: List[(state_dict, n_samples)]
        total = sum(n for _, n in client_payloads)
        aggregated = copy.deepcopy(client_payloads[0][0])

        for k in aggregated.keys():
            # init with first client weighted
            aggregated[k] = aggregated[k] * (client_payloads[0][1] / total)

            for sd, n in client_payloads[1:]:
                aggregated[k] += sd[k] * (n / total)

        return aggregated
