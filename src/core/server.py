import torch
import torch.nn as nn
import copy

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
        self.optimizers = [torch.optim.SGD(m.parameters(), lr=lr, **kwargs) for m in self.models]
        
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

    def aggregate_server_models(self):
        """
        Performs Federated Averaging (FedAvg) on the server side models.
        Updates the global server model and synchronizes all client-specific server models.
        """
        global_state = copy.deepcopy(self.models[0].state_dict())
        for key in global_state.keys():
            for i in range(1, self.num_clients):
                global_state[key] += self.models[i].state_dict()[key]
            
            if global_state[key].dtype.is_floating_point:
                global_state[key] = torch.div(global_state[key], self.num_clients)
            else:
                global_state[key] = torch.div(global_state[key], self.num_clients, rounding_mode='trunc')
                
        self.model.load_state_dict(global_state)
        for m in self.models:
            m.load_state_dict(global_state)

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
