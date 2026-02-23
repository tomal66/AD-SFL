import torch
import torch.nn as nn
import copy

class SplitFedServer:
    """
    Simulates the central server in a Split Federated Learning setup.
    """
    def __init__(self, model, num_clients, lr=0.01, device='cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        
        # To store aggregated client models (if utilizing FedAvg on client side of split network)
        self.num_clients = num_clients
        self.client_weights = []

    def train_step(self, smashed_data, labels):
        """
        Completes the forward pass from the cut layer, calculates loss,
        and returns the gradients to be sent back to the client.
        """
        self.model.train()
        self.optimizer.zero_grad()
        smashed_data = smashed_data.to(self.device)
        labels = labels.to(self.device)

        # Forward
        outputs = self.model(smashed_data)
        loss = self.criterion(outputs, labels)

        # Calculate Accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total if total > 0 else 0.0

        # Backward
        loss.backward()
        self.optimizer.step()

        # The gradients with respect to the input of the server model (smashed data)
        grad_to_client = smashed_data.grad.clone().detach()

        return grad_to_client, loss.item(), accuracy

    def aggregate_client_models(self, client_weights_list):
        """
        Performs Federated Averaging (FedAvg) on the client side models.
        Returns the aggregated weights to be broadcast to all clients.
        """
        aggregated_weights = copy.deepcopy(client_weights_list[0])
        for key in aggregated_weights.keys():
            for i in range(1, len(client_weights_list)):
                aggregated_weights[key] += client_weights_list[i][key]
            aggregated_weights[key] = torch.div(aggregated_weights[key], len(client_weights_list))
        return aggregated_weights
