import argparse
import torch
from src.data.datasets import get_datasets
from src.data.partition import partition_data_iid
from src.models.split import ClientModel, ServerModel
from src.core.client import SplitFedClient
from src.core.server import SplitFedServer

def main():
    parser = argparse.ArgumentParser(description="SplitFed Simulation")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of simulated clients")
    parser.add_argument("--epochs", type=int, default=2, help="Number of global epochs (or iterations over clients)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per client")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Loading data...")
    train_data, test_data = get_datasets(args.dataset)
    client_datasets = partition_data_iid(train_data, args.num_clients)

    # 2. Initialize Models
    print("Initializing components...")
    # Need to match the input sizes
    if args.dataset == "MNIST":
        client_model_template = ClientModel(in_channels=1, hidden_channels=32)
        server_model = ServerModel(in_channels=32, hidden_channels=64, num_classes=10, input_size=(14, 14))
    else:  # CIFAR10
        client_model_template = ClientModel(in_channels=3, hidden_channels=32)
        server_model = ServerModel(in_channels=32, hidden_channels=64, num_classes=10, input_size=(16, 16))

    server = SplitFedServer(model=server_model, num_clients=args.num_clients, device=device)

    # Initialize clients with their own instantiations of the client model and partitioned data
    clients = []
    for i in range(args.num_clients):
        # We start them with the same initial weights
        import copy
        c_model = copy.deepcopy(client_model_template)
        client = SplitFedClient(client_id=i, model=c_model, dataset=client_datasets[i], 
                                batch_size=args.batch_size, device=device)
        clients.append(client)

    # 3. Simulation Loop
    print("Starting Training Simulation...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # In SplitFed v1, clients might train sequentially or in parallel. 
        # Here we simulate sequentially for simplicity.
        for client in clients:
            # Client forward pass to cut layer
            smashed_data, labels = client.forward_pass()
            
            # Server forward + loss + backward to cut layer
            grad_to_client, loss = server.train_step(smashed_data, labels)
            epoch_loss += loss
            
            # Client receives gradients and completes backward pass
            client.backward_pass(grad_to_client)
            
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss/args.num_clients:.4f}")
        
        # After an epoch (or some rounds), aggregate client models (e.g. SFL-V1/V2 differences)
        print("Aggregating client models...")
        client_weights = [c.get_weights() for c in clients]
        global_client_weights = server.aggregate_client_models(client_weights)
        
        # Broadcast back to clients
        for client in clients:
            client.set_weights(global_client_weights)

    print("Training finished.")

if __name__ == "__main__":
    main()
