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
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100", "ImageNet"])
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token for downloading datasets")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Loading data...")
    train_data, test_data = get_datasets(args.dataset, hf_token=args.hf_token)
    client_datasets = partition_data_iid(train_data, args.num_clients)

    # 2. Initialize Models
    from src.models.split import get_split_models
    client_model_template, server_model = get_split_models(args.dataset)

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

    from src.algorithms import run_sfl_round
    
    # 3. Simulation Loop
    print("Starting Training Simulation...")
    for epoch in range(args.epochs):
        # We simulate the SFL-V1 process utilizing the external algorithm file
        avg_loss, avg_acc = run_sfl_round(clients, server)
            
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
