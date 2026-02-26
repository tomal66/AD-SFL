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
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for both server and clients")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "CIFAR100", "ImageNet"])
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face access token for downloading datasets")
    
    # Attack Configuration
    parser.add_argument("--attack_type", type=str, default="none", choices=["none", "pair_flip", "targeted", "backdoor"])
    parser.add_argument("--malicious_fraction", type=float, default=0.0, help="Fraction of clients to be malicious")
    
    # Backdoor Arguments
    parser.add_argument("--backdoor_poison_fraction", type=float, default=1.0)
    parser.add_argument("--backdoor_target_label", type=int, default=0)
    parser.add_argument("--backdoor_source_labels", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--trigger_size", type=int, default=3)
    parser.add_argument("--trigger_value_raw", type=float, default=1.0)
    parser.add_argument("--trigger_pos", type=str, default="br")
    
    # Targeted Arguments
    parser.add_argument("--targeted_poison_fraction", type=float, default=0.3)
    parser.add_argument("--targeted_target_label", type=int, default=0)
    parser.add_argument("--targeted_source_labels", type=int, nargs="+", default=[1, 2, 3])
    
    # Pair Flip Arguments
    parser.add_argument("--flip_fraction", type=float, default=1.0)
    parser.add_argument("--label_pairs_to_flip", type=str, default="1:8,2:7,3:6", help="Comma separated pairs, e.g. 1:8,2:7")    
    
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

    server = SplitFedServer(model=server_model, num_clients=args.num_clients, lr=args.lr, device=device)

    # Pre-parse label pairs
    label_pairs = []
    if args.label_pairs_to_flip:
        pairs = args.label_pairs_to_flip.split(',')
        for p in pairs:
            if ':' in p:
                a, b = p.split(':')
                label_pairs.append((int(a), int(b)))

    # Gather attack kwargs
    attack_kwargs = {
        "backdoor_poison_fraction": args.backdoor_poison_fraction,
        "backdoor_target_label": args.backdoor_target_label,
        "backdoor_source_labels": args.backdoor_source_labels,
        "trigger_size": args.trigger_size,
        "trigger_value_raw": args.trigger_value_raw,
        "trigger_pos": args.trigger_pos,
        "targeted_poison_fraction": args.targeted_poison_fraction,
        "targeted_target_label": args.targeted_target_label,
        "targeted_source_labels": args.targeted_source_labels,
        "flip_fraction": args.flip_fraction,
        "label_pairs_to_flip": label_pairs,
    }
    
    # Determine malicious clients
    import numpy as np
    num_malicious = int(args.num_clients * args.malicious_fraction)
    malicious_clients_indices = set(np.random.choice(args.num_clients, num_malicious, replace=False))
    
    if num_malicious > 0:
        print(f"Selected malicious clients: {malicious_clients_indices}")

    # Initialize clients with their own instantiations of the client model and partitioned data
    clients = []
    from src.data.poisoned_dataset import PoisonedDataset
    
    for i in range(args.num_clients):
        # We start them with the same initial weights
        import copy
        c_model = copy.deepcopy(client_model_template)
        is_mal = i in malicious_clients_indices
        
        c_dataset = client_datasets[i]
        if is_mal and args.attack_type != "none":
            c_dataset = PoisonedDataset(c_dataset, attack_type=args.attack_type, 
                                        attack_kwargs=attack_kwargs, dataset_name=args.dataset,
                                        seed=42 + i)
            
        client = SplitFedClient(client_id=i, model=c_model, dataset=c_dataset, 
                                batch_size=args.batch_size, lr=args.lr, device=device,
                                is_malicious=is_mal)
        clients.append(client)

    from src.algorithms import run_sfl_round
    from src.algorithms.evaluate import (
        evaluate_accuracy, 
        evaluate_backdoor_asr, 
        evaluate_targeted_asr, 
        evaluate_pair_flip_asr
    )
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 3. Simulation Loop
    print("Starting Training Simulation...")
    for epoch in range(args.epochs):
        # We simulate the SFL-V1 process utilizing the external algorithm file
        train_loss, train_acc = run_sfl_round(clients, server)
        
        # Evaluate on the test set using the latest global weights (from client 0)
        eval_client = clients[0].model
        test_acc = evaluate_accuracy(eval_client, server.model, test_loader, device)
        
        asr = 0.0
        if args.attack_type == "backdoor":
            asr = evaluate_backdoor_asr(eval_client, server.model, test_loader, 
                                        args.backdoor_source_labels, args.backdoor_target_label, 
                                        attack_kwargs, device)
        elif args.attack_type == "targeted":
            asr = evaluate_targeted_asr(eval_client, server.model, test_loader, 
                                        args.targeted_source_labels, args.targeted_target_label, device)
        elif args.attack_type == "pair_flip":
            asr = evaluate_pair_flip_asr(eval_client, server.model, test_loader, 
                                         attack_kwargs["label_pairs_to_flip"], device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}, ASR: {asr:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
