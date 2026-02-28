import os
import sys
import copy
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.datasets import get_datasets
from src.data.partition import partition_data_iid, partition_data_dirichlet
from src.models.split import get_split_models
from src.core.client import SplitFedClient
from src.core.server import SplitFedServer
from src.core.seed import set_seed
from src.data.poisoned_dataset import PoisonedDataset
from src.algorithms import run_sfl_round, run_sfl_gold_round, run_sfl_centinel_round, CentinelState
from src.algorithms.evaluate import (
    evaluate_accuracy,
    evaluate_backdoor_asr,
    evaluate_targeted_asr,
    evaluate_pair_flip_asr
)

def main():
    parser = argparse.ArgumentParser(description="Run SFL Experiment")
    parser.add_argument("--baseline", type=str, required=True, choices=['sfl', 'sfl_gold', 'centinel'])
    parser.add_argument("--malicious_fraction", type=float, required=True)
    parser.add_argument("--attack_type", type=str, required=True, choices=['none', 'backdoor', 'pair_flip'])
    parser.add_argument("--partition", type=str, required=True, choices=['iid', 'non_iid'])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    # Hardware Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Global Config
    num_clients = 10
    batch_size = 64
    learning_rate = 3e-3
    GLOBAL_SEED = 42
    DATASET = "CIFAR100"
    
    set_seed(GLOBAL_SEED)

    # Attack configuration mapping depending on CIFAR100 labels (100 total)
    ATTACK_KWARGS = {}
    if args.attack_type == 'backdoor':
        ATTACK_KWARGS.update({
            "backdoor_poison_fraction": 1.0,
            "backdoor_target_label": 0,
            "backdoor_source_labels": [1, 2, 3],
            "trigger_size": 3,
            "trigger_value_raw": 1.0,
            "trigger_pos": "br"
        })
    elif args.attack_type == 'pair_flip':
        ATTACK_KWARGS.update({
            "flip_fraction": 1.0,
            # Just an example mapping, valid indices are 0-99
            "label_pairs_to_flip": [(1, 8), (2, 7), (3, 9)]
        })

    # 1. Load Data
    train_data, test_data = get_datasets(dataset_name=DATASET)
    if args.partition == 'iid':
        client_datasets = partition_data_iid(train_data, num_clients)
    else:
        # non_iid with dirichlet alpha 0.5
        client_datasets = partition_data_dirichlet(train_data, num_clients, alpha=0.5)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    ref_loader = test_loader # use test set as reference for centinel

    # 2. Init Models
    # User requested default pretrained, so it uses `weights="DEFAULT"` or automatically
    # assuming ResNet/WideResNet definitions handle it. Let's just pass weights="DEFAULT"
    base_client_model, server_model = get_split_models(DATASET, weights="DEFAULT")
    server = SplitFedServer(model=server_model, num_clients=num_clients, lr=learning_rate, device=device)

    # Determine Malicious Clients
    num_malicious = int(num_clients * args.malicious_fraction)
    malicious_clients_indices = set(np.random.choice(num_clients, num_malicious, replace=False))

    clients = []
    for i in range(num_clients):
        client_model = copy.deepcopy(base_client_model)
        is_mal = i in malicious_clients_indices
        c_dataset = client_datasets[i]
        
        if is_mal and args.attack_type != 'none':
            c_dataset = PoisonedDataset(
                c_dataset, attack_type=args.attack_type, 
                attack_kwargs=ATTACK_KWARGS, dataset_name=DATASET, seed=GLOBAL_SEED+i
            )
            
        client = SplitFedClient(
            client_id=i, model=client_model, dataset=c_dataset,
            batch_size=batch_size, lr=learning_rate, device=device, is_malicious=is_mal
        )
        clients.append(client)

    # 3. Choose Algorithm and State
    if args.baseline == 'centinel':
        state = CentinelState(num_clients=num_clients, tau=0.1, omega=0.7)
    
    # 4. Simulation Loop
    results = {
        "train_acc": [],
        "test_acc": [],
        "asr": [],
        "config": vars(args)
    }

    print(f"[{args.baseline.upper()}] Starting GPU {args.gpu} - Malicious: {args.malicious_fraction} - Attack: {args.attack_type} - {args.partition}")
    for r in range(args.rounds):
        if args.baseline == 'sfl':
            train_loss, train_acc = run_sfl_round(clients, server, local_epochs=1)
        elif args.baseline == 'sfl_gold':
            train_loss, train_acc = run_sfl_gold_round(clients, server, local_epochs=1)
        elif args.baseline == 'centinel':
            train_loss, train_acc, _, _ = run_sfl_centinel_round(
                clients, server, state, ref_loader, local_epochs=1, device=device
            )

        # Evaluation
        eval_client = clients[0].model
        test_acc = evaluate_accuracy(eval_client, server.model, test_loader, device)

        asr = 0.0
        if args.attack_type == 'backdoor':
            asr = evaluate_backdoor_asr(eval_client, server.model, test_loader, 
                                        ATTACK_KWARGS.get('backdoor_source_labels', []), 
                                        ATTACK_KWARGS.get('backdoor_target_label', 0), 
                                        ATTACK_KWARGS, device)
        elif args.attack_type == 'pair_flip':
            asr = evaluate_pair_flip_asr(eval_client, server.model, test_loader, 
                                         ATTACK_KWARGS.get('label_pairs_to_flip', []), 
                                         device)

        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
        results["asr"].append(asr)

        # Periodically flush
        if (r + 1) % 5 == 0:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=4)

    # Final save
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"[{args.baseline.upper()}] Finished. Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
