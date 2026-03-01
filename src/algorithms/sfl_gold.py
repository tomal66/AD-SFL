import copy
import torch

_BN_BUFFER_KEYS = ("running_mean", "running_var", "num_batches_tracked")

def _is_bn_buffer_key(k: str) -> bool:
    return any(s in k for s in _BN_BUFFER_KEYS)

def fedavg_state_dicts_weighted(state_dicts, weights, skip_bn_buffers=True):
    """
    Weighted FedAvg over state_dicts.
    - weights: typically local dataset sizes
    - skip_bn_buffers: do not average BN running stats/counters (important for non-IID)
    """
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Total weight must be > 0")

    out = copy.deepcopy(state_dicts[0])

    for k in out.keys():
        if skip_bn_buffers and _is_bn_buffer_key(k):
            continue

        v0 = state_dicts[0][k]
        if torch.is_floating_point(v0):
            out[k] = v0 * (weights[0] / total_w)
            for sd, w in zip(state_dicts[1:], weights[1:]):
                out[k] += sd[k] * (w / total_w)
        else:
            # ints / counters: keep from first (averaging ints is meaningless)
            out[k] = v0

    return out

def run_sfl_gold_round(clients, server, local_epochs=1):
    """
    SplitFed Gold (oracle): trains only on honest clients and aggregates only honest updates.
    Fixes:
      - Weighted FedAvg on client models (by local sample count)
      - Aggregate server models only over honest clients (and BN-safe if supported)
      - Avoid O(n) label lookup via labels_map
    """

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    honest_clients = [c for c in clients if not c.is_malicious]
    if not honest_clients:
        return 0.0, 0.0

    # Track which honest clients actually produced at least one batch (for safety)
    active_ids_in_round = set()

    for epoch in range(local_epochs):
        for client in honest_clients:
            client.reset_iterator()

        while True:
            smashed_list = []
            labels_map = {}
            active_clients = []

            # Client forward
            for client in honest_clients:
                smashed_data, labels = client.forward_pass(global_round=epoch)
                if smashed_data is None:
                    continue
                smashed_list.append((client.id, smashed_data))
                labels_map[client.id] = labels
                active_clients.append(client)
                active_ids_in_round.add(client.id)

            if not active_clients:
                break

            grad_to_clients_map = {}

            # Server train step per client
            for client_id, smashed_data in smashed_list:
                labels = labels_map[client_id]
                grad_to_client, loss, acc = server.train_step(
                    smashed_data, labels, client_id=client_id
                )

                grad_to_clients_map[client_id] = grad_to_client
                total_loss += float(loss)
                total_acc += float(acc)
                total_batches += 1

            # Client backward
            for client in active_clients:
                grad = grad_to_clients_map[client.id]
                client.backward_pass(grad)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0

    # ---------------------------------------------------------
    # FedAvg (client-side) over HONEST clients: WEIGHTED + BN-safe
    # ---------------------------------------------------------
    client_state_dicts = [c.get_weights() for c in honest_clients]
    client_nsamples = [len(c.dataloader.dataset) for c in honest_clients]

    global_client_weights = fedavg_state_dicts_weighted(
        client_state_dicts,
        client_nsamples,
        skip_bn_buffers=True,
    )

    # Broadcast aggregated client to ALL clients (including malicious)
    for client in clients:
        client.set_weights(global_client_weights)

    # ---------------------------------------------------------
    # FedAvg (server-side) over honest clients only
    # Prefer aggregating only those that actually trained in this round.
    # ---------------------------------------------------------
    honest_ids = sorted(list(active_ids_in_round)) if active_ids_in_round else [c.id for c in honest_clients]

    server_weights = [len(clients[i].dataloader.dataset) for i in honest_ids]
    server.aggregate_server_models(active_client_indices=honest_ids, weights=server_weights, skip_bn_buffers=True)

    return avg_loss, avg_acc
