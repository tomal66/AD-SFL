import torch
import torch.nn.functional as F
import copy


_BN_BUFFER_KEYS = ("running_mean", "running_var", "num_batches_tracked")

def _is_bn_buffer_key(k: str) -> bool:
    return any(s in k for s in _BN_BUFFER_KEYS)

def fedavg_state_dicts_weighted(state_dicts, weights, skip_bn_buffers=True):
    """
    Weighted FedAvg over a list of state_dicts.
    - state_dicts: list[OrderedDict[str, Tensor]]
    - weights: list[float] (e.g., local sample counts)
    - skip_bn_buffers: if True, do NOT aggregate BN running stats / counters
    Returns averaged_state_dict.
    """
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Total FedAvg weight must be > 0")

    out = copy.deepcopy(state_dicts[0])

    for k in out.keys():
        if skip_bn_buffers and _is_bn_buffer_key(k):
            # keep the first client's buffers (or you can keep global later)
            continue

        # initialize with weighted first
        v = state_dicts[0][k]
        if torch.is_floating_point(v):
            out[k] = v * (weights[0] / total_w)
            for sd, w in zip(state_dicts[1:], weights[1:]):
                out[k] += sd[k] * (w / total_w)
        else:
            # integer buffers (rare): take from first (or majority). Averaging ints is meaningless.
            out[k] = v

    return out


class CentinelState:
    """
    Tracks Centinel thresholds + subjective-logic state + global reference centroids.
    This version matches the reputation logic in your original TensorFlow code:
      u_i = 1 - Q_i
      b_i = (1-u_i) * alpha_i/(alpha_i+beta_i)   (if denom>0)
      gamma_i = b_i
    """
    def __init__(
        self,
        num_clients,
        tau=0.1,
        omega=0.7,
        Q_i=0.8,
        rho=0.4,
        eta=0.6,
        kappa=0.7,
        zeta=0.3,
    ):
        self.num_clients = num_clients
        self.tau = tau
        self.omega = omega

        # Subjective-logic hyperparameters (as in your TF code defaults)
        self.Q_i = Q_i
        self.rho = rho
        self.eta = eta
        self.kappa = kappa
        self.zeta = zeta

        # Each client's history of alpha_r / beta_r (recent interactions)
        self.alpha_history = {i: [] for i in range(num_clients)}
        self.beta_history = {i: [] for i in range(num_clients)}

        # Reputation history (gamma)
        self.reputation_history = {i: [] for i in range(num_clients)}

        # Global centroids per label: dict[int] -> torch.Tensor
        self.global_centroids = {}

    def get_accepted_clients(self):
        """
        Convenience helper: accepts by last-known reputation.
        NOTE: In run_sfl_centinel_round we compute acceptance fresh each round.
        """
        accepted = set()
        for i in range(self.num_clients):
            if not self.reputation_history[i]:
                accepted.add(i)
            elif self.reputation_history[i][-1] >= self.omega:
                accepted.add(i)
        return accepted


def compute_centroids(activations, labels):
    """
    Computes mean activation vector for each label in a batch.
    Args:
        activations: (B, D) torch.Tensor
        labels:      (B,) torch.Tensor (int labels)
    Returns:
        centroids: dict[label] -> (D,) torch.Tensor
        counts:    dict[label] -> int
    """
    centroids = {}
    counts = {}

    unique_labels = torch.unique(labels)
    for lbl in unique_labels:
        lbl_val = int(lbl.item())
        mask = (labels == lbl)
        acts_for_label = activations[mask]
        if acts_for_label.numel() == 0:
            continue
        centroids[lbl_val] = acts_for_label.mean(dim=0).detach()
        counts[lbl_val] = int(acts_for_label.size(0))

    return centroids, counts


def _accumulate_centroids_over_client(
    client,
    device="cpu",
    max_batches=None,
):
    """
    Robust centroid estimation: iterates over the client's whole local stream
    (or up to max_batches), accumulates per-label sums and counts.
    Uses client.get_next_batch() and client.reset_iterator().
    """
    client.model.eval()
    client.reset_iterator()

    sums = {}    # label -> sum vector
    counts = {}  # label -> count

    nb = 0
    while True:
        batch = client.get_next_batch()
        if batch is None:
            break
        data, target = batch
        if data is None:
            break

        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            acts = client.model(data)

        # accumulate per label
        for lbl in torch.unique(target):
            lbl_val = int(lbl.item())
            mask = (target == lbl)
            acts_lbl = acts[mask]
            if acts_lbl.numel() == 0:
                continue
            s = acts_lbl.sum(dim=0).detach()
            c = int(acts_lbl.size(0))

            if lbl_val in sums:
                sums[lbl_val] = sums[lbl_val] + s
                counts[lbl_val] += c
            else:
                sums[lbl_val] = s
                counts[lbl_val] = c

        nb += 1
        if max_batches is not None and nb >= max_batches:
            break

    # convert sums to means
    centroids = {}
    for lbl_val, s in sums.items():
        c = counts[lbl_val]
        if c > 0:
            centroids[lbl_val] = (s / float(c)).detach()

    # important: reset again so training can use full data
    client.reset_iterator()
    return centroids, counts


def _update_reputation_from_scores(scores, state: CentinelState):
    """
    Matches your TF code:
      alpha_p/beta_p are means over histories (or 0.0 if empty)
      alpha_i = kappa*rho*alpha_r + zeta*rho*alpha_p
      beta_i  = kappa*eta*beta_r  + zeta*eta*beta_p
      u_i = 1 - Q_i
      b_i = (1-u_i) * alpha_i/(alpha_i+beta_i)   if denom>0 else 0
      gamma_i = b_i
    """
    accepted_ids = set()

    for cid in range(state.num_clients):
        score = float(scores.get(cid, 0.0))

        # recent interaction
        if score > state.tau:
            alpha_r, beta_r = 0.0, 1.0  # negative
        else:
            alpha_r, beta_r = 1.0, 0.0  # positive

        # past interaction means (TF code uses 0.0 when empty)
        if state.alpha_history[cid]:
            alpha_p = sum(state.alpha_history[cid]) / len(state.alpha_history[cid])
        else:
            alpha_p = 0.0

        if state.beta_history[cid]:
            beta_p = sum(state.beta_history[cid]) / len(state.beta_history[cid])
        else:
            beta_p = 0.0

        alpha_i = state.kappa * state.rho * alpha_r + state.zeta * state.rho * alpha_p
        beta_i  = state.kappa * state.eta * beta_r  + state.zeta * state.eta * beta_p

        # log interaction history (store recent alpha_r/beta_r, as your TF did)
        state.alpha_history[cid].append(alpha_r)
        state.beta_history[cid].append(beta_r)

        u_i = 1.0 - state.Q_i
        if (alpha_i + beta_i) > 0:
            b_i = (1.0 - u_i) * (alpha_i / (alpha_i + beta_i))
        else:
            b_i = 0.0

        gamma_i = float(b_i)
        state.reputation_history[cid].append(gamma_i)

        if gamma_i >= state.omega:
            accepted_ids.add(cid)

    return accepted_ids


def run_sfl_centinel_round(
    clients,
    server,
    state: CentinelState,
    ref_loader,
    local_epochs=1,
    device="cpu",
    shift_max_batches=None,
):
    """
    Executes a single communication round of Centinel SplitFed defense.

    Fixes vs your draft:
      ✅ Reputation formula matches your TF code (no W=2 prior; uses Q_i uncertainty)
      ✅ Past alpha/beta defaults match TF (0.0, not (1.0,0.0))
      ✅ Centroid shift computed over whole client stream (or shift_max_batches)
      ✅ Device placement consistent (data/targets/centroids on same device)
      ✅ No O(n) next(...) label lookup; uses dict map
      ✅ Returns (avg_loss, avg_acc, scores_dict, accepted_ids)
    """
    server.model.to(device)
    for sm in server.models:
        sm.to(device)
    server.model.train()
    for sm in server.models:
        sm.train()

    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0

    # ---------------------------------------------------------
    # Step A: Measure each client's centroid shift (robust)
    # ---------------------------------------------------------
    shifts = {}
    # ensure global centroids are on device for distance computation
    global_centroids_dev = {lbl: c.to(device) for lbl, c in state.global_centroids.items()}

    for client in clients:
        # estimate client centroids over full local data (or capped)
        c_centroids, c_counts = _accumulate_centroids_over_client(
            client, device=device, max_batches=shift_max_batches
        )

        weighted_dist_sum = 0.0
        total_count = 0

        for lbl, count in c_counts.items():
            if lbl in global_centroids_dev and lbl in c_centroids:
                dist = torch.norm(c_centroids[lbl].to(device) - global_centroids_dev[lbl])
                weighted_dist_sum += float(dist.item()) * int(count)
                total_count += int(count)

        shifts[client.id] = (weighted_dist_sum / total_count) if total_count > 0 else 0.0

    # Normalize shifts into scores summing to 1 (as in TF code)
    total_shift = float(sum(shifts.values()))
    scores = {cid: (shifts[cid] / total_shift if total_shift > 1e-12 else 0.0) for cid in shifts}

    # ---------------------------------------------------------
    # Step B: Update reputation and select accepted clients
    # ---------------------------------------------------------
    accepted_ids = _update_reputation_from_scores(scores, state)
    accepted_clients = [c for c in clients if c.id in accepted_ids]

    if not accepted_clients:
        # No clients accepted: keep server as-is, keep centroids as-is, return.
        return 0.0, 0.0, scores, set()

    # ---------------------------------------------------------
    # Step C/D: Train only accepted clients
    # ---------------------------------------------------------
    for epoch in range(local_epochs):
        for client in accepted_clients:
            client.reset_iterator()

        while True:
            smashed_list = []
            labels_map = {}
            active_clients = []

            for client in accepted_clients:
                # forward_pass expected to yield (smashed_data, labels) or (None, None)
                smashed_data, labels = client.forward_pass(global_round=epoch)
                if smashed_data is None:
                    continue
                smashed_list.append((client.id, smashed_data.to(device)))
                labels_map[client.id] = labels.to(device)
                active_clients.append(client)

            if not active_clients:
                break

            grad_to_clients = {}

            for client_id, smashed_data in smashed_list:
                labels = labels_map[client_id]

                # server.train_step expected: (grad_to_client, loss, acc)
                grad_to_client, loss, acc = server.train_step(
                    smashed_data, labels, client_id=client_id
                )

                grad_to_clients[client_id] = grad_to_client
                total_loss += float(loss)
                total_acc += float(acc)
                total_batches += 1

            for client in active_clients:
                grad = grad_to_clients.get(client.id, None)
                if grad is not None:
                    client.backward_pass(grad)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0.0

    # ---------------------------------------------------------
    # Step E: FedAvg client-side weights over accepted clients (WEIGHTED), broadcast to all
    # ---------------------------------------------------------
    client_state_dicts = [c.get_weights() for c in accepted_clients]
    client_nsamples = [len(c.dataloader.dataset) for c in accepted_clients]  # robust

    global_client_weights = fedavg_state_dicts_weighted(
        client_state_dicts,
        client_nsamples,
        skip_bn_buffers=True,     # important for non-IID + BN
    )

    for client in clients:
        client.set_weights(global_client_weights)

    # Aggregate server side models for accepted clients
    accepted_idx = list(accepted_ids)
    accepted_ws = [len(clients[i].dataloader.dataset) for i in accepted_idx]

    server.aggregate_server_models(
        active_client_indices=accepted_idx,
        weights=accepted_ws,
        skip_bn_buffers=True
    )

    # ---------------------------------------------------------
    # Step F: Refresh reference centroids using updated client model (clients[0])
    # ---------------------------------------------------------
    ref_client = next(c for c in clients if c.id in accepted_ids)  # any accepted client
    ref_client.model.eval()

    new_sums = {}
    new_counts = {}

    with torch.no_grad():
        for data, target in ref_loader:
            data = data.to(device)
            target = target.to(device)

            acts = ref_client.model(data)
            for lbl in torch.unique(target):
                lbl_val = int(lbl.item())
                mask = (target == lbl)
                acts_lbl = acts[mask]
                if acts_lbl.numel() == 0:
                    continue

                s = acts_lbl.sum(dim=0).detach()
                c = int(acts_lbl.size(0))

                if lbl_val in new_sums:
                    new_sums[lbl_val] = new_sums[lbl_val] + s
                    new_counts[lbl_val] += c
                else:
                    new_sums[lbl_val] = s
                    new_counts[lbl_val] = c

    new_centroids = {}
    for lbl_val, s in new_sums.items():
        c = new_counts[lbl_val]
        if c > 0:
            new_centroids[lbl_val] = (s / float(c)).detach().cpu()

    # store global centroids on CPU (stable), moved to device when needed
    state.global_centroids = new_centroids

    return avg_loss, avg_acc, scores, accepted_ids