"""
ad_sfl.py
=========
Anomaly-Detection Split Federated Learning (AD-SFL) algorithm module.

Ported from `kde_reloaded copy.ipynb` and adapted to follow the repo's
existing architecture (centinel.py, sfl.py, sfl_gold.py).

Key components
--------------
* AdSflConfig           — all hyperparameters as a dataclass
* AdSflState            — round-to-round persistent state
* RLThresholdAgentOmega — DDPG agent that learns the acceptance threshold ω
* KDE / binning         — KL-divergence anomaly scoring (caller-configurable)
* Fisher tau            — adaptive anomaly threshold via Fisher's ratio
* detect_malicious_clients — subjective-logic reputation & client filtering
* run_ad_sfl_round      — entry-point: one full communication round
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Shared FedAvg helper (mirrors centinel.py)
# ---------------------------------------------------------------------------

_BN_BUFFER_KEYS = ("running_mean", "running_var", "num_batches_tracked")


def _is_bn_buffer(k: str) -> bool:
    return any(s in k for s in _BN_BUFFER_KEYS)


def fedavg_state_dicts_weighted(state_dicts, weights, skip_bn_buffers: bool = True):
    """Weighted FedAvg over a list of state_dicts."""
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Total FedAvg weight must be > 0")

    out = copy.deepcopy(state_dicts[0])
    for k in out.keys():
        if skip_bn_buffers and _is_bn_buffer(k):
            continue
        v = state_dicts[0][k]
        if torch.is_floating_point(v):
            out[k] = v * (weights[0] / total_w)
            for sd, w in zip(state_dicts[1:], weights[1:]):
                out[k] = out[k] + sd[k] * (w / total_w)
        else:
            out[k] = v
    return out


# ===========================================================================
# 1.  AdSflConfig
# ===========================================================================

@dataclass
class AdSflConfig:
    """All AD-SFL hyperparameters in one place.

    Pass an instance of this to AdSflState and run_ad_sfl_round so that
    callers can swap KDE vs binning, tune Fisher guards, etc. without
    touching internal logic.
    """

    # -------------------------------------------------------------------
    # Anomaly detection  (KDE vs binning is the caller's choice)
    # -------------------------------------------------------------------
    kl_estimator: str = "binning"       # "kde" or "binning"
    kl_eps: float = 1e-8
    kl_aggregation: str = "mean"        # "mean", "sum", or "max"

    # KDE estimator (used when kl_estimator == "kde")
    kde_bandwidth_mode: str = "silverman"   # "silverman" or "fixed"
    kde_bandwidth: float = 0.2              # used only when mode == "fixed"
    kde_max_ref_samples: int = 5000
    kde_max_client_samples: int = 5000

    # Histogram-binning estimator (used when kl_estimator == "binning")
    hist_num_bins: int = 60
    hist_range_mode: str = "percentile"    # "percentile" or "minmax"
    hist_range_pct: Tuple[float, float] = (1.0, 99.0)
    hist_max_ref_samples: int = 5000
    hist_max_client_samples: int = 5000

    # -------------------------------------------------------------------
    # Tau thresholding  (Fisher-only)
    # -------------------------------------------------------------------
    use_static_tau: bool = False
    static_tau_value: float = 0.1
    use_normalized_shifts: bool = True      # normalize raw scores before Fisher

    # Fisher tau safety knobs
    use_fisher_guard: bool = True
    fisher_min_ratio: float = 5.0
    fisher_min_sep_std: float = 2.0
    fisher_max_unimodal_cv: float = 0.05
    fisher_fallback_mode: str = "percentile"   # "mad", "percentile", "max"
    fisher_fallback_percentile: float = 99.0
    fisher_mad_k: float = 2.0
    fisher_n_candidates: int = 100

    # -------------------------------------------------------------------
    # Subjective-logic reputation
    # -------------------------------------------------------------------
    Q_i: float = 0.8
    rho: float = 0.4
    eta: float = 0.6
    kappa: float = 0.7
    zeta: float = 0.3
    min_accept_k: int = 1              # guarantee at least k clients accepted

    # -------------------------------------------------------------------
    # DDPG omega agent  (RLThresholdAgentOmega)
    # -------------------------------------------------------------------
    rl_hidden: int = 128
    rl_gamma: float = 0.995
    rl_tau_soft: float = 0.005
    rl_batch_size: int = 32
    rl_noise_scale: float = 0.05
    rl_memory_size: int = 2000
    rl_actor_lr: float = 1e-4
    rl_critic_lr: float = 1e-3
    rl_omega_min: float = 0.3
    rl_omega_max: float = 0.95

    # -------------------------------------------------------------------
    # Reference sampling
    # -------------------------------------------------------------------
    num_samples_per_label: int = 10
    num_classes: int = 10


# ===========================================================================
# 2.  RLThresholdAgentOmega  (DDPG)
# ===========================================================================

class RLThresholdAgentOmega:
    """
    DDPG agent that adaptively learns the acceptance threshold ω each round.

    State  : np.ndarray of shape (3 * num_clients,)
             Per client: [normalised_shift, normalised_loss, reputation]
    Action : scalar ω ∈ [cfg.rl_omega_min, cfg.rl_omega_max]
    Reward : macro-F1 of the model after the round (higher is better)

    Architecture
    ------------
    Actor    : Linear(3N→H)→ReLU→Linear(H→H//2)→ReLU→Linear(H//2→1)→Sigmoid
    Critic   : Linear(3N+1→H)→ReLU→Linear(H→H//2)→ReLU→Linear(H//2→1)
    Both have matching target networks, soft-updated each training step.
    """

    def __init__(self, state_dim: int, cfg: AdSflConfig, device: str = "cpu"):
        self.state_dim = state_dim
        self.cfg = cfg
        self.device = device

        H = cfg.rl_hidden

        def build_actor():
            return nn.Sequential(
                nn.Linear(state_dim, H), nn.ReLU(),
                nn.Linear(H, H // 2), nn.ReLU(),
                nn.Linear(H // 2, 1), nn.Sigmoid(),
            ).to(device)

        def build_critic():
            return nn.Sequential(
                nn.Linear(state_dim + 1, H), nn.ReLU(),
                nn.Linear(H, H // 2), nn.ReLU(),
                nn.Linear(H // 2, 1),
            ).to(device)

        self.actor = build_actor()
        self.actor_target = build_actor()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = build_critic()
        self.critic_target = build_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.rl_actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.rl_critic_lr)

        self.memory: List[Tuple] = []
        self._step_count = 0

    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, noise: bool = True) -> float:
        """Return ω clipped to [rl_omega_min, rl_omega_max]."""
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw = self.actor(x).cpu().numpy()[0, 0]
        if noise:
            raw = raw + self.cfg.rl_noise_scale * np.random.normal()
        return float(np.clip(raw, self.cfg.rl_omega_min, self.cfg.rl_omega_max))

    # ------------------------------------------------------------------
    def store(self, s: np.ndarray, a: float, r: float, s_next: np.ndarray):
        """Push a transition into the replay buffer."""
        self.memory.append((s, np.array([a], dtype=np.float32), r, s_next))
        if len(self.memory) > self.cfg.rl_memory_size:
            self.memory.pop(0)

    # ------------------------------------------------------------------
    def train(self):
        """One DDPG update step (skips silently if buffer is too small)."""
        if len(self.memory) < self.cfg.rl_batch_size:
            return

        idxs = np.random.choice(len(self.memory), self.cfg.rl_batch_size, replace=False)
        batch = [self.memory[i] for i in idxs]

        states      = torch.tensor(np.stack([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions     = torch.tensor(np.stack([t[1] for t in batch]), dtype=torch.float32, device=self.device)  # (B,1)
        rewards     = torch.tensor([[t[2]] for t in batch], dtype=torch.float32, device=self.device)          # (B,1)
        next_states = torch.tensor(np.stack([t[3] for t in batch]), dtype=torch.float32, device=self.device)

        # ---- Critic update ----
        with torch.no_grad():
            next_actions   = self.actor_target(next_states)              # (B,1)
            q_target_next  = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            td_target      = rewards + self.cfg.rl_gamma * q_target_next

        q_pred = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(q_pred, td_target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # ---- Actor update (every 2nd critic step) ----
        self._step_count += 1
        if self._step_count % 2 == 0:
            pred_actions = self.actor(states)
            actor_loss = -self.critic(torch.cat([states, pred_actions], dim=1)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        # ---- Soft target update ----
        tau = self.cfg.rl_tau_soft
        for tgt, src in zip(self.actor_target.parameters(), self.actor.parameters()):
            tgt.data.mul_(1.0 - tau).add_(src.data * tau)
        for tgt, src in zip(self.critic_target.parameters(), self.critic.parameters()):
            tgt.data.mul_(1.0 - tau).add_(src.data * tau)


# ===========================================================================
# 3.  AdSflState
# ===========================================================================

@dataclass
class AdSflState:
    """
    All round-to-round mutable state for AD-SFL.

    Create one instance before the training loop and pass it to every call
    of run_ad_sfl_round.  The function updates it in-place.
    """
    num_clients: int
    cfg: AdSflConfig
    device: str = "cpu"

    # Subjective-logic interaction histories  {i: {alpha_p: [], beta_p: []}}
    interactions: List[Dict] = field(default_factory=list)

    # Per-round tracking
    client_reputation_over_rounds: List[List[float]] = field(default_factory=list)
    centroid_shifts_history: List[List[float]] = field(default_factory=list)
    tau_history: List[float] = field(default_factory=list)
    omega_history: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    asr_history: List[float] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    metrics_per_round: List[Dict] = field(default_factory=list)
    accepted_clients_last: List[int] = field(default_factory=list)

    # Reference data (refreshed every round)
    ref_data: Optional[Dict] = field(default=None)

    # DDPG agent (initialised in __post_init__)
    rl_agent: Optional[RLThresholdAgentOmega] = field(default=None, init=False)

    # Internal bookkeeping
    _tau_prev: Optional[float] = field(default=None, init=False)
    _prev_detection_rate: Optional[float] = field(default=None, init=False)
    _prev_val_acc: float = field(default=0.0, init=False)

    def __post_init__(self):
        self.interactions = [{"alpha_p": [], "beta_p": []} for _ in range(self.num_clients)]
        self.client_reputation_over_rounds = [[] for _ in range(self.num_clients)]
        state_dim = 3 * self.num_clients
        self.rl_agent = RLThresholdAgentOmega(state_dim, self.cfg, self.device)

    def get_last_reputations(self) -> List[float]:
        return [
            self.client_reputation_over_rounds[i][-1]
            if self.client_reputation_over_rounds[i] else 0.0
            for i in range(self.num_clients)
        ]


# ===========================================================================
# 4.  Reference data helpers
# ===========================================================================

@torch.no_grad()
def sample_reference_data_per_label(
    dataset,
    num_samples_per_label: int,
    num_classes: int = 10,
    device: str = "cpu",
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Sample up to `num_samples_per_label` images per class from `dataset`.

    Returns
    -------
    {label: (Xs [K,C,H,W], Ys [K])}  — tensors on `device`
    """
    buckets: Dict[int, list] = {i: [] for i in range(num_classes)}
    for x, y in DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0):
        for xi, yi in zip(x, y):
            lbl = int(yi)
            if lbl in buckets and len(buckets[lbl]) < num_samples_per_label:
                buckets[lbl].append(xi)
        if all(len(buckets[i]) >= num_samples_per_label for i in range(num_classes)):
            break

    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for lab, imgs in buckets.items():
        if imgs:
            xs = torch.stack(imgs, dim=0).to(device)
            ys = torch.full((len(imgs),), lab, dtype=torch.long, device=device)
            out[lab] = (xs, ys)
    return out


@torch.no_grad()
def compute_ref_hist_data(
    client_model: nn.Module,
    ref_samples: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    cfg: AdSflConfig,
    device: str = "cpu",
) -> Dict[int, Dict]:
    """
    Compute histogram reference distributions per label.

    Returns
    -------
    {label: {"bin_edges": np.ndarray[B+1], "p_ref": np.ndarray[B]}}
    """
    client_model.eval()
    ref_hist: Dict[int, Dict] = {}
    eps = cfg.kl_eps

    for lab, (xs, _) in ref_samples.items():
        z = client_model(xs.to(device))
        vals = z.detach().cpu().numpy().reshape(-1).astype(np.float64)

        if vals.size == 0:
            continue
        if cfg.hist_max_ref_samples and vals.size > cfg.hist_max_ref_samples:
            idx = np.random.choice(vals.size, cfg.hist_max_ref_samples, replace=False)
            vals = vals[idx]

        if cfg.hist_range_mode == "percentile":
            lo, hi = np.percentile(vals, list(cfg.hist_range_pct))
        else:
            lo, hi = float(vals.min()), float(vals.max())

        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo, hi = float(vals.min()), float(vals.max() + 1e-6)

        bin_edges = np.linspace(lo, hi, cfg.hist_num_bins + 1)
        counts, _ = np.histogram(vals, bins=bin_edges)
        p_ref = counts.astype(np.float64) + eps
        p_ref /= p_ref.sum()

        ref_hist[int(lab)] = {"bin_edges": bin_edges, "p_ref": p_ref}

    return ref_hist


@torch.no_grad()
def compute_ref_kde_data(
    client_model: nn.Module,
    ref_samples: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    cfg: AdSflConfig,
    device: str = "cpu",
) -> Dict[int, Dict]:
    """
    Compute KDE reference distributions (values + bandwidth) per label.

    Returns
    -------
    {label: {"values": np.ndarray, "bandwidth": float}}
    """
    client_model.eval()
    ref_kde: Dict[int, Dict] = {}

    for lab, (xs, _) in ref_samples.items():
        z = client_model(xs.to(device))
        vals = z.detach().cpu().numpy().reshape(-1).astype(np.float64)

        if vals.size == 0:
            continue
        if cfg.kde_max_ref_samples and vals.size > cfg.kde_max_ref_samples:
            idx = np.random.choice(vals.size, cfg.kde_max_ref_samples, replace=False)
            vals = vals[idx]

        if cfg.kde_bandwidth_mode == "fixed":
            h = float(cfg.kde_bandwidth)
        else:
            std = float(np.std(vals))
            q75, q25 = np.percentile(vals, [75, 25])
            iqr = float(q75 - q25)
            if std <= 0 and iqr <= 0:
                sigma = 1.0
            elif std > 0 and iqr > 0:
                sigma = min(std, iqr / 1.34)
            else:
                sigma = max(std, iqr / 1.34)
            n = max(1, vals.size)
            h = 0.9 * sigma * (n ** (-0.2)) if n > 1 else 1.0

        ref_kde[int(lab)] = {"values": vals, "bandwidth": max(h, 1e-6)}

    return ref_kde


# ===========================================================================
# 5.  KDE helpers
# ===========================================================================

def gaussian_kde_logpdf_loo(samples: np.ndarray, h: float, eps: float = 1e-8) -> np.ndarray:
    """
    Leave-one-out Gaussian KDE log-density at each of `samples`.

    Numerically stable via per-row log-sum-exp.
    """
    samples = np.asarray(samples, dtype=np.float64)
    N = samples.size
    if N < 2:
        return np.full((N,), np.log(eps), dtype=np.float64)

    h = float(max(h, 1e-6))
    diff = (samples[:, None] - samples[None, :]) / h        # (N, N)
    exponent = -0.5 * diff * diff

    max_exp = np.max(exponent, axis=1, keepdims=True)
    exp_shifted = np.exp(exponent - max_exp)

    # subtract self-kernel (always exp(0)=1 after "shifted" => we need raw form)
    # The self term in exp_shifted is exp(0) = 1, so subtract 1.
    row_sum = exp_shifted.sum(axis=1) - 1.0
    row_sum = np.maximum(row_sum, 1e-300)

    log_sum = max_exp.squeeze(-1) + np.log(row_sum)
    log_norm = np.log(N - 1) + np.log(h) + 0.5 * np.log(2.0 * np.pi)
    log_density = log_sum - log_norm
    return np.maximum(log_density, np.log(eps))


def gaussian_kde_logpdf(
    eval_pts: np.ndarray,
    ref_vals: np.ndarray,
    h: float,
    eps: float = 1e-8,
) -> np.ndarray:
    """Standard Gaussian KDE log-density of `ref_vals` evaluated at `eval_pts`."""
    eval_pts = np.asarray(eval_pts, dtype=np.float64)
    ref_vals = np.asarray(ref_vals, dtype=np.float64)
    N = ref_vals.size
    h = float(max(h, 1e-6))

    diff = (eval_pts[:, None] - ref_vals[None, :]) / h      # (M, N)
    exponent = -0.5 * diff * diff

    max_exp = np.max(exponent, axis=1, keepdims=True)
    log_sum = max_exp.squeeze(-1) + np.log(np.exp(exponent - max_exp).sum(axis=1))
    log_norm = np.log(N) + np.log(h) + 0.5 * np.log(2.0 * np.pi)
    return np.maximum(log_sum - log_norm, np.log(eps))


# ===========================================================================
# 6.  Anomaly score (KL-divergence)
# ===========================================================================

@torch.no_grad()
def compute_client_kl_divergence_binning(
    client_model: nn.Module,
    loader: DataLoader,
    ref_hist_data: Dict[int, Dict],
    cfg: AdSflConfig,
    device: str = "cpu",
) -> float:
    """Histogram-binning KL(p_client || p_ref), aggregated across labels."""
    client_model.eval()
    label_vals: Dict[int, list] = {lab: [] for lab in ref_hist_data}
    label_img_counts: Dict[int, int] = {lab: 0 for lab in ref_hist_data}

    for x, y in loader:
        z = client_model(x.to(device)).detach().cpu()
        yc = y.cpu()
        for lab in label_vals:
            mask = (yc == lab)
            if mask.any():
                label_img_counts[lab] += int(mask.sum().item())
                label_vals[lab].append(z[mask].reshape(-1))

    per_kl, per_w = [], []
    for lab, chunks in label_vals.items():
        if not chunks or label_img_counts.get(lab, 0) < 2:
            continue
        vals = torch.cat(chunks).numpy().astype(np.float64)
        if vals.size == 0:
            continue
        if cfg.hist_max_client_samples and vals.size > cfg.hist_max_client_samples:
            idx = np.random.choice(vals.size, cfg.hist_max_client_samples, replace=False)
            vals = vals[idx]

        ref = ref_hist_data.get(lab)
        if ref is None:
            continue
        bin_edges = ref["bin_edges"]
        p_ref = ref["p_ref"]

        counts, _ = np.histogram(vals, bins=bin_edges)
        p_cli = counts.astype(np.float64) + cfg.kl_eps
        p_cli /= p_cli.sum() + 1e-12

        kl = float(np.sum(p_cli * (np.log(p_cli) - np.log(p_ref))))
        per_kl.append(kl)
        per_w.append(float(label_img_counts[lab]))

    if not per_kl:
        return 0.0

    kls = np.asarray(per_kl, dtype=np.float64)
    wts = np.asarray(per_w, dtype=np.float64)
    wts /= wts.sum() + 1e-12

    if cfg.kl_aggregation == "max":
        return float(kls.max())
    return float(np.sum(kls * wts))   # "mean" or "sum" both produce weighted mean


@torch.no_grad()
def compute_client_kl_divergence(
    client_model: nn.Module,
    loader: DataLoader,
    ref_kde_data: Dict[int, Dict],
    cfg: AdSflConfig,
    device: str = "cpu",
) -> float:
    """
    KDE-based KL-divergence anomaly score  KL(p_client || p_ref).

    Uses leave-one-out KDE for the client density to reduce self-density bias.
    """
    client_model.eval()
    label_vals: Dict[int, list] = {lab: [] for lab in ref_kde_data}
    label_img_counts: Dict[int, int] = {lab: 0 for lab in ref_kde_data}

    for x, y in loader:
        z = client_model(x.to(device)).detach().cpu()
        yc = y.cpu()
        for lab in label_vals:
            mask = (yc == lab)
            if mask.any():
                label_img_counts[lab] += int(mask.sum().item())
                label_vals[lab].append(z[mask].reshape(-1))

    per_kl, per_w = [], []
    for lab, chunks in label_vals.items():
        if not chunks or label_img_counts.get(lab, 0) < 2:
            continue
        vals = torch.cat(chunks).numpy().astype(np.float64)
        if vals.size < 2:
            continue
        if cfg.kde_max_client_samples and vals.size > cfg.kde_max_client_samples:
            idx = np.random.choice(vals.size, cfg.kde_max_client_samples, replace=False)
            vals = vals[idx]

        ref = ref_kde_data.get(lab)
        if ref is None or ref["values"].size < 2:
            continue

        h = ref["bandwidth"]
        log_p_i   = gaussian_kde_logpdf_loo(vals, h, eps=cfg.kl_eps)
        log_p_ref = gaussian_kde_logpdf(vals, ref["values"], h, eps=cfg.kl_eps)

        kl = float(np.mean(log_p_i - log_p_ref))
        per_kl.append(kl)
        per_w.append(float(label_img_counts[lab]))

    if not per_kl:
        return 0.0

    kls = np.asarray(per_kl, dtype=np.float64)
    wts = np.asarray(per_w, dtype=np.float64)
    wts /= wts.sum() + 1e-12

    if cfg.kl_aggregation == "max":
        return float(kls.max())
    return float(np.sum(kls * wts))


@torch.no_grad()
def compute_client_anomaly_score(
    client_model: nn.Module,
    loader: DataLoader,
    ref_data: Dict[int, Dict],
    cfg: AdSflConfig,
    device: str = "cpu",
) -> float:
    """
    Dispatcher: returns the KL-divergence anomaly score for one client.
    Routes to KDE or histogram-binning depending on cfg.kl_estimator.
    """
    if cfg.kl_estimator == "kde":
        return compute_client_kl_divergence(client_model, loader, ref_data, cfg, device)
    elif cfg.kl_estimator == "binning":
        return compute_client_kl_divergence_binning(client_model, loader, ref_data, cfg, device)
    else:
        raise ValueError(f"Unknown kl_estimator: '{cfg.kl_estimator}'. Use 'kde' or 'binning'.")


# ===========================================================================
# 7.  Fisher tau thresholding
# ===========================================================================

def _fisher_fallback_threshold(shifts: np.ndarray, cfg: AdSflConfig) -> float:
    """Conservative tau when Fisher grid search is not trustworthy."""
    shifts = np.asarray(shifts, dtype=float)
    mode = cfg.fisher_fallback_mode
    if mode == "mad":
        med = float(np.median(shifts))
        mad = float(np.median(np.abs(shifts - med))) + 1e-12
        return med + cfg.fisher_mad_k * mad
    elif mode == "percentile":
        return float(np.percentile(shifts, cfg.fisher_fallback_percentile))
    elif mode == "max":
        return float(shifts.max() + 1e-6)
    else:
        return float(np.median(shifts))


def optimal_fisher_threshold(shifts: np.ndarray, cfg: AdSflConfig) -> float:
    """
    Fisher's ratio-maximising threshold with unimodal guard.

    Falls back to a conservative threshold (MAD / percentile / max) when:
    - The distribution is too tight (coefficient of variation < cfg.fisher_max_unimodal_cv)
    - The best Fisher split is weak (ratio < cfg.fisher_min_ratio)
    - The cluster separation is small (< cfg.fisher_min_sep_std pooled std)
    """
    shifts = np.asarray(shifts, dtype=float)
    if shifts.size < 4:
        return float(np.median(shifts))

    # --- Global unimodal guard ---
    mean_all = float(shifts.mean())
    std_all  = float(shifts.std() + 1e-8)
    cv_all   = std_all / (abs(mean_all) + 1e-8)
    if cfg.use_fisher_guard and cv_all < cfg.fisher_max_unimodal_cv:
        return _fisher_fallback_threshold(shifts, cfg)

    # --- Grid search for best Fisher split ---
    candidates = np.linspace(shifts.min(), shifts.max(), cfg.fisher_n_candidates)
    best_ratio = -1.0
    best_tau   = float(np.median(shifts))
    best_sep   = 0.0

    for tau_cand in candidates:
        below = shifts[shifts <= tau_cand]
        above = shifts[shifts > tau_cand]
        if below.size < 2 or above.size < 2:
            continue

        mu1, s1 = below.mean(), below.std() + 1e-8
        mu2, s2 = above.mean(), above.std() + 1e-8
        denom  = s1 ** 2 + s2 ** 2 + 1e-8
        fisher = (mu2 - mu1) ** 2 / denom

        if fisher > best_ratio:
            best_ratio = fisher
            best_tau   = float(tau_cand)
            best_sep   = abs(mu2 - mu1) / (math.sqrt(denom) + 1e-8)

    # --- Weak split guard ---
    if cfg.use_fisher_guard:
        if best_ratio < cfg.fisher_min_ratio or best_sep < cfg.fisher_min_sep_std:
            return _fisher_fallback_threshold(shifts, cfg)

    return float(best_tau)


def compute_tau(
    centroid_shifts: List[float],
    cfg: AdSflConfig,
    prev_tau: Optional[float] = None,
) -> float:
    """
    Return tau for this round.

    If cfg.use_static_tau  → return cfg.static_tau_value.
    Otherwise              → Fisher threshold over centroid_shifts.
    The prev_tau argument is kept for API compatibility (Fisher is stateless).
    """
    if cfg.use_static_tau:
        return float(cfg.static_tau_value)
    arr = np.asarray(centroid_shifts, dtype=float)
    if arr.size == 0:
        return float(prev_tau) if prev_tau is not None else float(cfg.static_tau_value)
    return optimal_fisher_threshold(arr, cfg)


# ===========================================================================
# 8.  Detection & subjective-logic reputation
# ===========================================================================

def detect_malicious_clients(
    centroid_shifts: List[float],
    interactions: List[Dict],
    tau: float,
    omega: float,
    cfg: AdSflConfig,
    malicious_set: Optional[set] = None,
) -> Tuple[List[int], List[Dict], Dict, List[float]]:
    """
    Compute per-client reputation scores and return accepted clients.

    Algorithm (from the notebook)
    ------------------------------
    1. Normalise anomaly scores to sum to 1.
    2. For each client i:
       - If score_i > tau  →  alpha_r=0, beta_r=1  (negative interaction)
       - Else              →  alpha_r=1, beta_r=0  (positive interaction)
       - Combine with past history using kappa / zeta / rho / eta weights
       - gamma_i = b_i = (1 - Q_i) * alpha_i / (alpha_i + beta_i)
       - Accept if gamma_i >= omega
    3. If no one is accepted, forcibly accept the min_accept_k clients with
       the lowest anomaly scores.

    Parameters
    ----------
    malicious_set : optional ground-truth set for TP/TN/FP/FN metrics

    Returns
    -------
    (accepted_ids, updated_interactions, metrics_dict, reputation_scores)
    """
    ssum = max(1e-12, float(sum(centroid_shifts)))
    scores = [s / ssum for s in centroid_shifts]

    accepted: List[int] = []
    reputation_scores: List[float] = []
    tp = tn = fp = fn = 0

    for i, score in enumerate(scores):
        alpha_p = float(np.mean(interactions[i]["alpha_p"])) if interactions[i]["alpha_p"] else 0.0
        beta_p  = float(np.mean(interactions[i]["beta_p"]))  if interactions[i]["beta_p"]  else 0.0

        if score > tau:
            alpha_r, beta_r = 0.0, 1.0   # negative: score above threshold
        else:
            alpha_r, beta_r = 1.0, 0.0   # positive: score below threshold

        alpha_i = cfg.kappa * cfg.rho * alpha_r + cfg.zeta * cfg.rho * alpha_p
        beta_i  = cfg.kappa * cfg.eta  * beta_r  + cfg.zeta * cfg.eta  * beta_p

        u_i = 1.0 - cfg.Q_i
        denom = alpha_i + beta_i
        b_i = (1.0 - u_i) * (alpha_i / denom) if denom > 0 else 0.0

        gamma_i = float(b_i)
        reputation_scores.append(gamma_i)

        interactions[i]["alpha_p"].append(alpha_r)
        interactions[i]["beta_p"].append(beta_r)

        if gamma_i >= omega:
            accepted.append(i)

        # TP / TN / FP / FN
        if malicious_set is not None:
            if i in malicious_set:
                if gamma_i < omega: tp += 1
                else:               fn += 1
            else:
                if gamma_i >= omega: tn += 1
                else:                fp += 1

    # Guarantee at least `min_accept_k` clients are accepted
    if len(accepted) == 0 and cfg.min_accept_k > 0:
        ranked = np.argsort(centroid_shifts)[: cfg.min_accept_k].tolist()
        accepted = ranked

    metrics = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return accepted, interactions, metrics, reputation_scores


# ===========================================================================
# 9.  RL reward / state helpers
# ===========================================================================

def build_rl_state_per_client(
    norm_shifts: List[float],
    client_losses: Dict[int, float],
    rep_scores: List[float],
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build the DDPG state vector of shape (3 * N,).

    Per client: [normalised_shift, normalised_loss, reputation]
    - norm_shifts already sum to 1 (passed from run_ad_sfl_round)
    - client_losses are normalised by the mean across all clients
    """
    N = len(norm_shifts)
    losses = np.array([float(client_losses.get(i, 0.0)) for i in range(N)], dtype=np.float32)
    mean_loss = float(losses.mean()) + eps
    losses_norm = losses / mean_loss

    reps = np.array([float(rep_scores[i]) for i in range(N)], dtype=np.float32)

    state = []
    for i in range(N):
        state.extend([float(norm_shifts[i]), float(losses_norm[i]), float(reps[i])])
    return np.array(state, dtype=np.float32)


def compute_rl_reward(f1_macro: float) -> float:
    """Reward = macro-F1 of the updated model (higher is better)."""
    return float(f1_macro)


# ===========================================================================
# 10.  Client loss evaluation helper
# ===========================================================================

@torch.no_grad()
def compute_client_losses(
    client_models: List[nn.Module],
    server_model: nn.Module,
    client_loaders: List[DataLoader],
    device: str = "cpu",
) -> Dict[int, float]:
    """Evaluate per-client cross-entropy loss on each client's local data."""
    criterion = nn.CrossEntropyLoss()
    losses: Dict[int, float] = {}

    for i, (model, loader) in enumerate(zip(client_models, client_loaders)):
        model.eval()
        server_model.eval()
        total = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z      = model(x)
            logits = server_model(z)
            loss   = criterion(logits, y)
            total += loss.item() * x.size(0)
            count += x.size(0)
        losses[i] = total / max(count, 1)

    return losses


# ===========================================================================
# 11.  Evaluation helpers (macro-F1 + accuracy)
# ===========================================================================

@torch.no_grad()
def evaluate_accuracy(client_model, server_model, loader, device="cpu") -> float:
    client_model.eval(); server_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = server_model(client_model(x)).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_f1_macro(
    client_model, server_model, loader,
    num_classes: int = 10, device: str = "cpu", eps: float = 1e-12
) -> float:
    client_model.eval(); server_model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = server_model(client_model(x)).argmax(dim=1)
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[int(t), int(p)] += 1

    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    prec   = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1     = 2 * prec * recall / (prec + recall + eps)
    return float(f1.mean().item())


# ===========================================================================
# 12.  run_ad_sfl_round  (main entry-point)
# ===========================================================================

def run_ad_sfl_round(
    clients,
    server,
    state: AdSflState,
    ref_samples: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    test_loader: DataLoader,
    malicious_indices: List[int],
    rnd: int,
    total_rounds: int,
    cfg: AdSflConfig,
    device: str = "cpu",
    local_epochs: int = 1,
) -> Dict:
    """
    Execute one communication round of AD-SFL.

    Parameters
    ----------
    clients          : list of SplitFedClient — loaders already contain any
                       poisoning applied externally (no attack logic here).
    server           : SplitFedServer
    state            : AdSflState (mutated in-place across rounds)
    ref_samples      : output of sample_reference_data_per_label()
    test_loader      : DataLoader for overall accuracy evaluation
    malicious_indices: ground-truth malicious client indices (for TP/FP stats)
    rnd              : 0-indexed current round number
    total_rounds     : total rounds (used for Fisher normalisation)
    cfg              : AdSflConfig
    device           : torch device string
    local_epochs     : SFL local epochs per round

    Returns
    -------
    dict with keys:
        test_acc, f1_macro, tau, omega, latency,
        accepted_clients, metrics, reputation_scores,
        centroid_shifts, asr
    """
    t0 = time.time()
    num_clients = len(clients)
    client_models  = [c.model for c in clients]
    client_loaders = [c.dataloader for c in clients]

    # ------------------------------------------------------------------
    # A  Compute ref_data (refreshed at the start of each round)
    # ------------------------------------------------------------------
    if cfg.kl_estimator == "kde":
        state.ref_data = compute_ref_kde_data(client_models[0], ref_samples, cfg, device)
    else:
        state.ref_data = compute_ref_hist_data(client_models[0], ref_samples, cfg, device)

    # ------------------------------------------------------------------
    # B  Anomaly scores per client
    # ------------------------------------------------------------------
    raw_shifts = [
        compute_client_anomaly_score(client_models[i], client_loaders[i], state.ref_data, cfg, device)
        for i in range(num_clients)
    ]

    sum_shift  = max(1e-12, float(sum(raw_shifts)))
    norm_shifts = [s / sum_shift for s in raw_shifts]
    shifts_for_detection = norm_shifts if cfg.use_normalized_shifts else raw_shifts

    state.centroid_shifts_history.append(shifts_for_detection[:])

    # ------------------------------------------------------------------
    # C  Fisher tau
    # ------------------------------------------------------------------
    tau = compute_tau(shifts_for_detection, cfg, prev_tau=state._tau_prev)
    state.tau_history.append(tau)
    state._tau_prev = tau

    # ------------------------------------------------------------------
    # D  Per-client losses  +  RL state  +  DDPG omega
    # ------------------------------------------------------------------
    client_losses = compute_client_losses(client_models, server.model, client_loaders, device)

    rep_scores = state.get_last_reputations()
    rl_state   = build_rl_state_per_client(norm_shifts, client_losses, rep_scores)
    omega      = state.rl_agent.select_action(rl_state, noise=True)
    state.omega_history.append(omega)

    # ------------------------------------------------------------------
    # E  Detection & reputation
    # ------------------------------------------------------------------
    malicious_set = set(malicious_indices)
    accepted_ids, state.interactions, round_metrics, reputation_scores = detect_malicious_clients(
        shifts_for_detection,
        state.interactions,
        tau=tau,
        omega=omega,
        cfg=cfg,
        malicious_set=malicious_set,
    )

    state.metrics_per_round.append(round_metrics)
    state.accepted_clients_last = accepted_ids

    for i, gamma in enumerate(reputation_scores):
        state.client_reputation_over_rounds[i].append(gamma)

    # update adaptive attack tracking
    attackers = [i for i in malicious_indices]  # all potentially attack
    if attackers:
        detected = sum(1 for c in attackers if c not in accepted_ids)
        state._prev_detection_rate = detected / len(attackers)
    else:
        state._prev_detection_rate = 1.0

    # ------------------------------------------------------------------
    # F  Training  (accepted clients only — mirrors sfl.py pattern)
    # ------------------------------------------------------------------
    server.model.to(device)
    for sm in server.models:
        sm.to(device)

    accepted_clients = [c for c in clients if c.id in accepted_ids]

    total_loss = 0.0
    total_acc  = 0.0
    total_batches = 0

    for _epoch in range(local_epochs):
        for c in accepted_clients:
            c.reset_iterator()

        while True:
            smashed_list = []
            labels_map   = {}
            active_clients = []

            for c in accepted_clients:
                smashed, labels = c.forward_pass(global_round=_epoch)
                if smashed is None:
                    continue
                smashed_list.append((c.id, smashed.to(device)))
                labels_map[c.id] = labels.to(device)
                active_clients.append(c)

            if not active_clients:
                break

            grad_map = {}
            for cid, smashed in smashed_list:
                labels = labels_map[cid]
                grad, loss, acc = server.train_step(smashed, labels, client_id=cid)
                grad_map[cid] = grad
                total_loss    += float(loss)
                total_acc     += float(acc)
                total_batches += 1

            for c in active_clients:
                g = grad_map.get(c.id)
                if g is not None:
                    c.backward_pass(g)

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    avg_acc  = total_acc  / total_batches if total_batches > 0 else 0.0

    # ------------------------------------------------------------------
    # G  Weighted FedAvg on client models  (accepted only → broadcast all)
    # ------------------------------------------------------------------
    if accepted_clients:
        client_sds = [c.get_weights() for c in accepted_clients]
        client_ns  = [len(c.dataloader.dataset) for c in accepted_clients]
        global_client_weights = fedavg_state_dicts_weighted(client_sds, client_ns, skip_bn_buffers=True)
        for c in clients:
            c.set_weights(global_client_weights)

    # ------------------------------------------------------------------
    # H  Server-side FedAvg  (accepted clients only)
    # ------------------------------------------------------------------
    if accepted_ids:
        server_ws = [len(clients[i].dataloader.dataset) for i in accepted_ids]
        server.aggregate_server_models(
            active_client_indices=accepted_ids,
            weights=server_ws,
            skip_bn_buffers=True,
        )

    # ------------------------------------------------------------------
    # I  Evaluate  (accuracy + macro-F1)
    # ------------------------------------------------------------------
    ref_client = accepted_clients[0] if accepted_clients else clients[0]
    test_acc   = evaluate_accuracy(ref_client.model, server.model, test_loader, device)
    f1_macro   = evaluate_f1_macro(ref_client.model, server.model, test_loader,
                                   num_classes=cfg.num_classes, device=device)

    state.test_acc.append(test_acc)
    state._prev_val_acc = test_acc

    # ------------------------------------------------------------------
    # J  RL update
    # ------------------------------------------------------------------
    reward          = compute_rl_reward(f1_macro)
    next_rep_scores = [float(g) for g in reputation_scores]
    next_losses     = compute_client_losses(client_models, server.model, client_loaders, device)
    next_state      = build_rl_state_per_client(norm_shifts, next_losses, next_rep_scores)

    state.rl_agent.store(rl_state, float(omega), reward, next_state)
    state.rl_agent.train()

    elapsed = time.time() - t0
    state.latency.append(elapsed)

    print(
        f"Round {rnd + 1:03d} | accepted={len(accepted_ids):02d}/{num_clients} | "
        f"acc={test_acc * 100:.2f}% | f1={f1_macro:.4f} | "
        f"tau={tau:.4f} | omega={omega:.3f} | reward={reward:.4f} | "
        f"time={elapsed:.2f}s"
    )

    return {
        "test_acc":             test_acc,
        "f1_macro":             f1_macro,
        "avg_loss":             avg_loss,
        "avg_acc":              avg_acc,
        "tau":                  tau,
        "omega":                omega,
        "latency":              elapsed,
        "accepted_clients":     accepted_ids,
        "metrics":              round_metrics,
        "reputation_scores":    reputation_scores,
        "centroid_shifts":      shifts_for_detection,
    }
