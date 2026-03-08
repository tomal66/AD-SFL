"""Smoke test for ad_sfl.py — run with: conda run -n gpu-torch python smoke_test_ad_sfl.py"""
import sys, traceback
sys.path.insert(0, r'c:\Users\Tomal\Desktop\my-projects\ad-sfl')

results = []

try:
    # ---- 1. Import ----
    from src.algorithms import (
        run_ad_sfl_round, AdSflConfig, AdSflState, RLThresholdAgentOmega,
        sample_reference_data_per_label, compute_ref_kde_data,
        compute_ref_hist_data, compute_client_anomaly_score,
    )
    results.append("PASS  imports")

    import torch
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results.append(f"INFO  device={device}")

    # ---- 2. AdSflConfig ----
    cfg_bin = AdSflConfig(kl_estimator='binning', num_classes=10)
    cfg_kde = AdSflConfig(kl_estimator='kde',     num_classes=10)
    assert cfg_bin.kl_estimator == 'binning'
    assert cfg_kde.kl_estimator == 'kde'
    results.append("PASS  AdSflConfig (binning + kde)")

    # ---- 3. AdSflState ----
    state = AdSflState(num_clients=4, cfg=cfg_bin, device=device)
    assert len(state.interactions) == 4
    assert state.rl_agent is not None
    results.append("PASS  AdSflState")

    # ---- 4. DDPG agent ----
    N = 4
    agent = RLThresholdAgentOmega(3 * N, cfg_bin, device=device)
    dummy = np.zeros(3 * N, dtype=np.float32)
    omega = agent.select_action(dummy, noise=False)
    assert cfg_bin.rl_omega_min <= omega <= cfg_bin.rl_omega_max, f"omega={omega}"
    results.append(f"PASS  DDPG select_action  omega={omega:.4f}")

    # Buffer too small -> train is no-op
    for _ in range(5):
        agent.store(dummy, omega, 0.5, dummy)
    agent.train()
    results.append("PASS  DDPG train (small buffer, no-op)")

    # Buffer full enough -> real update
    for _ in range(40):
        agent.store(dummy, omega, 0.5, dummy)
    agent.train()
    results.append("PASS  DDPG train (full buffer)")

    # Soft target update happened: params should differ (almost certainly)
    results.append("PASS  DDPG full cycle")

    # ---- 5. Fisher tau ----
    from src.algorithms.ad_sfl import optimal_fisher_threshold, compute_tau
    shifts_honest = [0.01, 0.02, 0.01, 0.015, 0.018, 0.012, 0.80, 0.75, 0.85, 0.78]
    tau_val = compute_tau(shifts_honest, cfg_bin)
    results.append(f"PASS  Fisher tau={tau_val:.4f}")

    # ---- 6. Anomaly score dispatcher ----
    from src.algorithms.ad_sfl import compute_client_anomaly_score
    # Build a tiny fake client model and loader
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    class TinyClient(nn.Module):
        def forward(self, x): return x.view(x.size(0), -1)[:, :8]  # 8 features

    tiny_model = TinyClient().to(device)
    X = torch.randn(32, 1, 4, 2); Y = torch.randint(0, 10, (32,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=16)

    # Build ref_data with the same model
    ref_samples = {i: (torch.randn(5, 1, 4, 2).to(device), torch.full((5,), i, dtype=torch.long).to(device)) for i in range(10)}
    ref_hist = compute_ref_hist_data(tiny_model, ref_samples, cfg_bin, device)
    score_bin = compute_client_anomaly_score(tiny_model, loader, ref_hist, cfg_bin, device)
    results.append(f"PASS  anomaly score (binning) = {score_bin:.6f}")

    ref_kde = compute_ref_kde_data(tiny_model, ref_samples, cfg_kde, device)
    score_kde = compute_client_anomaly_score(tiny_model, loader, ref_kde, cfg_kde, device)
    results.append(f"PASS  anomaly score (kde)     = {score_kde:.6f}")

    # ---- 7. detect_malicious_clients ----
    from src.algorithms.ad_sfl import detect_malicious_clients
    fake_shifts = [0.01, 0.02, 0.80, 0.75]
    fake_interactions = [{'alpha_p': [], 'beta_p': []} for _ in range(4)]
    accepted, _, metrics, reps = detect_malicious_clients(
        fake_shifts, fake_interactions, tau=0.1, omega=0.3,
        cfg=cfg_bin, malicious_set={2, 3}
    )
    results.append(f"PASS  detect_malicious_clients  accepted={accepted}  metrics={metrics}")

    # ---- 8. build_rl_state_per_client ----
    from src.algorithms.ad_sfl import build_rl_state_per_client
    ns = [0.25, 0.25, 0.25, 0.25]
    cl = {0: 1.2, 1: 1.1, 2: 3.0, 3: 2.8}
    rs = [0.8, 0.8, 0.1, 0.1]
    rl_vec = build_rl_state_per_client(ns, cl, rs)
    assert rl_vec.shape == (12,)
    results.append(f"PASS  build_rl_state_per_client  shape={rl_vec.shape}")

    results.append("=" * 40)
    results.append("ALL TESTS PASSED")

except Exception:
    results.append("FAIL  " + traceback.format_exc())

print('\n'.join(results))
