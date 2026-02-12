# safesplit.py
import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft

# Use torch for all operations to avoid scipy dependency issues on user env.

StateDict = Dict[str, "torch.Tensor"]

class SafeSplit:
    """
    SafeSplit defense implementation aligned with Algorithm 1 in the paper.
    Now implemented using pure PyTorch logic to avoid Scipy/Numpy incompatibilities.
    """

    def __init__(
        self,
        N: int,
        defense_enabled: bool = True,
        low_freq_ratio: float = 0.25,
        eps: float = 1e-12,
        device: str = "cpu"
    ):
        if N < 2:
            raise ValueError("SafeSplit requires N >= 2.")
        if not (0.0 < low_freq_ratio <= 1.0):
            raise ValueError("low_freq_ratio must be in (0, 1].")

        self.N = N
        self.defense_enabled = defense_enabled
        self.low_freq_ratio = low_freq_ratio
        self.eps = eps
        self.device = device

        # Keep last N+1 backbones (to compute N diffs)
        self.backbone_history: List[StateDict] = []

        self._prev_adn_by_step: Dict[int, float] = {}
        self._step_ids: List[int] = []
        self._next_step_id: int = 0

    # -----------------------------
    # FIFO history maintenance
    # -----------------------------
    def update_history(self, backbone_state: StateDict) -> None:
        """Append a deep-copied backbone checkpoint into FIFO."""
        # Store on CPU to save GPU memory usually, but for calculations we might move to device.
        # Here we keep as is (likely CPU from state_dict).
        self.backbone_history.append({k: v.clone().detach() for k, v in backbone_state.items()})
        self._step_ids.append(self._next_step_id)
        self._next_step_id += 1

        # Keep at most N+1 checkpoints
        while len(self.backbone_history) > self.N + 1:
            self.backbone_history.pop(0)
            self._step_ids.pop(0)

    # -----------------------------
    # Core math helpers
    # -----------------------------
    def _smallest_majority(self, values: torch.Tensor) -> torch.Tensor:
        """
        Implements SMALLESTMAJORITY: sorted(v)[1..N/2+1]
        """
        k = self.N // 2 + 1
        sorted_vals, _ = torch.sort(values)
        # Python slicing is exclusive at end, so :k
        return sorted_vals[:k]

    def _angular_displacement(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        dot = torch.dot(v1, v2)
        n1 = torch.norm(v1)
        n2 = torch.norm(v2)
        cos = dot / (n1 * n2 + self.eps)
        cos = torch.clamp(cos, -1.0, 1.0)
        return float(torch.acos(cos).item())

    def _flatten_state(self, state_dict: StateDict) -> torch.Tensor:
        keys = sorted(state_dict.keys())
        parts = []
        for k in keys:
            parts.append(state_dict[k].view(-1))
        if not parts:
            return torch.zeros(0, device=self.device)
        return torch.cat(parts, dim=0).to(torch.float32)

    # -----------------------------
    # Torch DCT (DCT-II Orthonormal)
    # -----------------------------
    def _dct_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        1D DCT-II using FFT.
        x: input tensor, transform along last dimension.
        """
        N = x.shape[-1]
        if N == 0:
            return x
        
        # Prepare for FFT: reorder -> [x0, x2, ..., x(N-1), x(N-2), ..., x1]
        # x_even = x[..., ::2]
        # x_odd = x[..., 1::2]
        # v = cat(x_even, x_odd.flip)
        
        x_cont = x.contiguous()
        x_even = x_cont[..., ::2]
        x_odd = x_cont[..., 1::2].flip(dims=[-1])
        v = torch.cat([x_even, x_odd], dim=-1)
        
        V = torch.fft.fft(v, dim=-1)
        
        k = torch.arange(N, device=x.device, dtype=x.dtype)
        # factor = 2 * exp(-j * pi * k / (2N))
        # But we need ortho norm.
        # Scipy ortho: 
        # f[0] = 1/sqrt(4N) * ...?
        # Actually for ortho norm:
        # X[k] = sqrt(2/N) * sum... 
        # except X[0] is sqrt(1/N)
        
        # Standard DCT-II via FFT gives: 
        # Y[k] = 2 * Re( V[k] * exp(-j * pi * k / (2N)) )
        
        # Let's compute Y[k] first (standard unnormalized DCT-II)
        # V[k] * exp(...)
        
        angle = -math.pi * k / (2.0 * N)
        # Using complex exp
        # complex_factor = cos(angle) + j sin(angle)
        # we can use polar construction
        complex_factor = torch.polar(torch.ones_like(angle), angle)
        
        Z = V * complex_factor
        Y = 2 * Z.real
        
        # Now applying ortho normalization
        # X_ortho[0] = Y[0] * sqrt(1/(4N))  * 2 ? No using the formula X_k = c_k * ...
        # Standard: X_k = w(k) \sum x_n cos...
        # w(0) = sqrt(1/N), w(k) = sqrt(2/N)
        
        # Our Y is 2 * sum.
        # So we need to divide by 2 then multiply by w(k).
        # Or simply:
        # X[k] = Y[k] * (1/2) * w(k)
        #      = Y[k] * (1/2) * sqrt(2/N)  for k>0
        #      = Y[k] * sqrt(1/(2N))       for k>0
        # For k=0:
        # X[0] = Y[0] * (1/2) * sqrt(1/N)
        #      = Y[0] * sqrt(1/(4N))
        
        # So just scale Y:
        scale = torch.empty_like(Y)
        scale[..., 0] = math.sqrt(1.0 / (4.0 * N))
        scale[..., 1:] = math.sqrt(1.0 / (2.0 * N))
        
        return Y * scale

    def _dct2(self, mat: torch.Tensor) -> torch.Tensor:
        """2D DCT on last 2 dimensions."""
        # DCT row-wise then col-wise
        # To reuse _dct_1d which works on last dim:
        # 1. DCT on rows: mat is (..., H, W). Last dim is W (rows).
        r1 = self._dct_1d(mat)
        # 2. DCT on cols: swap last two dims, DCT, swap back
        r1_t = r1.transpose(-1, -2)
        r2_t = self._dct_1d(r1_t)
        return r2_t.transpose(-1, -2)

    def _low_freq_extract(self, freq: torch.Tensor) -> torch.Tensor:
        """Select low-frequency components."""
        if freq.ndim == 0:
            return freq.reshape(1)
        if freq.ndim == 1:
            k = max(1, int(math.ceil(freq.shape[0] * self.low_freq_ratio)))
            return freq[:k]
        if freq.ndim == 2:
            h, w = freq.shape
            kh = max(1, int(math.ceil(h * self.low_freq_ratio)))
            kw = max(1, int(math.ceil(w * self.low_freq_ratio)))
            return freq[:kh, :kw].reshape(-1)
        
        # ndim >= 3: treat as batch of 2D slices
        leading_dims = freq.shape[:-2]
        h, w = freq.shape[-2], freq.shape[-1]
        freq_reshaped = freq.view(-1, h, w)
        
        kh = max(1, int(math.ceil(h * self.low_freq_ratio)))
        kw = max(1, int(math.ceil(w * self.low_freq_ratio)))
        
        # Slice top-left block
        blocks = freq_reshaped[:, :kh, :kw].reshape(freq_reshaped.shape[0], -1)
        return blocks.reshape(-1)

    def _dct_low_of_update(self, update_state: StateDict) -> torch.Tensor:
        keys = sorted(update_state.keys())
        feats = []
        
        for k in keys:
            x = update_state[k].float() # ensures float
            if x.numel() == 0:
                continue
            
            if x.ndim == 0:
                feats.append(x.reshape(1))
                continue
            
            if x.ndim == 1:
                # 1D DCT
                freq = self._dct_1d(x)
                feats.append(self._low_freq_extract(freq))
                continue
            
            # ndim >= 2: apply 2D DCT on last 2 dims
            # reshape to (-1, H, W)
            orig_shape = x.shape
            if x.ndim > 2:
                leading = int(math.prod(orig_shape[:-2]))
                h, w = orig_shape[-2:]
                x2 = x.reshape(leading, h, w)
            else:
                h, w = orig_shape
                x2 = x.unsqueeze(0)
            
            freq_slices = self._dct2(x2)
            feats.append(self._low_freq_extract(freq_slices))
            
        if not feats:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
            
        return torch.cat(feats, dim=0)

    def _state_subtract(self, a: StateDict, b: StateDict) -> StateDict:
        out: StateDict = {}
        for k in a.keys():
            out[k] = a[k] - b[k]
        return out

    # -----------------------------
    # Main detection (Algorithm 1)
    # -----------------------------
    def check_for_rollback(self) -> Optional[int]:
        if (not self.defense_enabled) or (len(self.backbone_history) < self.N + 1):
            return None

        # history[i] is B_i
        history_states = self.backbone_history
        history_steps = self._step_ids
        
        # Scored items are indices 1..N (corresponding to t-N+1 .. t)
        scored_states = history_states[1:]
        scored_steps = history_steps[1:]
        
        # prev for scored[i] is history[i]
        prev_states = history_states[:-1]

        # --------------------------------------
        # Frequency Domain: S_i, distances, E_i
        # --------------------------------------
        S: List[torch.Tensor] = []
        for i in range(self.N):
            update = self._state_subtract(scored_states[i], prev_states[i])
            S.append(self._dct_low_of_update(update))
            
        # Pairwise distances
        # S is list of 1D tensors (possibly different lengths if shapes changed? No, architecture constant)
        # Convert S to matrix?
        # They should be same size.
        S_stack = torch.stack(S) # (N, FeatureDim)
        
        # Compute pairwise distance matrix e
        # e[i,j] = norm(S[i] - S[j])
        # Using broadcasting
        # (N, 1, D) - (1, N, D) -> (N, N, D)
        diff = S_stack.unsqueeze(1) - S_stack.unsqueeze(0)
        e = torch.norm(diff, dim=2) # (N, N)
        
        # E_i = sum(SmallestMajority(e[i, :]))
        E = torch.zeros(self.N, device=self.device)
        for i in range(self.N):
            E[i] = torch.sum(self._smallest_majority(e[i, :]))

        # --------------------------------------
        # Rotational: theta, adn, omega, R
        # --------------------------------------
        flat_B_list = [self._flatten_state(sd) for sd in scored_states]
        flat_B = torch.stack(flat_B_list) # (N, ParamDim)
        
        # Pairwise angular displacement
        # theta[i,j]
        theta = torch.zeros((self.N, self.N), device=self.device)
        for i in range(self.N):
            for j in range(self.N):
                if i == j: 
                    theta[i, j] = 0.0
                else:
                    theta[i, j] = self._angular_displacement(flat_B[i], flat_B[j])
                    
        adn = torch.zeros(self.N, device=self.device)
        for i in range(self.N):
            adn[i] = torch.sum(self._smallest_majority(theta[i, :]))
            
        omega = torch.zeros(self.N, device=self.device)
        for i in range(self.N):
            step_id = scored_steps[i]
            prev = self._prev_adn_by_step.get(step_id, None)
            if prev is None:
                omega[i] = 0.0
            else:
                omega[i] = adn[i] - prev
                
        R = omega / (2.0 * math.pi)
        
        # Update cache
        self._prev_adn_by_step = {scored_steps[i]: float(adn[i].item()) for i in range(self.N)}
        
        # --------------------------------------
        # Majority & Decision
        # --------------------------------------
        k = self.N // 2 + 1
        
        # Argsort returns indices
        freq_indices = torch.argsort(E)[:k].tolist()
        rot_indices = torch.argsort(R)[:k].tolist()
        
        benign = set(freq_indices).intersection(set(rot_indices))
        
        newest_scored_idx = self.N - 1
        if newest_scored_idx in benign:
            return None
            
        for i in range(newest_scored_idx, -1, -1):
            if i in benign:
                return i + 1 # offset to history index
                
        return 1
