"""Helper methods to split a model at a layer.

Usually used if we wrap standard PyTorch models dynamically.
Here we've implemented explicit Client and Server classes in their respective
modules, which is standard practice for SplitFed setups.
"""

from typing import Tuple
import torch.nn as nn
import torch

def split_model_output(output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper to detach the client output and require gradients on the detached tensor.
    This simulates the network split where the client sends `client_output` to the server,
    but we must keep a graph connection on the client to backpropagate gradients later.
    
    Args:
        output: The forward pass output tensor from the client model.
        
    Returns:
        client_out: The original output (used on client side to attach grad hooks if needed)
        server_in: The cloned, detached tensor with requires_grad=True (sent to server)
    """
    server_in = output.clone().detach().requires_grad_(True)
    return output, server_in
