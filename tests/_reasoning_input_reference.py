"""Independent literal input RMS normalization and SiLU-gated MLP."""
import torch


def input_reference(reasoner, latent, token_embedding):
    latent = latent.float()
    normalized = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + reasoner.rms_norm_eps)
    normalized = normalized * reasoner.norm.weight
    mlp = reasoner.input_projection
    gate = normalized @ mlp.gate_proj.weight.T
    up = normalized @ mlp.up_proj.weight.T
    return token_embedding + ((gate * torch.sigmoid(gate) * up) @ mlp.down_proj.weight.T) * reasoner.input_mlp_output_scale
