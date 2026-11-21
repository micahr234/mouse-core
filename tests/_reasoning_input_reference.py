"""Literal RMS normalization followed by the shared input linear layer."""
import torch


def input_reference(reasoner, latent):
    latent = latent.float()
    normalized = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + reasoner.rms_norm_eps)
    normalized = normalized * reasoner.norm.weight
    return torch.nn.functional.linear(normalized, reasoner.input_projection.weight, reasoner.input_projection.bias)
