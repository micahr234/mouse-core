"""Explicit prediction layouts for synthetic objective fixtures."""
import torch

def fixture_head_steps(data):
    counts = data.get("readout_counts")
    if counts is not None:
        return torch.repeat_interleave(torch.arange(len(counts), device=counts.device), counts)
    column = data.get("action", next(iter(data.values())))
    return torch.arange(len(column), device=column.device)
