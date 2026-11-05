"""Device requirements for tests that differentiate through FlexAttention."""

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "flex_training: differentiates through FlexAttention; requires CUDA")


@pytest.fixture(autouse=True)
def flex_training_device(request):
    if request.node.get_closest_marker("flex_training") is None:
        yield
        return
    if not torch.cuda.is_available():
        pytest.skip("PyTorch FlexAttention backward requires CUDA")
    if request.node.get_closest_marker("flex_training").kwargs.get("default_device", True):
        with torch.device("cuda"):
            yield
    else:
        yield
