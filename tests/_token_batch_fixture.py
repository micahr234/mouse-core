"""Build synthetic packed model inputs for tests."""
import numpy as np
from mouse_core.data import make_token_batch


def token_batch_fixture(*, head_positions, **fields):
    mask = np.zeros(len(fields["token_ids"]), dtype=bool)
    mask[np.asarray(head_positions, dtype=np.int64)] = True
    return make_token_batch(head_token_mask=mask, **fields)
