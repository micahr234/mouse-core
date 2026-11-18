"""Offline reasoning notebooks load saved events; evaluation lives in example 15."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from tests._reasoning_replay_fixture import recorded_stores
from tests._hub_example_fixture import capture_upload
from tests.test_autoregressive_reasoning import local_checkpoint

EXAMPLES = Path(__file__).parents[1] / 'examples'


@pytest.mark.parametrize('kind,filename,class_name', [
    ('residual', '25_train_residual_reasoning_dqn.ipynb', 'ResidualDqn'),
    ('autoregressive', '23_train_autoregressive_reasoning_dqn.ipynb', 'ReasoningDqn'),
])
def test_offline_checkpoint_loads_and_evaluates_only_in_inference_notebook(tmp_path, monkeypatch, kind, filename, class_name):
    source, output = tmp_path / 'base', tmp_path / 'saved'
    uploads = capture_upload(monkeypatch, output)
    local_checkpoint(source)
    namespace = {}
    def forbidden_environment(*args, **kwargs):
        raise AssertionError('Offline training notebooks must never create an environment')
    # Real model construction, tokenization, offline loading and checkpoint saving
    # on CPU; CUDA-only gradient training has its own tests.
    with monkeypatch.context() as patch:
        patch.setattr(gym, 'make', forbidden_environment)
        for index, cell in enumerate(json.loads((EXAMPLES / filename).read_text())['cells']):
            if cell['cell_type'] != 'code':
                continue
            code = ''.join(cell['source'])
            if 'if not torch.cuda.is_available():' in code:
                start = code.index('if not torch.cuda.is_available():')
                end = code.index('torch.manual_seed(SEED)', start)
                code = code[:start] + 'device = torch.device("cpu")\n' + code[end:]
            exec(compile(code, filename, 'exec'), namespace)
            if index == 1:
                namespace.update(PRETRAINED=str(source), load_stores_from_hub=lambda **_: recorded_stores(),
                    SAMPLE_BUDGET=2, NUM_WORKERS=0, MAX_EPISODE_STEPS=2, NUM_CYCLES=0, MAX_REASONING_STEPS=2,
                    GENERATION_START_TOKEN='word2', MAX_NEW_TOKENS=64, MAX_ACTIVE_REQUESTS=2)
    assert len(uploads) == 1
    assert 'pytorch_model.bin' in uploads[0]['files']
    import huggingface_hub
    downloads = []
    def snapshot_download(*, repo_id, repo_type):
        downloads.append(repo_id)
        assert repo_type == 'model'
        return str(output)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', snapshot_download)
    inference = {}
    cells = json.loads((EXAMPLES / '15_inference.ipynb').read_text())['cells']
    start = next(i for i, cell in enumerate(cells)
                 if cell['cell_type'] == 'markdown'
                 and ''.join(cell['source']).startswith('## Evaluate an Example ')
                 and kind.capitalize() in ''.join(cell['source']))
    for cell in cells[start + 1:]:
        source_code = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown' and source_code.startswith('## '):
            break
        if cell['cell_type'] == 'code':
            exec(compile(source_code, '15_inference.ipynb', 'exec'), inference)
    loaded = inference[f'load_{kind}_checkpoint'](uploads[0]['repo_id'], device=torch.device('cpu'))
    assert downloads == [uploads[0]["repo_id"]]
    policy = loaded[0]
    for name, parameter in namespace['online'].named_parameters():
        torch.testing.assert_close(dict(policy.named_parameters())[name], parameter)
    _, spec, backend, settings = loaded
    original_steps = namespace['train_transform'](recorded_stores(episodes=1)[0].to_dataset())
    if kind == 'residual':
        encoded = inference['encode_residual_observation'](observation=0, action=None, reward=0.,
            token_spec=spec, text_backend=backend, settings=settings)
        np.testing.assert_array_equal(encoded['token_ids'].cpu(), original_steps[0].token_ids)
    names = (('residual_policy', 'residual_tokenizer', 'residual_backend', 'residual_settings')
             if kind == 'residual' else ('autoregressive_policy', 'autoregressive_tokenizer', 'autoregressive_backend', 'autoregressive_settings'))
    inference.update(zip(names, loaded, strict=True))
    if kind == 'autoregressive':
        events = [inference['format_autoregressive_observation'](
            observation=step, action=None if step == 0 else 1, reward=0., step_index=step)
            for step in range(2)]
        for actual, expected in zip(spec.encode(events), original_steps[:2], strict=True):
            np.testing.assert_array_equal(actual.token_ids, expected.token_ids)
            np.testing.assert_array_equal(actual.head_mask, expected.head_mask)
    readouts = []
    hook = policy.register_forward_pre_hook(
        lambda _module, args: readouts.append(int(args[0]['head_mask'].sum()))
    )
    try:
        returns = inference[f'run_{kind}_eval'](num_episodes=1, seed=1000000)
    finally:
        hook.remove()
    assert len(returns) == 1 and np.isfinite(returns).all()
    assert readouts and all(count == 1 for count in readouts)
    assert all(parameter.grad is None for parameter in policy.parameters())
