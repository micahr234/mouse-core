"""Execute example upload cells with a local Hub sink and reload their bundles."""
from typing import Any
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from huggingface_hub import HfApi
import pytest
from torch import nn
from tokenizers import Tokenizer as TextTokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from mouse_core.data import Tokenizer, save_tokenizer, load_tokenizer
from mouse_core.models import save_model, load_model, push_model_to_hub
from tests._hub_example_fixture import capture_upload


@pytest.mark.parametrize('filename', [
    '19_train_online_pendulum_dqn.ipynb', '20_train_sequential_actions_dqn.ipynb',
    '21_train_dqn_cosine.ipynb', '22_train_offline_dqn_time.ipynb',
    '24_train_offline_diffusion_dqn.ipynb',
])
def test_upload_cell_preserves_model_tokenizer_and_custom_artifacts(tmp_path, monkeypatch, filename):
    text = PreTrainedTokenizerFast(tokenizer_object=TextTokenizer(WordLevel({'?': 0, 'x': 1}, unk_token='?')),
                                  unk_token='?')
    tokenizer = Tokenizer(tokenizers={'text': text}, input_fields=[
        dict(type='token', provider='environment', input_field='action_requested', format='x', head_output="all")])
    modules = nn.ModuleDict({'encoder': nn.Linear(2, 2), 'q_head': nn.Linear(2, 4)})
    output = tmp_path / 'uploaded'
    uploads = capture_upload(monkeypatch, output)
    namespace: dict[str, Any] = dict(json=json, Path=Path, TemporaryDirectory=TemporaryDirectory, HfApi=HfApi, nn=nn,
        modules=modules, encoder=modules['encoder'], q_head=modules['q_head'], transition=None,
        MODEL_ID='test/example', tokenizer=tokenizer, text_tokenizer=text,
        save_model=save_model, save_tokenizer=save_tokenizer, push_model_to_hub=push_model_to_hub,
        MAX_STEPS_PER_EPISODE=2, head=SimpleNamespace(action_dim=1), ACTION_COMPONENTS=2,
        ACTION_FIELDS=['a', 'b'], OBSERVATION_FIELDS=['x', 'y'], BACKUP_MODE='next_step')
    path = Path(__file__).parents[1] / 'examples' / filename
    cell = next(c for c in reversed(json.loads(path.read_text())['cells']) if c['cell_type'] == 'code')
    exec(compile(''.join(cell['source']), filename, 'exec'), namespace)
    assert len(uploads) == 1 and uploads[0]['repo_id'] == 'test/example'
    custom = filename.startswith(('19_', '20_'))
    load_model(model=namespace['modules'], repo_id_or_path=output / 'model' if custom else output)
    spec = load_tokenizer(repo_id_or_path=str(output / 'packing' if custom else output), tokenizers={'text': text})
    assert spec.input_fields[0].head_output
    if custom:
        assert json.loads((output / 'environment.json').read_text())['max_episode_steps'] == 2
        assert (output / 'text_tokenizer/tokenizer.json').is_file()
