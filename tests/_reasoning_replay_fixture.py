"""Recorded event fixtures for offline reasoning examples; no live environment."""
from mouse_core.data import Datastore


def pack_prefixes(prompts):
    """Independent synthetic requests for reasoner unit tests, not replay sampling."""
    import numpy as np
    from mouse_core.data import make_token_batch
    lengths = np.asarray([len(prompt) for prompt in prompts], dtype=np.int64)
    ids = np.concatenate(prompts)
    return make_token_batch(
        token_ids=ids, numeric_values=np.zeros(len(ids)), numeric_mask=np.zeros(len(ids), bool),
        sequence_ids=np.repeat(np.arange(len(prompts)), lengths),
        head_mask=np.isin(np.arange(len(ids)), lengths.cumsum() - 1),
        num_sequences=len(prompts), tokenizer_names=("text",),
    )


class DigitsBackend:
    def __len__(self):
        return 16

    def encode(self, text, **kwargs):
        return [int(word) for word in text.split()]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text)}


def reasoning_loader(example, *, stores=None, events_budget=12, seed=7, num_workers=0):
    """Use the notebook's real transform/sampling with a tiny test-owned tokenizer."""
    from mouse_core.data import DataLoader, Tokenizer, compose
    tokenizer = Tokenizer(
        tokenizers={"text": DigitsBackend()},
        input_fields=[
            dict(type="text", provider="environment", input_field="prompt", format="{prompt}"),
            dict(type="token", provider="environment", input_field="action_requested",
                 format="9", head_output=True, autoregressive=True),
        ],
        objective_fields=[dict(input_field=key, missing_value=value) for key, value in
                          [("action", None), ("reward", 0.), ("terminated", False), ("truncated", False)]],
    )
    example.format_events.__globals__["format_observation"] = (
        lambda *, observation, action, reward, requested: "2 3" if action is None else str(observation + 4)
    )
    return DataLoader(
        stores=recorded_stores() if stores is None else stores,
        transform=compose(stages=(example.format_events, tokenizer.encode)),
        events_budget=events_budget, sample_ranges=example.sample_ranges,
        sample_eligible=example.sample_eligible, num_workers=num_workers, seed=seed,
    )


def recorded_stores(*, episodes=2, horizon=2):
    store = Datastore(name='recorded-frozenlake')
    for episode in range(episodes):
        store.append(data=dict(episode_index=episode, step_index=0, provider='coordinator',
                               payload={'_episode_start': True}))
        store.append(data=dict(episode_index=episode, step_index=0, provider='environment',
                               payload={'observation': 0, 'action_requested': True}))
        for step in range(1, horizon + 1):
            store.append(data=dict(episode_index=episode, step_index=step, provider='agent',
                                   payload={'action': 1}))
            store.append(data=dict(episode_index=episode, step_index=step, provider='environment',
                payload={'observation': step % 16, **({'_truncated': True} if step == horizon
                                                     else {'action_requested': True})}))
    return [store]


def configure_notebook_optimizer(example, notebook):
    """Execute the notebook's real optimizer setup with test-owned modules/rates."""
    import ast
    import json
    namespace = example.run_train.__globals__
    nodes: list[ast.stmt] = [node for cell in json.loads(notebook.read_text())['cells']
             if cell['cell_type'] == 'code'
             for node in ast.parse(''.join(cell['source'])).body
             if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == 'optimizer' for target in node.targets)]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'), namespace)
