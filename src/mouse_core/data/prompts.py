"""FrozenLake ``Tokenizer`` ``group_prefix`` text for examples and benches."""


def frozenlake_group_prefix(*, max_task_episodes: int) -> str:
    """FrozenLake game / format / strategy text for ``Tokenizer`` ``group_prefix``.

    ``max_task_episodes`` is the episode budget for one task (mouse-gym
    ``EnvConfig.max_task_episodes``, experiment ``MAX_EPISODES_PER_TASK``).
    Inserted once per ``task_index`` segment; step lines follow as
    ``{action},{observation}`` with optional ``r=`` / ``d=`` / ``e=``.
    """
    if max_task_episodes < 1:
        raise ValueError(
            "frozenlake_group_prefix max_task_episodes must be >= 1, got "
            f"{max_task_episodes!r}"
        )
    return (
        "Your job is to predict the future sum of rewards in FrozenLake. "
        "Navigate a grid; reach the goal for reward; a hole ends the "
        "episode with none. You have {max_task_episodes} episodes to solve "
        "the task. The grid is permuted, so squares are not in order; "
        "action ids may be remapped.\n"
        "Strategy: explore; keep a mental map of what has and has not been "
        "explored; avoid holes you have already fallen in; once you have a "
        "path to the goal, repeat it.\n"
        "Predict when you see a new line. Step format: "
        "action,observation[,r=reward][,d=done][,e=episode].\n"
    ).format(max_task_episodes=max_task_episodes)
