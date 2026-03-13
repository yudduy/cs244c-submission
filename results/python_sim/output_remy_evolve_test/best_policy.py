
def evolved_policy(memory):
    """Copa-inspired: increase when queue low, decrease when queue high."""
    from alphacc.remy_eval import RemyAction
    delta = 0.5
    if memory.rtt_ratio > 1.0 + delta:
        return RemyAction(window_increment=-1, window_multiple=1.0, intersend=0.0)
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
