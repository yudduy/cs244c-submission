
def evolved_policy(memory):
    """Adaptive CCA using send/receive rate signals for bandwidth estimation.
    Tracks state via function attributes for rate-based pacing."""
    from alphacc.remy_eval import RemyAction
    import math

    # Initialize state
    if not hasattr(evolved_policy, '_state'):
        evolved_policy._state = {'phase': 'startup', 'last_ratio': 1.0, 'stable_count': 0}
    st = evolved_policy._state

    # Bandwidth signal: if rec_ewma < send_ewma, we're sending faster than receiving
    if memory.send_ewma > 0 and memory.rec_ewma > 0:
        rate_ratio = memory.rec_ewma / memory.send_ewma  # >1 = bottlenecked
    else:
        rate_ratio = 1.0

    # Phase-based control
    if st['phase'] == 'startup':
        # Exponential growth until RTT inflates or rate saturates
        if memory.rtt_ratio > 1.3 or rate_ratio > 1.5:
            st['phase'] = 'steady'
            return RemyAction(window_increment=0, window_multiple=0.9, intersend=0.0)
        return RemyAction(window_increment=2, window_multiple=1.0, intersend=0.0)

    # Steady state: delay-based with rate awareness
    if memory.rtt_ratio > 2.0:
        # Heavy congestion: halve
        st['stable_count'] = 0
        return RemyAction(window_increment=0, window_multiple=0.5, intersend=0.0)
    elif memory.rtt_ratio > 1.3:
        # Moderate: back off gently
        return RemyAction(window_increment=-1, window_multiple=1.0, intersend=0.0)
    elif rate_ratio < 1.1:
        # Receiving as fast as sending, room to grow
        return RemyAction(window_increment=2, window_multiple=1.0, intersend=0.0)
    else:
        return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)
