def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "q_ewma": 1.0,
            "util_ewma": 1.0,
            "trend": 0.0,
        }
    st = evolved_policy._state

    send = max(float(memory.send_ewma), 1e-6)
    rec = max(float(memory.rec_ewma), 1e-6)
    slow_rec = max(float(memory.slow_rec_ewma), 1e-6)
    rtt_ratio = max(float(memory.rtt_ratio), 1.0)
    min_rtt = max(float(memory.min_rtt), 1e-3)

    util = send / rec
    util_slow = send / slow_rec
    q = rtt_ratio - 1.0

    st["q_ewma"] = 0.85 * st["q_ewma"] + 0.15 * q
    st["util_ewma"] = 0.85 * st["util_ewma"] + 0.15 * util
    st["trend"] = 0.8 * st["trend"] + 0.2 * (util - util_slow)

    bdp_pkts = max(1.0, min_rtt / rec)
    scale = max(1.0, min(12.0, math.sqrt(bdp_pkts)))

    # Base pacing from measured receive rate (key for cross-rate generalization)
    target_rate = 1.0 / rec  # pkt/ms

    # Queue and utilization pressure
    queue_pressure = max(0.0, min(2.0, (rtt_ratio - 1.15) / 0.85))
    underutil = max(0.0, min(1.5, util - 1.0))
    overdrive = max(0.0, min(1.5, 1.0 - util))

    # Pace slightly faster when underutilized and low queue, slower when queue builds
    pace_gain = 1.0 + 0.22 * underutil - 0.30 * queue_pressure - 0.10 * max(0.0, st["q_ewma"] - 0.2)
    pace_gain = max(0.6, min(1.35, pace_gain))
    intersend = 1.0 / max(1e-6, target_rate * pace_gain)

    # Add mild burst cap for very low BDP links
    intersend = max(0.0, min(8.0, intersend))

    # Window control with mandatory growth at low RTT ratio
    if rtt_ratio < 1.3:
        inc = int(max(1.0, round(0.55 * scale + 0.8 * underutil + 0.5 * max(0.0, st["trend"]))))
        mult = 1.0
    elif rtt_ratio > 2.0:
        severity = max(0.0, min(1.0, (rtt_ratio - 2.0) / 1.2))
        mult = 1.0 - (0.22 + 0.38 * severity)
        mult = max(0.5, min(0.9, mult))
        inc = -int(max(1.0, round(0.25 * scale * (1.0 + severity))))
    else:
        mid_q = max(0.0, min(1.0, (rtt_ratio - 1.3) / 0.7))
        inc_float = 0.35 * scale * (1.0 - mid_q) + 0.25 * underutil - 0.45 * mid_q
        if inc_float >= 0:
            inc = int(max(1.0 if rtt_ratio < 1.35 else 0.0, round(inc_float)))
        else:
            inc = -int(max(1.0, round(-inc_float)))
        mult = 1.0 - 0.08 * mid_q
        mult = max(0.88, min(1.0, mult))

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)