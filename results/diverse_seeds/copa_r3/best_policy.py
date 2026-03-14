def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "base_rec": None,
            "q_ema": 0.0,
            "util_ema": 0.0,
            "rate_ema": 1.0,
        }
    st = evolved_policy._state

    send = max(memory.send_ewma, 1e-6)
    rec = max(memory.rec_ewma, 1e-6)
    slow_rec = max(memory.slow_rec_ewma, 1e-6)
    rtt_ratio = max(memory.rtt_ratio, 1.0)
    min_rtt = max(memory.min_rtt, 1e-3)

    if st["base_rec"] is None:
        st["base_rec"] = slow_rec

    st["base_rec"] = min(st["base_rec"], slow_rec) * 0.998 + slow_rec * 0.002
    base_rec = max(st["base_rec"], 1e-6)

    q = max(0.0, rtt_ratio - 1.0)
    st["q_ema"] = 0.90 * st["q_ema"] + 0.10 * q
    q_ema = st["q_ema"]

    send_rec_ratio = send / rec
    st["util_ema"] = 0.88 * st["util_ema"] + 0.12 * (send_rec_ratio - 1.0)
    util_ema = st["util_ema"]

    recv_rate = 1.0 / rec
    base_rate = 1.0 / base_rec
    rate_ratio = recv_rate / max(base_rate, 1e-9)
    st["rate_ema"] = 0.95 * st["rate_ema"] + 0.05 * rate_ratio
    rate_ema = st["rate_ema"]

    bdp_est = max(1.0, min(600.0, min_rtt / rec))
    scale = max(1.0, min(30.0, 1.6 * math.sqrt(bdp_est)))

    if rtt_ratio > 2.6:
        pace_factor = 0.58
    elif rtt_ratio > 2.2:
        pace_factor = 0.68
    elif rtt_ratio > 2.0:
        pace_factor = 0.76
    elif rtt_ratio > 1.6:
        pace_factor = 0.88
    elif rtt_ratio > 1.3:
        pace_factor = 0.96
    else:
        pace_factor = 1.06

    if util_ema > 0.08 and rtt_ratio < 1.35:
        pace_factor *= 1.05
    if q_ema > 0.9:
        pace_factor *= 0.88
    elif q_ema > 0.5:
        pace_factor *= 0.94

    target_rate = max(1e-6, recv_rate * pace_factor)
    intersend = 1.0 / target_rate
    intersend = max(0.0, min(3.0 * rec, intersend))

    if rtt_ratio > 2.6:
        wm = 0.62
        wi = -max(1, int(1.2 * scale))
    elif rtt_ratio > 2.2:
        wm = 0.72
        wi = -max(1, int(0.9 * scale))
    elif rtt_ratio > 2.0:
        wm = 0.80
        wi = -max(1, int(0.6 * scale))
    elif rtt_ratio > 1.6:
        wm = 0.90
        wi = -max(1, int(0.25 * scale))
    elif rtt_ratio >= 1.3:
        wm = 0.98
        wi = 0
    else:
        wm = 1.0
        low_q_gain = max(0.15, 1.35 - rtt_ratio)
        underfill = max(0.0, min(1.5, 1.0 - rate_ema))
        probe_bias = 1.0 + 0.35 * max(0.0, util_ema) + 0.45 * underfill
        wi = max(1, int(scale * 0.45 * low_q_gain * probe_bias))

    return RemyAction(window_increment=wi, window_multiple=wm, intersend=intersend)