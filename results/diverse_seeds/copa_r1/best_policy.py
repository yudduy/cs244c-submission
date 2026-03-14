def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "base_rec": None,
            "base_rtt": 1.0,
            "util_ema": 1.0,
        }
    s = evolved_policy._state

    rec = max(memory.rec_ewma, 1e-6)
    send = max(memory.send_ewma, 1e-6)
    slow_rec = max(memory.slow_rec_ewma, 1e-6)
    rtt_ratio = max(memory.rtt_ratio, 1.0)
    min_rtt = max(memory.min_rtt, 1e-3)

    if s["base_rec"] is None:
        s["base_rec"] = rec
        s["base_rtt"] = rtt_ratio
    else:
        if rtt_ratio < 1.12:
            a = 0.01
        elif rtt_ratio < 1.30:
            a = 0.003
        else:
            a = 0.0008
        s["base_rec"] = (1.0 - a) * s["base_rec"] + a * rec
        s["base_rtt"] = 0.995 * s["base_rtt"] + 0.005 * rtt_ratio

    recv_rate_fast = 1.0 / rec
    recv_rate_slow = 1.0 / slow_rec
    send_rate = 1.0 / send

    mismatch = send / rec
    trend = rec / slow_rec
    queue = max(0.0, rtt_ratio - 1.0)

    # BDP estimate in packets (rate * RTT), clipped for robustness
    bdp_pkts = max(1.0, min(300.0, min_rtt / rec))
    scale = max(1.0, min(24.0, math.sqrt(bdp_pkts)))

    # Utilization proxy from delivered rate vs baseline no-queue rate
    base_rate = 1.0 / max(s["base_rec"], 1e-6)
    util = recv_rate_fast / max(base_rate, 1e-6)
    s["util_ema"] = 0.97 * s["util_ema"] + 0.03 * util
    util_e = s["util_ema"]

    # Rate-based pacing target (primary control path)
    target_rate = recv_rate_slow

    if rtt_ratio < 1.06:
        target_rate *= 1.16
    elif rtt_ratio < 1.15:
        target_rate *= 1.09
    elif rtt_ratio < 1.30:
        target_rate *= 1.02
    elif rtt_ratio < 1.60:
        target_rate *= 0.93
    elif rtt_ratio < 2.00:
        target_rate *= 0.82
    else:
        target_rate *= 0.64

    if trend < 0.985:
        target_rate *= 1.05
    elif trend > 1.02:
        target_rate *= 0.92

    if mismatch < 0.95:
        target_rate *= 1.05
    elif mismatch > 1.06:
        target_rate *= 0.91

    if util_e < 0.90 and rtt_ratio < 1.25:
        target_rate *= 1.06
    elif util_e > 1.05 and rtt_ratio > 1.35:
        target_rate *= 0.95

    target_rate = max(1e-4, min(target_rate, recv_rate_fast * 1.35))
    intersend = 1.0 / target_rate

    # Window adaptation (secondary, must satisfy required behavior)
    if rtt_ratio < 1.3:
        inc = max(1, int(math.ceil(0.30 * scale + 0.08 * scale * max(0.0, 1.18 - rtt_ratio))))
        mult = 1.0
    elif rtt_ratio > 2.0:
        inc = 0
        if rtt_ratio > 3.0:
            mult = 0.45
        elif rtt_ratio > 2.5:
            mult = 0.55
        else:
            mult = 0.68
    else:
        if queue > 0.55 or mismatch > 1.10:
            inc = -max(1, int(math.ceil(0.12 * scale)))
            mult = 0.97
        elif queue < 0.22 and mismatch < 1.02 and trend <= 1.01:
            inc = max(1, int(math.ceil(0.12 * scale)))
            mult = 1.0
        else:
            inc = 0
            mult = 1.0

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)