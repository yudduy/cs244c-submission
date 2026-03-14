def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "last_rr": 1.0,
            "rr_trend": 0.0,
            "rate_ratio_ema": 1.0,
        }
    st = evolved_policy._state

    send = max(memory.send_ewma, 1e-6)
    rec = max(memory.rec_ewma, 1e-6)
    slow_rec = max(memory.slow_rec_ewma, 1e-6)
    rr = max(memory.rtt_ratio, 1.0)
    min_rtt = max(memory.min_rtt, 1e-3)

    recv_rate = 1.0 / rec
    send_rate = 1.0 / send
    rate_ratio = recv_rate / max(send_rate, 1e-9)

    st["rate_ratio_ema"] = 0.9 * st["rate_ratio_ema"] + 0.1 * rate_ratio
    rr_delta = rr - st["last_rr"]
    st["last_rr"] = rr
    st["rr_trend"] = 0.88 * st["rr_trend"] + 0.12 * rr_delta

    rr_trend = st["rr_trend"]
    rate_ratio_ema = st["rate_ratio_ema"]

    bdp_pkts = max(1.0, min_rtt / rec)
    bdp_scale = max(1.0, math.sqrt(bdp_pkts))
    bdp_gain = max(1.0, min(8.0, 0.55 * bdp_scale + 0.18 * bdp_pkts))

    recent_vs_slow = slow_rec / rec

    queue_penalty = 1.0
    if rr > 1.03:
        queue_penalty *= 1.0 / (1.0 + 0.55 * (rr - 1.03))
    if rr > 1.25:
        queue_penalty *= 1.0 / (1.0 + 1.05 * (rr - 1.25))
    if rr > 1.6:
        queue_penalty *= 0.9
    if rr > 2.0:
        queue_penalty *= 0.75

    trend_penalty = 1.0
    if rr_trend > 0.006:
        trend_penalty *= 0.93
    elif rr_trend < -0.006:
        trend_penalty *= 1.04

    history_adj = 1.0
    if recent_vs_slow < 0.97:
        history_adj *= 0.95
    elif recent_vs_slow > 1.04:
        history_adj *= 1.03

    target_rate = recv_rate * queue_penalty * trend_penalty * history_adj
    target_rate = max(1e-4, min(recv_rate * 1.08, target_rate))
    intersend = 1.0 / target_rate

    if rr < 1.3:
        headroom = max(0.0, 1.08 - rate_ratio_ema)
        inc_f = 1.0 + bdp_gain * (0.55 + 1.8 * headroom)
        inc = int(max(1.0, min(40.0, inc_f)))
        wm = 1.0
    elif rr > 2.0:
        severity = min(1.0, (rr - 2.0) / 1.2)
        wm = max(0.5, 0.82 - 0.32 * severity)
        dec = int(max(1.0, min(30.0, 0.22 * bdp_gain * (1.0 + severity))))
        inc = -dec
    elif rr > 1.6:
        wm = 0.9
        dec = int(max(1.0, min(18.0, 0.12 * bdp_gain * (1.0 + max(0.0, rr_trend * 40.0)))))
        inc = -dec
    else:
        wm = 1.0
        if rr_trend > 0.01 or rate_ratio_ema < 0.9:
            inc = -int(max(1.0, min(8.0, 0.08 * bdp_gain + 1.0)))
        elif rate_ratio_ema > 1.03 and rr < 1.45:
            inc = int(max(1.0, min(12.0, 0.18 * bdp_gain + 1.0)))
        else:
            inc = 1

    return RemyAction(window_increment=inc, window_multiple=wm, intersend=intersend)