def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "pacing_ewma": None,
            "util_ewma": 1.0,
            "q_ewma": 1.0,
        }
    st = evolved_policy._state

    send = max(float(memory.send_ewma), 1e-6)
    rec = max(float(memory.rec_ewma), 1e-6)
    slow_rec = max(float(memory.slow_rec_ewma), 1e-6)
    rtt_ratio = max(float(memory.rtt_ratio), 1.0)
    min_rtt = max(float(memory.min_rtt), 0.05)

    util = send / rec
    st["util_ewma"] = 0.92 * st["util_ewma"] + 0.08 * util
    st["q_ewma"] = 0.9 * st["q_ewma"] + 0.1 * rtt_ratio

    bdp_pkts = max(1.0, min_rtt / rec)
    bdp_scale = max(1.0, math.sqrt(bdp_pkts))

    growth_pressure = max(0.0, 1.22 - rtt_ratio) + 0.7 * max(0.0, 1.03 - st["util_ewma"])
    drain_pressure = max(0.0, rtt_ratio - 1.38) + 0.8 * max(0.0, st["util_ewma"] - 1.10)

    if rtt_ratio > 2.0:
        wm = 0.50 if rtt_ratio > 2.5 else 0.62
        wi = 0
    elif drain_pressure > 0.22:
        wm = max(0.76, 0.97 - 0.20 * min(1.0, drain_pressure))
        wi = 0
    else:
        wm = 1.0
        base_inc = 0.8 + 0.95 * min(5.0, bdp_scale)
        wi = int(max(1, round(base_inc * (1.0 + 0.75 * growth_pressure))))

    target_rec = rec
    if rtt_ratio > 1.30:
        target_rec = rec * (1.0 + 0.42 * min(1.5, rtt_ratio - 1.30))
    elif rtt_ratio < 1.10 and st["util_ewma"] < 1.0:
        target_rec = rec * (1.0 - 0.14 * min(1.0, 1.0 - st["util_ewma"]))

    target_rec = min(target_rec, slow_rec * 1.45)
    target_rec = max(target_rec, slow_rec * 0.55)

    if st["pacing_ewma"] is None:
        st["pacing_ewma"] = target_rec
    alpha = 0.22 if rtt_ratio > 1.5 else 0.12
    st["pacing_ewma"] = (1.0 - alpha) * st["pacing_ewma"] + alpha * target_rec

    intersend = max(0.0, st["pacing_ewma"])
    return RemyAction(window_increment=wi, window_multiple=wm, intersend=intersend)