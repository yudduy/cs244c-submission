def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "base_rec": None,
            "base_send": None,
            "base_rtt": None,
            "last_intersend": 0.0,
        }
    s = evolved_policy._state

    rec = max(1e-6, float(memory.rec_ewma))
    send = max(1e-6, float(memory.send_ewma))
    slow_rec = max(1e-6, float(memory.slow_rec_ewma))
    rtt_ratio = max(1.0, float(memory.rtt_ratio))
    min_rtt = max(1e-3, float(memory.min_rtt))

    if s["base_rec"] is None:
        s["base_rec"] = rec
        s["base_send"] = send
        s["base_rtt"] = rtt_ratio
    else:
        s["base_rec"] = 0.997 * s["base_rec"] + 0.003 * rec
        s["base_send"] = 0.997 * s["base_send"] + 0.003 * send
        s["base_rtt"] = 0.999 * s["base_rtt"] + 0.001 * rtt_ratio

        s["base_rec"] = max(min(s["base_rec"], rec * 2.2), rec * 0.45)
        s["base_send"] = max(min(s["base_send"], send * 2.2), send * 0.45)
        s["base_rtt"] = min(max(1.0, s["base_rtt"]), 3.0)

    base_rec = max(1e-6, s["base_rec"])

    recv_rel = base_rec / rec
    trend = slow_rec / rec
    send_recv = send / rec

    bdp_est = max(0.2, min(600.0, min_rtt / rec))
    bdp_scale = max(1.0, min(18.0, math.sqrt(bdp_est)))

    queue = max(0.0, rtt_ratio - 1.0)
    qnorm = min(1.6, queue / 1.2)
    underuse = max(0.0, send_recv - 1.0)

    boost = 1.0
    boost += 0.30 * (recv_rel - 1.0)
    boost += 0.12 * (trend - 1.0)
    boost += 0.12 * underuse
    boost -= 0.42 * qnorm
    boost = max(0.45, min(1.90, boost))

    target_intersend = rec / boost
    if s["last_intersend"] <= 0.0:
        intersend = target_intersend
    else:
        alpha = 0.22 if rtt_ratio < 1.5 else 0.30
        intersend = (1.0 - alpha) * s["last_intersend"] + alpha * target_intersend

    intersend = max(0.0, min(300.0, intersend))
    s["last_intersend"] = intersend

    if rtt_ratio > 2.4:
        mult = 0.50 if rtt_ratio > 2.9 else 0.58
        return RemyAction(window_increment=0, window_multiple=mult, intersend=intersend)
    if rtt_ratio > 2.0:
        return RemyAction(window_increment=0, window_multiple=0.70, intersend=intersend)

    if rtt_ratio < 1.3:
        inc_f = 1.2 + 0.95 * bdp_scale + 2.0 * max(0.0, recv_rel - 0.92) + 1.0 * underuse
        inc = int(max(1.0, min(36.0, inc_f)))
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    if rtt_ratio < 1.55:
        inc_f = 0.7 + 0.55 * bdp_scale + 1.2 * max(0.0, recv_rel - 0.98)
        inc = int(max(1.0, min(22.0, inc_f)))
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    if rtt_ratio < 1.8:
        return RemyAction(window_increment=0, window_multiple=0.94, intersend=intersend)

    return RemyAction(window_increment=0, window_multiple=0.84, intersend=intersend)