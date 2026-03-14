def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    rtt = max(1.0, float(memory.rtt_ratio))
    send_ewma = max(1e-6, float(memory.send_ewma))
    rec_ewma = max(1e-6, float(memory.rec_ewma))
    slow_rec = max(1e-6, float(memory.slow_rec_ewma))
    min_rtt = max(1e-6, float(memory.min_rtt))

    send_rate = 1.0 / send_ewma
    rec_rate = 1.0 / rec_ewma
    slow_rec_rate = 1.0 / slow_rec

    util = rec_rate / max(send_rate, 1e-9)
    trend = rec_rate / max(slow_rec_rate, 1e-9)

    q = max(0.0, rtt - 1.0)
    q_target = 0.18 + 0.10 * min(1.0, min_rtt / 120.0)
    q_err = q_target - q

    if rtt > 2.6:
        mult = 0.50
        inc = -3
    elif rtt > 2.2:
        mult = 0.62
        inc = -2
    elif rtt > 2.0:
        mult = 0.72
        inc = -1
    elif rtt > 1.7:
        mult = 0.84
        inc = -1
    elif rtt < 1.3:
        mult = 1.0
        grow_score = 0.9 * (util - 0.92) + 0.7 * (trend - 0.98) + 0.6 * max(0.0, q_err)
        if grow_score > 0.18:
            inc = 3
        elif grow_score > 0.04:
            inc = 2
        else:
            inc = 1
    else:
        mult = 1.0
        score = 2.2 * q_err + 0.9 * (util - 1.0) + 0.6 * (trend - 1.0)
        if score > 0.20:
            inc = 2
        elif score > 0.06:
            inc = 1
        elif score < -0.20:
            inc = -2
        elif score < -0.06:
            inc = -1
        else:
            inc = 0

    if rtt < 1.3 and inc <= 0:
        inc = 1
        mult = 1.0

    gain = 1.0 + 0.70 * q_err + 0.18 * (util - 1.0) + 0.12 * (trend - 1.0)
    gain = min(1.35, max(0.65, gain))
    target_rate = rec_rate * gain

    if rtt > 2.3:
        target_rate *= 0.72
    elif rtt > 2.0:
        target_rate *= 0.82
    elif rtt > 1.6:
        target_rate *= 0.92
    elif rtt < 1.15 and util > 0.95 and trend >= 0.98:
        target_rate *= 1.12

    intersend = 1.0 / max(target_rate, 1e-6)
    intersend = min(200.0, max(0.0, intersend))

    return RemyAction(window_increment=int(inc), window_multiple=float(mult), intersend=float(intersend))