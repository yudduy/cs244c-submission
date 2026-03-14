def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    s = getattr(evolved_policy, "_state", None)
    if s is None:
        s = {"pace": 0.0, "rr_s": 1.0}
        evolved_policy._state = s

    send = max(memory.send_ewma, 1e-6)
    rec = max(memory.rec_ewma, 1e-6)
    slow_rec = max(memory.slow_rec_ewma, 1e-6)
    rtt_ratio = max(memory.rtt_ratio, 1.0)
    min_rtt = max(memory.min_rtt, 1e-3)

    # Relative, rate-agnostic estimators
    bdp_est = max(1.0, min(300.0, min_rtt / rec))          # packets
    rate_ratio = rec / send                                 # >1 under-sending, <1 over-sending
    trend = slow_rec / rec                                  # >1 improving recv pace, <1 worsening
    q = max(0.0, rtt_ratio - 1.0)

    # Smooth key signal to reduce oscillation
    s["rr_s"] = 0.9 * s["rr_s"] + 0.1 * rate_ratio
    rr_s = s["rr_s"]

    # Base pacing from observed receive rate
    recv_rate = 1.0 / rec
    pace_factor = 1.0 + 0.55 * q
    if rr_s < 0.92:
        pace_factor *= 1.08
    elif rr_s > 1.08:
        pace_factor *= 0.94
    if trend < 0.97:
        pace_factor *= 1.06
    elif trend > 1.03:
        pace_factor *= 0.97
    pace_factor = max(0.70, min(2.20, pace_factor))
    target_intersend = rec * pace_factor

    # Mild smoothing for pacing output
    if s["pace"] <= 0.0:
        s["pace"] = target_intersend
    else:
        s["pace"] = 0.82 * s["pace"] + 0.18 * target_intersend

    # BDP-scaled additive term for generalization across link rates
    ai = max(1, int(0.18 * math.sqrt(bdp_est) + 0.025 * bdp_est))
    ai_hi = max(ai + 1, int(0.10 * bdp_est + 0.25 * math.sqrt(bdp_est) + 1))

    # Control law
    if rtt_ratio > 2.5:
        wm = 0.45
        wi = -max(2, int(0.18 * bdp_est))
        intersend = s["pace"] * 1.20
    elif rtt_ratio > 2.0:
        wm = 0.58
        wi = -max(1, int(0.10 * bdp_est))
        intersend = s["pace"] * 1.12
    elif rtt_ratio > 1.6:
        wm = 0.82
        wi = -max(1, int(0.03 * bdp_est))
        intersend = s["pace"] * 1.05
    elif rtt_ratio < 1.3:
        wm = 1.0
        if rr_s > 1.05 and trend >= 0.99:
            wi = ai_hi
            intersend = s["pace"] * 0.92
        elif rr_s > 0.98:
            wi = ai
            intersend = s["pace"] * 0.97
        else:
            wi = max(1, ai // 2)
            intersend = s["pace"] * 1.00
    else:
        wm = 1.0
        wi = max(1, ai // 3)
        intersend = s["pace"] * (1.00 + 0.12 * (rtt_ratio - 1.3))

    # Enforce required behavior: always increase when rtt_ratio < 1.3
    if rtt_ratio < 1.3 and wi <= 0:
        wi = 1
        wm = 1.0

    intersend = max(0.0, min(2000.0, intersend))
    return RemyAction(window_increment=int(wi), window_multiple=float(wm), intersend=float(intersend))