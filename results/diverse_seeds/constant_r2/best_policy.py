def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "rec_fast": 1.0,
            "rec_slow": 1.0,
            "send_slow": 1.0,
            "rtt_slow": 1.0,
            "bdp": 16.0,
            "pace": 1.0,
            "err_i": 0.0,
        }
    st = evolved_policy._state

    send = max(1e-6, float(memory.send_ewma))
    rec = max(1e-6, float(memory.rec_ewma))
    slow_rec_obs = max(1e-6, float(memory.slow_rec_ewma))
    rtt_ratio = max(1.0, float(memory.rtt_ratio))
    min_rtt = max(0.05, float(memory.min_rtt))

    st["rec_fast"] = 0.72 * st["rec_fast"] + 0.28 * rec
    st["rec_slow"] = 0.985 * st["rec_slow"] + 0.015 * slow_rec_obs
    st["send_slow"] = 0.97 * st["send_slow"] + 0.03 * send
    st["rtt_slow"] = 0.985 * st["rtt_slow"] + 0.015 * rtt_ratio

    bdp_inst = min_rtt / rec
    bdp_inst = max(1.0, min(8000.0, bdp_inst))
    st["bdp"] = 0.95 * st["bdp"] + 0.05 * bdp_inst
    bdp = max(1.0, st["bdp"])

    util = send / rec
    qtrend = st["rec_fast"] / max(1e-6, st["rec_slow"])
    rtt_dev = rtt_ratio / max(1e-6, st["rtt_slow"])

    target_rr = 1.10 + 0.28 * math.tanh((math.log(max(1.0, bdp)) - 1.8) / 1.4)
    target_rr = max(1.05, min(1.42, target_rr))

    err = target_rr - rtt_ratio
    st["err_i"] = 0.92 * st["err_i"] + 0.08 * err
    ctrl = 0.55 * err + 0.30 * st["err_i"]

    pace_gain = 1.0 - 0.42 * (rtt_ratio - target_rr) - 0.20 * (qtrend - 1.0) + 0.10 * (util - 1.0)
    if rtt_dev > 1.04:
        pace_gain *= 0.90
    elif rtt_dev < 0.98:
        pace_gain *= 1.03
    pace_gain = max(0.60, min(1.22, pace_gain))

    target_pace = rec * pace_gain
    st["pace"] = 0.78 * st["pace"] + 0.22 * target_pace
    intersend = max(1e-4, min(2000.0, st["pace"]))

    scale = max(1.0, math.sqrt(bdp))
    base_ai = max(1, int(0.18 * scale + 0.5))
    boost_ai = int(max(0.0, ctrl) * (0.80 * scale))
    window_increment = base_ai + boost_ai
    window_multiple = 1.0

    if rtt_ratio < 1.10:
        window_increment += max(1, int(0.06 * scale))
    elif rtt_ratio < 1.30:
        window_increment += 0
    elif rtt_ratio < 1.60:
        window_increment = max(1, int(0.55 * window_increment))
        window_multiple = 0.996
    elif rtt_ratio < 2.0:
        window_increment = 0
        md = 0.93 - 0.10 * min(1.0, (rtt_ratio - 1.60) / 0.40)
        window_multiple = max(0.80, md)
    else:
        window_increment = 0
        sev = min(1.0, (rtt_ratio - 2.0) / 1.2)
        window_multiple = 0.72 - 0.18 * sev

    if qtrend < 0.97:
        window_multiple = min(window_multiple, 0.90)
        window_increment = min(window_increment, max(1, int(0.35 * scale)))
    elif qtrend > 1.03 and rtt_ratio < 1.35:
        window_increment += max(1, int(0.04 * scale))

    if util < 0.90 and rtt_ratio >= 1.20:
        window_multiple = min(window_multiple, 0.88)
        window_increment = min(window_increment, 1)

    if rtt_dev > 1.08:
        window_multiple = min(window_multiple, 0.86)
    if rtt_ratio > 2.0:
        window_multiple = min(window_multiple, 0.70)

    if rtt_ratio < 1.3 and window_increment <= 0:
        window_increment = max(1, base_ai)

    return RemyAction(
        window_increment=int(window_increment),
        window_multiple=float(window_multiple),
        intersend=float(intersend),
    )