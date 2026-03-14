def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {}

    mid = id(memory)
    if mid not in evolved_policy._state:
        evolved_policy._state[mid] = {
            "phase": "startup",
            "util_ema": 1.0,
            "rate_ema": 0.0,
            "q_ema": 0.0,
            "last_intersend": 0.0,
            "cooldown": 0,
        }
    st = evolved_policy._state[mid]

    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else 1e-6)
    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1e-6)
    slow_rec = max(1e-6, float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec)
    rtt_ratio = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)
    min_rtt = max(0.5, float(memory.min_rtt) if memory.min_rtt > 0 else 1.0)

    rec_rate = 1.0 / rec
    slow_rate = 1.0 / max(rec, slow_rec)
    util = send / rec

    st["util_ema"] = 0.90 * st["util_ema"] + 0.10 * util
    st["q_ema"] = 0.92 * st["q_ema"] + 0.08 * max(0.0, rtt_ratio - 1.0)
    if st["rate_ema"] <= 0.0:
        st["rate_ema"] = rec_rate
    else:
        st["rate_ema"] = 0.88 * st["rate_ema"] + 0.12 * rec_rate

    bdp = max(1.0, st["rate_ema"] * min_rtt)
    scale = max(1.0, min(20.0, math.sqrt(bdp)))

    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    if rtt_ratio > 2.0:
        sev = min(1.0, (rtt_ratio - 2.0) / 1.2)
        mult = 0.72 - 0.24 * sev
        target_rate = max(1e-6, slow_rate * (0.84 - 0.18 * sev))
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.45 * st["last_intersend"] + 0.55 * intersend
        st["last_intersend"] = intersend
        st["phase"] = "drain"
        st["cooldown"] = 3
        return RemyAction(window_increment=0, window_multiple=mult, intersend=intersend)

    if st["phase"] == "startup":
        if rtt_ratio > 1.22 or st["util_ema"] > 1.10:
            st["phase"] = "drain"
        gain = 1.16 if bdp > 10 else 1.10
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.65 * st["last_intersend"] + 0.35 * intersend
        st["last_intersend"] = intersend
        inc = max(1, int(round(0.50 * scale)))
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    if st["phase"] == "drain":
        target_rate = max(1e-6, slow_rate * 0.93)
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.60 * st["last_intersend"] + 0.40 * intersend
        st["last_intersend"] = intersend
        if rtt_ratio < 1.12 and st["util_ema"] < 1.04:
            st["phase"] = "steady"
        return RemyAction(window_increment=1, window_multiple=0.95, intersend=intersend)

    if rtt_ratio < 1.3:
        headroom = (1.3 - rtt_ratio) / 0.3
        underuse = max(0.0, 1.03 - st["util_ema"]) / 0.20
        gain = 1.02 + 0.12 * min(1.0, 0.75 * headroom + 0.5 * underuse)
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.78 * st["last_intersend"] + 0.22 * intersend
        st["last_intersend"] = intersend

        inc = max(1, int(round(0.28 * scale + 1.6 * headroom + 0.7 * underuse)))
        if st["cooldown"] > 0:
            inc = max(1, inc - 1)
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    sev = min(1.0, (rtt_ratio - 1.3) / 0.7)
    mult = 1.0 - 0.16 * sev
    if st["cooldown"] > 0:
        mult = min(mult, 0.97)

    target_rate = max(1e-6, slow_rate * (1.0 - 0.12 * sev))
    intersend = 1.0 / target_rate
    if st["last_intersend"] > 0:
        intersend = 0.80 * st["last_intersend"] + 0.20 * intersend
    st["last_intersend"] = intersend

    if sev < 0.45:
        inc = 1
    elif sev < 0.8:
        inc = 0
    else:
        inc = -max(1, int(round(0.12 * scale)))

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)