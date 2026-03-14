def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_states"):
        evolved_policy._states = {}

    mid = id(memory)
    if mid not in evolved_policy._states:
        evolved_policy._states[mid] = {
            "phase": "startup",
            "last_intersend": 0.0,
            "util_ema": 1.0,
            "rtt_ema": 1.0,
            "q_ema": 0.0,
            "cooldown": 0,
        }
    st = evolved_policy._states[mid]

    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else 1e-6)
    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1e-6)
    slow_rec_raw = float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec
    slow_rec = max(1e-6, slow_rec_raw)
    rtt = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)
    min_rtt = max(0.1, float(memory.min_rtt) if memory.min_rtt > 0 else 1.0)

    send_rate = 1.0 / send
    rec_rate = 1.0 / rec
    slow_rate = 1.0 / slow_rec

    util = send / rec
    st["util_ema"] = 0.90 * st["util_ema"] + 0.10 * util
    st["rtt_ema"] = 0.90 * st["rtt_ema"] + 0.10 * rtt
    st["q_ema"] = 0.88 * st["q_ema"] + 0.12 * max(0.0, rtt - 1.0)

    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    bdp_pkts = max(1.0, min_rtt / rec)
    scale = max(1.0, min(14.0, math.sqrt(bdp_pkts)))
    inc_hi = max(2, min(24, int(0.9 * scale + 1.0)))
    inc_lo = max(1, min(12, int(0.45 * scale + 1.0)))

    if st["phase"] == "startup":
        if rtt > 1.18 or st["util_ema"] > 1.10:
            st["phase"] = "drain"
            st["cooldown"] = 3
            target_rate = max(1e-6, slow_rate * 0.90)
            intersend = 1.0 / target_rate
            if st["last_intersend"] > 0:
                intersend = 0.65 * st["last_intersend"] + 0.35 * intersend
            st["last_intersend"] = intersend
            return RemyAction(window_increment=1, window_multiple=0.88, intersend=intersend)

        gain = 1.08 + 0.10 * max(0.0, min(1.0, (1.22 - rtt) / 0.22))
        target_rate = max(send_rate * 1.03, rec_rate * gain)
        intersend = 1.0 / max(1e-6, target_rate)
        if st["last_intersend"] > 0:
            intersend = 0.72 * st["last_intersend"] + 0.28 * intersend
        st["last_intersend"] = intersend
        return RemyAction(window_increment=inc_hi, window_multiple=1.0, intersend=intersend)

    if st["phase"] == "drain":
        if rtt < 1.08 and st["util_ema"] < 1.06:
            st["phase"] = "steady"
        target_rate = max(1e-6, slow_rate * 0.95)
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.70 * st["last_intersend"] + 0.30 * intersend
        st["last_intersend"] = intersend
        if rtt > 2.0:
            return RemyAction(window_increment=0, window_multiple=0.60, intersend=intersend)
        return RemyAction(window_increment=1, window_multiple=0.93, intersend=intersend)

    if rtt > 2.0:
        sev = min(2.0, rtt - 2.0)
        mult = max(0.45, 0.66 - 0.08 * sev)
        target_rate = max(1e-6, slow_rate * (0.80 - 0.06 * min(1.0, sev)))
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.55 * st["last_intersend"] + 0.45 * intersend
        st["last_intersend"] = intersend
        st["cooldown"] = 3
        return RemyAction(window_increment=0, window_multiple=mult, intersend=intersend)

    if rtt < 1.3:
        headroom = max(0.0, 1.3 - rtt) / 0.3
        trend = max(0.0, slow_rate / max(1e-6, rec_rate) - 1.0)
        gain = 1.02 + 0.12 * headroom + 0.05 * min(1.0, trend)
        gain = min(1.24, gain)
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        if st["last_intersend"] > 0:
            intersend = 0.80 * st["last_intersend"] + 0.20 * intersend
        st["last_intersend"] = intersend

        inc = inc_lo if st["cooldown"] == 0 else 1
        return RemyAction(window_increment=max(1, inc), window_multiple=1.0, intersend=intersend)

    sev = min(1.0, max(0.0, (rtt - 1.3) / 0.7))
    mult = max(0.74, min(0.97, 1.0 - (0.08 + 0.17 * sev)))
    if st["cooldown"] > 0:
        mult = min(mult, 0.92)

    if sev < 0.30:
        inc = 1
    elif sev < 0.65:
        inc = 0
    else:
        inc = -1

    rate_gain = 1.0 - (0.07 + 0.15 * sev)
    target_rate = max(1e-6, slow_rate * rate_gain)
    intersend = 1.0 / target_rate
    if st["last_intersend"] > 0:
        intersend = 0.82 * st["last_intersend"] + 0.18 * intersend
    st["last_intersend"] = intersend

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)