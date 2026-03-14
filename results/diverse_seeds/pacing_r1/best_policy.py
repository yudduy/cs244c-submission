def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_states"):
        evolved_policy._states = {}

    mid = id(memory)
    if mid not in evolved_policy._states:
        evolved_policy._states[mid] = {
            "phase": "startup",
            "util_ema": 1.0,
            "rtt_ema": max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0),
            "trend": 0.0,
            "last_intersend": 0.0,
            "cooldown": 0,
        }
    st = evolved_policy._states[mid]

    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else 1e-6)
    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1e-6)
    slow_rec_raw = float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec
    slow_rec = max(rec, slow_rec_raw)
    rtt = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)
    min_rtt = max(1e-3, float(memory.min_rtt) if memory.min_rtt > 0 else 1.0)

    send_rate = 1.0 / send
    rec_rate = 1.0 / rec
    slow_rate = 1.0 / slow_rec

    util = send / rec
    st["util_ema"] = 0.90 * st["util_ema"] + 0.10 * util
    prev_rtt_ema = st["rtt_ema"]
    st["rtt_ema"] = 0.92 * st["rtt_ema"] + 0.08 * rtt
    st["trend"] = 0.85 * st["trend"] + 0.15 * (st["rtt_ema"] - prev_rtt_ema)

    bdp_pkts = max(1.0, min_rtt / rec)
    bdp_scale = max(0.7, min(6.0, math.sqrt(bdp_pkts)))

    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    if st["phase"] == "startup":
        if rtt > 1.20 or st["util_ema"] > 1.15:
            st["phase"] = "drain"
            st["cooldown"] = 3
        else:
            gain = 1.18 + 0.06 * min(1.0, bdp_scale / 3.0)
            target_rate = max(send_rate * 1.02, rec_rate * gain)
            intersend = 1.0 / max(1e-6, target_rate)
            st["last_intersend"] = intersend
            inc = max(1, int(1 + 1.4 * bdp_scale))
            return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    if st["phase"] == "drain":
        target_rate = max(1e-6, slow_rate * 0.90)
        intersend = 1.0 / target_rate
        intersend = 0.65 * st["last_intersend"] + 0.35 * intersend if st["last_intersend"] > 0 else intersend
        st["last_intersend"] = intersend
        if rtt < 1.10 and st["util_ema"] < 1.08:
            st["phase"] = "steady"
        if rtt > 2.0:
            return RemyAction(window_increment=0, window_multiple=0.60, intersend=intersend)
        return RemyAction(window_increment=1, window_multiple=0.94, intersend=intersend)

    queue_pressure = max(0.0, rtt - 1.0)
    util_pressure = max(0.0, st["util_ema"] - 1.0)
    trend_pressure = max(0.0, st["trend"] * 8.0)
    pressure = 0.65 * queue_pressure + 0.25 * util_pressure + 0.10 * trend_pressure

    if rtt > 2.0:
        sev = min(1.0, (rtt - 2.0) / 1.2)
        mult = 0.65 - 0.20 * sev
        target_rate = max(1e-6, slow_rate * (0.82 - 0.12 * sev))
        intersend = 1.0 / target_rate
        intersend = 0.55 * st["last_intersend"] + 0.45 * intersend if st["last_intersend"] > 0 else intersend
        st["last_intersend"] = intersend
        st["cooldown"] = 3
        return RemyAction(window_increment=0, window_multiple=mult, intersend=intersend)

    if rtt < 1.3:
        headroom = max(0.0, 1.3 - rtt) / 0.3
        gain = 1.02 + 0.18 * headroom - 0.10 * min(1.0, pressure)
        gain = max(1.01, min(1.22, gain))
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        intersend = 0.80 * st["last_intersend"] + 0.20 * intersend if st["last_intersend"] > 0 else intersend
        st["last_intersend"] = intersend

        base_inc = 1 + 0.9 * bdp_scale * headroom
        if st["cooldown"] > 0:
            base_inc *= 0.6
        inc = max(1, int(base_inc))
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    sev = min(1.0, (rtt - 1.3) / 0.7)
    mult = 1.0 - (0.10 + 0.16 * sev)
    if st["cooldown"] > 0:
        mult = min(mult, 0.94)

    gain = 1.0 - (0.06 + 0.14 * sev)
    gain = max(0.78, min(0.96, gain))
    target_rate = max(1e-6, slow_rate * gain)
    intersend = 1.0 / target_rate
    intersend = 0.82 * st["last_intersend"] + 0.18 * intersend if st["last_intersend"] > 0 else intersend
    st["last_intersend"] = intersend

    if rtt < 1.3:
        inc = 1
    elif sev < 0.55:
        inc = 1
    elif sev < 0.85:
        inc = 0
    else:
        inc = -1

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)