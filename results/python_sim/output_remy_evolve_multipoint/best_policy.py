def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "mode": "startup",
            "pressure_ema": 0.0,
            "util_ema": 1.0,
            "last_rr": 1.0,
        }
    st = evolved_policy._state

    send = max(memory.send_ewma, 1e-6)
    rec = max(memory.rec_ewma, 1e-6)
    slow_rec = max(memory.slow_rec_ewma, 1e-6)
    rr = max(memory.rtt_ratio, 1.0)
    min_rtt = max(memory.min_rtt, 1e-3)

    rate_ratio = rec / send
    recv_trend = rec / slow_rec
    q = max(0.0, rr - 1.0)

    bdp_pkts = min_rtt / rec
    bdp_pkts = max(1.0, min(512.0, bdp_pkts))

    util_signal = send / rec
    st["util_ema"] = 0.9 * st["util_ema"] + 0.1 * util_signal

    pressure = (
        0.60 * q
        + 0.25 * max(0.0, rate_ratio - 1.0)
        + 0.15 * max(0.0, recv_trend - 1.0)
    )
    st["pressure_ema"] = 0.85 * st["pressure_ema"] + 0.15 * pressure

    if st["mode"] == "startup":
        if rr > 1.22 or rate_ratio > 1.18 or recv_trend > 1.10:
            st["mode"] = "drain"
            return RemyAction(window_increment=1, window_multiple=0.85, intersend=0.0)
        inc = max(1, int(math.ceil(0.10 * bdp_pkts)))
        inc = min(inc, 24)
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=0.0)

    if st["mode"] == "drain":
        if rr < 1.10 and rate_ratio < 1.05:
            st["mode"] = "steady"
            inc = max(1, int(math.ceil(0.03 * bdp_pkts)))
            return RemyAction(window_increment=inc, window_multiple=1.0, intersend=0.0)
        mult = 0.80 if rr > 1.35 else 0.90
        return RemyAction(window_increment=1, window_multiple=mult, intersend=0.0)

    if rr > 2.0:
        st["mode"] = "drain"
        return RemyAction(window_increment=1, window_multiple=0.5, intersend=0.0)

    mild_cong = (rr > 1.35) or (st["pressure_ema"] > 0.22 and rate_ratio > 1.05)
    if mild_cong:
        mult = 0.82 if rr > 1.6 else 0.90
        return RemyAction(window_increment=1, window_multiple=mult, intersend=0.0)

    if rr < 1.3:
        base_inc = 0.02 * bdp_pkts
        if rr < 1.08 and rate_ratio < 1.02 and recv_trend <= 1.02:
            base_inc += 0.03 * bdp_pkts
        elif rr < 1.15:
            base_inc += 0.015 * bdp_pkts
        inc = max(1, int(math.ceil(base_inc)))
        inc = min(inc, 20)
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=0.0)

    inc = max(1, int(math.ceil(0.008 * bdp_pkts)))
    inc = min(inc, 6)
    return RemyAction(window_increment=inc, window_multiple=1.0, intersend=0.0)