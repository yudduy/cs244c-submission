def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        rec0 = float(memory.rec_ewma) if memory.rec_ewma > 0 else 1.0
        send0 = float(memory.send_ewma) if memory.send_ewma > 0 else rec0
        rr0 = float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0
        min0 = float(memory.min_rtt) if memory.min_rtt > 0 else 10.0
        evolved_policy._state = {
            "mode": "startup",
            "base_rec": max(1e-6, rec0),
            "base_send": max(1e-6, send0),
            "base_rr": max(1.0, rr0),
            "last_rr": rr0,
            "qtrend": 0.0,
            "util_f": 1.0,
            "tick": 0,
            "drain": 0,
            "min_rtt_f": max(1e-3, min0),
        }

    st = evolved_policy._state

    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else st["base_rec"])
    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else st["base_send"])
    slow_rec = max(1e-6, float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec)
    rr = float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0
    min_rtt = max(1e-3, float(memory.min_rtt) if memory.min_rtt > 0 else st["min_rtt_f"])

    st["base_rec"] = 0.997 * st["base_rec"] + 0.003 * rec
    st["base_send"] = 0.997 * st["base_send"] + 0.003 * send
    st["base_rr"] = 0.997 * st["base_rr"] + 0.003 * rr
    st["min_rtt_f"] = min(st["min_rtt_f"], min_rtt)

    util = send / rec
    st["util_f"] = 0.92 * st["util_f"] + 0.08 * util

    rec_trend = rec / max(1e-6, slow_rec)
    rr_delta = rr - st["last_rr"]
    st["last_rr"] = rr
    st["qtrend"] = 0.88 * st["qtrend"] + 0.12 * rr_delta

    bdp = max(1.0, min(2000.0, st["min_rtt_f"] / rec))
    ai = int(max(1, min(700, 0.12 * bdp + 0.95 * math.sqrt(bdp))))
    ai_hi = max(2, int(1.7 * ai))
    ai_md = max(1, int(1.0 * ai))
    ai_lo = max(1, int(0.5 * ai))

    severe = (rr > 2.0) or (rr > 1.75 and (st["qtrend"] > 0.006 or rec_trend > 1.018))
    highq = (rr > 1.42) or (rr > 1.30 and (st["qtrend"] > 0.003 or rec_trend > 1.010))
    lowq = (rr < 1.10 and rec_trend < 1.004 and st["qtrend"] <= 0.0015)

    st["tick"] += 1
    cyc = st["tick"] % 20

    if st["mode"] == "startup":
        if severe or rr > 1.55 or rec_trend > 1.03:
            st["mode"] = "probe"
            st["drain"] = 5
        else:
            gain = 1.20 if rr < 1.08 else (1.12 if rr < 1.18 else 1.06)
            return RemyAction(
                window_increment=max(2, ai_hi),
                window_multiple=1.0,
                intersend=max(0.0, rec / gain),
            )

    st["mode"] = "probe"

    if severe:
        st["drain"] = 5
        return RemyAction(
            window_increment=1,
            window_multiple=0.56,
            intersend=max(0.0, rec / 0.76),
        )

    if st["drain"] > 0:
        st["drain"] -= 1
        return RemyAction(
            window_increment=1,
            window_multiple=0.90,
            intersend=max(0.0, rec / 0.88),
        )

    if highq:
        mult = 0.86 if rr > 1.65 else 0.93
        return RemyAction(
            window_increment=max(1, ai_lo),
            window_multiple=mult,
            intersend=max(0.0, rec / 0.97),
        )

    if rr < 1.08:
        gain = 1.10 if cyc < 14 else 1.04
        return RemyAction(
            window_increment=max(2, ai_hi),
            window_multiple=1.0,
            intersend=max(0.0, rec / gain),
        )

    if rr < 1.30:
        gain = 1.05 if cyc < 12 else 1.01
        return RemyAction(
            window_increment=max(1, ai_md),
            window_multiple=1.0,
            intersend=max(0.0, rec / gain),
        )

    return RemyAction(
        window_increment=max(1, ai_lo),
        window_multiple=0.99 if not lowq else 1.0,
        intersend=max(0.0, rec / (1.005 if lowq else 0.992)),
    )