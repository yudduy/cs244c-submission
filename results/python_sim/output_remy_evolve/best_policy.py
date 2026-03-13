def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "phase": "startup",
            "rtt_base": max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0),
            "trend": 0.0,
            "util_ema": 1.0,
            "last_intersend": 0.0,
            "cooldown": 0,
        }
    st = evolved_policy._state

    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else 1e-6)
    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1e-6)
    slow_rec = max(rec, float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec)
    rtt = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)

    send_rate = 1.0 / send
    rec_rate = 1.0 / rec
    slow_rec_rate = 1.0 / slow_rec

    util = send / rec
    st["util_ema"] = 0.85 * st["util_ema"] + 0.15 * util
    rtt_grad = rtt - st["rtt_base"]
    st["trend"] = 0.8 * st["trend"] + 0.2 * rtt_grad
    st["rtt_base"] = 0.995 * st["rtt_base"] + 0.005 * min(rtt, st["rtt_base"])

    if st["cooldown"] > 0:
        st["cooldown"] -= 1

    if st["phase"] == "startup":
        if rtt > 1.25 or util > 1.20:
            st["phase"] = "drain"
            st["cooldown"] = 2
            target_rate = max(1e-6, rec_rate * 0.95)
            intersend = 1.0 / target_rate
            st["last_intersend"] = intersend
            return RemyAction(window_increment=1, window_multiple=0.92, intersend=intersend)
        target_rate = max(send_rate, rec_rate * 1.30)
        intersend = 1.0 / max(1e-6, target_rate)
        st["last_intersend"] = intersend
        return RemyAction(window_increment=3, window_multiple=1.0, intersend=intersend)

    if st["phase"] == "drain":
        if rtt < 1.12 and util < 1.08:
            st["phase"] = "steady"
        target_rate = max(1e-6, slow_rec_rate * 0.92)
        intersend = 1.0 / target_rate
        intersend = 0.7 * st["last_intersend"] + 0.3 * intersend
        st["last_intersend"] = intersend
        if rtt > 2.0:
            return RemyAction(window_increment=0, window_multiple=0.60, intersend=intersend)
        return RemyAction(window_increment=1, window_multiple=0.96, intersend=intersend)

    # steady phase
    congestion = (rtt - 1.0) + 0.5 * max(0.0, st["trend"]) + 0.6 * max(0.0, st["util_ema"] - 1.0)

    if rtt > 2.0:
        target_rate = max(1e-6, slow_rec_rate * 0.80)
        intersend = 1.0 / target_rate
        intersend = 0.6 * st["last_intersend"] + 0.4 * intersend
        st["last_intersend"] = intersend
        st["cooldown"] = 2
        return RemyAction(window_increment=0, window_multiple=0.55, intersend=intersend)

    if rtt < 1.3:
        if congestion < 0.10:
            inc = 3
            gain = 1.18
        elif congestion < 0.22:
            inc = 2
            gain = 1.08
        else:
            inc = 1
            gain = 1.02
        target_rate = max(1e-6, rec_rate * gain)
        intersend = 1.0 / target_rate
        intersend = 0.75 * st["last_intersend"] + 0.25 * intersend
        st["last_intersend"] = intersend
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    # 1.3 <= rtt <= 2.0 : proportional mild backoff
    sev = min(1.0, max(0.0, (rtt - 1.3) / 0.7))
    mult = 1.0 - 0.22 * sev
    if st["cooldown"] > 0:
        mult = min(mult, 0.95)

    # keep additive non-negative unless severe, to reduce oscillation
    if sev < 0.5:
        inc = 1
    elif sev < 0.8:
        inc = 0
    else:
        inc = -1

    gain = 1.0 - 0.18 * sev
    target_rate = max(1e-6, slow_rec_rate * gain)
    intersend = 1.0 / target_rate
    intersend = 0.8 * st["last_intersend"] + 0.2 * intersend
    st["last_intersend"] = intersend

    return RemyAction(window_increment=inc, window_multiple=mult, intersend=intersend)