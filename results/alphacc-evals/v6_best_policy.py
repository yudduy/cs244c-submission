def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        init_rec = float(memory.rec_ewma) if getattr(memory, "rec_ewma", 0) > 0 else 1.0
        evolved_policy._state = {
            "phase": "startup",
            "util_ema": 1.0,
            "rtt_ema": 1.0,
            "rec_fast": max(1e-6, init_rec),
            "rec_slow": max(1e-6, init_rec),
            "base_rec": max(1e-6, init_rec),
            "pace_ema": max(1e-6, init_rec),
            "cooldown": 0,
            "loss_guard": 0,
        }
    st = evolved_policy._state

    send = max(1e-6, float(memory.send_ewma) if getattr(memory, "send_ewma", 0) > 0 else 1.0)
    rec = max(1e-6, float(memory.rec_ewma) if getattr(memory, "rec_ewma", 0) > 0 else send)
    slow_rec_in = max(1e-6, float(memory.slow_rec_ewma) if getattr(memory, "slow_rec_ewma", 0) > 0 else rec)
    rtt_ratio = max(1.0, float(memory.rtt_ratio) if getattr(memory, "rtt_ratio", 0) > 0 else 1.0)
    min_rtt = max(1e-3, float(memory.min_rtt) if getattr(memory, "min_rtt", 0) > 0 else 10.0)

    util = send / rec
    st["util_ema"] = 0.88 * st["util_ema"] + 0.12 * util
    st["rtt_ema"] = 0.90 * st["rtt_ema"] + 0.10 * rtt_ratio
    st["rec_fast"] = 0.74 * st["rec_fast"] + 0.26 * rec
    st["rec_slow"] = 0.982 * st["rec_slow"] + 0.018 * slow_rec_in

    st["base_rec"] = min(st["base_rec"] * 1.00025, st["rec_slow"] * 1.0025, rec * 1.004)
    base_rec = max(1e-6, st["base_rec"])

    if st["cooldown"] > 0:
        st["cooldown"] -= 1
    if st["loss_guard"] > 0:
        st["loss_guard"] -= 1

    delivery_ratio = max(0.50, min(2.10, base_rec / rec))
    trend_ratio = max(0.60, min(1.60, st["rec_slow"] / st["rec_fast"]))
    q = max(0.0, rtt_ratio - 1.0)

    bdp_pkts = max(1.0, min(2000.0, min_rtt / rec))
    scale = max(1.0, min(55.0, math.sqrt(bdp_pkts)))

    inc1 = max(1, int(round(0.22 * scale)))
    inc2 = max(1, int(round(0.45 * scale)))
    inc3 = max(2, int(round(0.78 * scale)))
    inc4 = max(2, int(round(1.10 * scale)))

    # Hard congestion handling first
    if rtt_ratio > 2.0:
        st["phase"] = "steady"
        st["cooldown"] = 8
        st["loss_guard"] = 5
        target = rec * 1.30
        st["pace_ema"] = 0.62 * st["pace_ema"] + 0.38 * target
        return RemyAction(window_increment=0, window_multiple=0.52, intersend=max(0.0, st["pace_ema"]))

    if rtt_ratio > 1.75:
        st["phase"] = "steady"
        st["cooldown"] = max(st["cooldown"], 6)
        target = rec * 1.18
        st["pace_ema"] = 0.66 * st["pace_ema"] + 0.34 * target
        return RemyAction(window_increment=0, window_multiple=0.70, intersend=max(0.0, st["pace_ema"]))

    # Startup / drain
    if st["phase"] == "startup":
        if rtt_ratio > 1.10 or st["util_ema"] > 1.03:
            st["phase"] = "drain"
            st["cooldown"] = 3
            target = rec * 1.06
            st["pace_ema"] = 0.70 * st["pace_ema"] + 0.30 * target
            return RemyAction(window_increment=max(1, inc1), window_multiple=0.92, intersend=max(0.0, st["pace_ema"]))
        target = rec * 0.82
        st["pace_ema"] = 0.66 * st["pace_ema"] + 0.34 * target
        return RemyAction(window_increment=inc4, window_multiple=1.0, intersend=max(0.0, st["pace_ema"]))

    if st["phase"] == "drain":
        if rtt_ratio < 1.04 and st["util_ema"] < 1.01:
            st["phase"] = "steady"
        target = rec * 1.03
        st["pace_ema"] = 0.74 * st["pace_ema"] + 0.26 * target
        return RemyAction(window_increment=1, window_multiple=0.97, intersend=max(0.0, st["pace_ema"]))

    # Steady-state growth logic
    if st["cooldown"] > 0:
        win_inc = 1
    else:
        growth = 0.0
        if rtt_ratio < 1.02:
            growth += 1.15
        elif rtt_ratio < 1.06:
            growth += 0.90
        elif rtt_ratio < 1.12:
            growth += 0.60
        elif rtt_ratio < 1.20:
            growth += 0.32
        elif rtt_ratio < 1.30:
            growth += 0.12

        growth += 0.85 * (delivery_ratio - 1.0)
        growth += 0.35 * (trend_ratio - 1.0)
        growth -= 0.40 * max(0.0, st["util_ema"] - 1.0)
        growth -= 0.22 * max(0.0, st["rtt_ema"] - 1.12)

        if growth > 1.05:
            win_inc = inc4
        elif growth > 0.65:
            win_inc = inc3
        elif growth > 0.30:
            win_inc = inc2
        elif growth > 0.04:
            win_inc = inc1
        else:
            win_inc = 1

    if rtt_ratio < 1.3 and win_inc <= 0:
        win_inc = 1

    # Relative pacing (primary control)
    if rtt_ratio < 1.02:
        pace_gain = 0.80
    elif rtt_ratio < 1.06:
        pace_gain = 0.87
    elif rtt_ratio < 1.12:
        pace_gain = 0.94
    elif rtt_ratio < 1.20:
        pace_gain = 0.99
    elif rtt_ratio < 1.35:
        pace_gain = 1.05
    else:
        pace_gain = 1.11

    pace_gain *= (1.0 - 0.18 * max(0.0, delivery_ratio - 1.0))
    pace_gain *= (1.0 + 0.11 * max(0.0, st["util_ema"] - 1.0))
    pace_gain *= (1.0 + 0.07 * q)

    if st["loss_guard"] > 0:
        pace_gain *= 1.03

    pace_gain = max(0.72, min(1.24, pace_gain))
    target = rec * pace_gain
    st["pace_ema"] = 0.70 * st["pace_ema"] + 0.30 * target

    return RemyAction(window_increment=win_inc, window_multiple=1.0, intersend=max(0.0, st["pace_ema"]))