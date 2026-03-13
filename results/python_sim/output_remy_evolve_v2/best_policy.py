def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        rec0 = float(memory.rec_ewma) if memory.rec_ewma > 0 else 1.0
        send0 = float(memory.send_ewma) if memory.send_ewma > 0 else rec0
        rtt0 = float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0
        min0 = float(memory.min_rtt) if memory.min_rtt > 0 else 10.0
        evolved_policy._state = {
            "rec_ema": max(1e-6, rec0),
            "send_ema": max(1e-6, send0),
            "util_ema": 1.0,
            "rtt_ema": max(1.0, rtt0),
            "q_ema": max(0.0, rtt0 - 1.0),
            "min_rtt_ema": max(1e-3, min0),
            "good": 0,
            "bad": 0,
        }

    st = evolved_policy._state

    rec = max(1e-6, float(memory.rec_ewma) if memory.rec_ewma > 0 else 1.0)
    send = max(1e-6, float(memory.send_ewma) if memory.send_ewma > 0 else rec)
    slow_rec = max(1e-6, float(memory.slow_rec_ewma) if memory.slow_rec_ewma > 0 else rec)
    rtt_ratio = max(1.0, float(memory.rtt_ratio) if memory.rtt_ratio > 0 else 1.0)
    min_rtt = max(1e-3, float(memory.min_rtt) if memory.min_rtt > 0 else st["min_rtt_ema"])

    util = send / rec
    q = max(0.0, rtt_ratio - 1.0)
    rec_trend = slow_rec / rec

    st["rec_ema"] = 0.90 * st["rec_ema"] + 0.10 * rec
    st["send_ema"] = 0.90 * st["send_ema"] + 0.10 * send
    st["util_ema"] = 0.86 * st["util_ema"] + 0.14 * util
    st["rtt_ema"] = 0.86 * st["rtt_ema"] + 0.14 * rtt_ratio
    st["q_ema"] = 0.86 * st["q_ema"] + 0.14 * q
    st["min_rtt_ema"] = min(st["min_rtt_ema"], min_rtt)

    min_base = max(1e-3, st["min_rtt_ema"])
    bdp_pkts = max(1.0, min(5000.0, min_base / rec))
    bdp_scale = max(1.0, math.sqrt(bdp_pkts))

    if rtt_ratio < 1.12 and util < 1.03:
        st["good"] = min(800, st["good"] + 1)
        st["bad"] = max(0, st["bad"] - 1)
    elif rtt_ratio > 1.45 or util > 1.10:
        st["bad"] = min(800, st["bad"] + 1)
        st["good"] = max(0, st["good"] - 1)
    else:
        st["good"] = max(0, st["good"] - 1)
        st["bad"] = max(0, st["bad"] - 1)

    target_q = 0.04 + 0.40 / (1.0 + math.sqrt(bdp_pkts))
    q_err = q - target_q

    pace_gain = 1.0 - 1.05 * q_err - 0.28 * (st["util_ema"] - 1.0) - 0.10 * (st["rtt_ema"] - 1.0)
    if rec_trend > 1.01:
        pace_gain += 0.03
    elif rec_trend < 0.99:
        pace_gain -= 0.03
    if st["good"] > 28:
        pace_gain += 0.03
    if st["bad"] > 10:
        pace_gain -= 0.05

    if rtt_ratio > 3.0:
        pace_gain = min(pace_gain, 0.42)
    elif rtt_ratio > 2.4:
        pace_gain = min(pace_gain, 0.56)
    elif rtt_ratio > 2.0:
        pace_gain = min(pace_gain, 0.70)
    elif rtt_ratio > 1.7:
        pace_gain = min(pace_gain, 0.86)

    pace_gain = max(0.45, min(1.55, pace_gain))
    intersend = rec / max(1e-6, pace_gain)
    intersend = max(0.0, min(2000.0, intersend))

    ai_base = 0.22 * bdp_scale + 0.020 * bdp_pkts
    ai_step = max(1, int(min(320.0, ai_base)))

    if rtt_ratio > 3.0:
        inc = 0
        mult = 0.36
    elif rtt_ratio > 2.4:
        inc = 0
        mult = 0.50
    elif rtt_ratio > 2.0:
        inc = 0
        mult = 0.62
    elif rtt_ratio > 1.7:
        inc = 0
        mult = 0.78
    elif rtt_ratio > 1.45:
        inc = max(0, int(0.01 * ai_step))
        mult = 0.90
    elif rtt_ratio < 1.05:
        inc = max(1, int(0.52 * ai_step + 1))
        mult = 1.0
    elif rtt_ratio < 1.3:
        inc = max(1, int(0.30 * ai_step + 1))
        mult = 1.0
    else:
        inc = max(1, int(0.12 * ai_step + 1))
        mult = 1.0

    if st["good"] > 50 and rtt_ratio < 1.10:
        inc += max(1, int(0.16 * ai_step))
    if st["bad"] > 18 and rtt_ratio > 1.45:
        inc = min(inc, 1)
        mult = min(mult, 0.74)

    if rtt_ratio < 1.3 and inc <= 0:
        inc = 1
        if mult < 1.0:
            mult = 1.0

    if rtt_ratio > 2.0:
        inc = 0

    return RemyAction(window_increment=int(inc), window_multiple=float(mult), intersend=float(intersend))