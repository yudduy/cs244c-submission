def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    st = getattr(evolved_policy, "_state", None)
    if st is None:
        st = {}
        evolved_policy._state = st

    send = max(float(memory.send_ewma), 1e-6)
    rec = max(float(memory.rec_ewma), 1e-6)
    slow_rec = max(float(memory.slow_rec_ewma), 1e-6)
    rtt_ratio = max(float(memory.rtt_ratio), 1.0)
    min_rtt = max(float(memory.min_rtt), 1e-3)

    util = send / rec
    trend = slow_rec / rec
    bdp_est = max(1.0, min(4000.0, min_rtt / rec))
    bdp_scale = max(1.0, math.sqrt(bdp_est))

    rtt_s = st.get("rtt_s", rtt_ratio)
    rtt_s = 0.86 * rtt_s + 0.14 * rtt_ratio
    prev_rtt_s = st.get("prev_rtt_s", rtt_s)
    drtt = rtt_s - prev_rtt_s
    st["rtt_s"] = rtt_s
    st["prev_rtt_s"] = rtt_s

    util_s = st.get("util_s", util)
    util_s = 0.85 * util_s + 0.15 * util
    st["util_s"] = util_s

    trend_s = st.get("trend_s", trend)
    trend_s = 0.90 * trend_s + 0.10 * trend
    st["trend_s"] = trend_s

    rec_s = st.get("rec_s", rec)
    rec_s = 0.88 * rec_s + 0.12 * rec
    st["rec_s"] = rec_s

    q = max(0.0, rtt_s - 1.0)
    q_norm = min(1.0, q / 1.5)

    pace_gain = 1.06 - 0.34 * q_norm
    if drtt > 0.015:
        pace_gain *= 0.90
    elif drtt < -0.008:
        pace_gain *= 1.04

    if trend_s < 0.985:
        pace_gain *= 0.93
    elif trend_s > 1.015:
        pace_gain *= 1.03

    if util_s < 0.94:
        pace_gain *= 0.97
    elif util_s > 1.06:
        pace_gain *= 1.03

    pace_gain = max(0.62, min(1.22, pace_gain))
    intersend = max(0.0, min(200.0, rec_s / pace_gain))

    ai_small = max(1, int(0.18 * bdp_scale))
    ai_med = max(1, int(0.34 * bdp_scale))
    ai_big = max(1, int(0.58 * bdp_scale))

    if rtt_ratio < 1.3:
        if rtt_s < 1.08 and util_s >= 0.99 and trend_s >= 0.995:
            win_inc = ai_big
        elif util_s >= 0.96 and trend_s >= 0.99:
            win_inc = ai_med
        else:
            win_inc = ai_small
        if drtt > 0.02:
            win_inc = max(1, int(0.7 * win_inc))
        win_mult = 1.0
    elif rtt_s > 2.0:
        sev = min(1.0, max(0.0, (rtt_s - 2.0) / 1.2))
        md = 0.82 - 0.30 * sev
        if drtt > 0.01:
            md -= 0.06
        if trend_s < 0.98:
            md -= 0.05
        if util_s < 0.95:
            md -= 0.03
        win_mult = max(0.45, min(0.88, md))
        win_inc = 0
    else:
        if rtt_s < 1.45 and util_s > 0.99 and trend_s >= 0.995 and drtt <= 0.01:
            win_inc = ai_small
            win_mult = 1.0
        elif rtt_s > 1.75 or (drtt > 0.015 and trend_s < 0.99):
            win_inc = 0
            win_mult = 0.90 if rtt_s < 2.0 else 0.82
        else:
            win_inc = max(1, int(0.12 * bdp_scale))
            win_mult = 1.0

    return RemyAction(
        window_increment=int(win_inc),
        window_multiple=float(win_mult),
        intersend=float(intersend),
    )