def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "util_ewma": 1.0,
            "q_ewma": 1.0,
            "pace_ms": 0.0,
            "startup": True,
        }
    st = evolved_policy._state

    send = memory.send_ewma if memory.send_ewma and memory.send_ewma > 1e-9 else 1e-9
    rec = memory.rec_ewma if memory.rec_ewma and memory.rec_ewma > 1e-9 else send
    slow_rec = memory.slow_rec_ewma if memory.slow_rec_ewma and memory.slow_rec_ewma > 1e-9 else rec
    rtt_ratio = memory.rtt_ratio if memory.rtt_ratio and memory.rtt_ratio > 0 else 1.0
    min_rtt = memory.min_rtt if memory.min_rtt and memory.min_rtt > 1e-9 else 20.0

    util = send / rec
    st["util_ewma"] = 0.85 * st["util_ewma"] + 0.15 * util
    st["q_ewma"] = 0.9 * st["q_ewma"] + 0.1 * rtt_ratio

    # BDP estimate in packets (rate * RTT): (1/rec ms^-1) * min_rtt ms
    bdp_est = min_rtt / max(rec, 1e-6)
    bdp_est = max(1.0, min(300.0, bdp_est))

    # Scale additive step with BDP for cross-rate generalization
    add_step = int(max(1, min(24, round(0.10 * bdp_est + 0.5))))

    # Queue pressure and trend proxies
    q_pressure = rtt_ratio - 1.0
    recv_slowdown = rec / max(slow_rec, 1e-6)  # >1 means recent receive got slower (possible queueing)

    # Rate-based pacing target from receive rate, adapted by queue/utilization signals
    gain = 1.0
    if rtt_ratio > 2.2:
        gain = 0.70
    elif rtt_ratio > 2.0:
        gain = 0.78
    elif rtt_ratio > 1.6:
        gain = 0.88
    elif rtt_ratio > 1.3:
        gain = 0.95
    else:
        # under low queue, be more aggressive if fully utilized
        gain = 1.08 if st["util_ewma"] > 0.95 else 1.02

    if recv_slowdown > 1.08:
        gain *= 0.92
    elif recv_slowdown < 0.96 and rtt_ratio < 1.25:
        gain *= 1.04

    target_pace = rec / max(gain, 1e-6)
    target_pace = max(0.02, min(200.0, target_pace))

    if st["pace_ms"] <= 0:
        st["pace_ms"] = target_pace
    else:
        alpha = 0.25 if rtt_ratio > 1.6 else 0.12
        st["pace_ms"] = (1.0 - alpha) * st["pace_ms"] + alpha * target_pace

    intersend = max(0.01, min(200.0, st["pace_ms"]))

    # Startup: quickly find bandwidth using additive growth + pacing
    if st["startup"]:
        if rtt_ratio > 1.35 or st["util_ewma"] < 0.88:
            st["startup"] = False
        else:
            return RemyAction(window_increment=max(2, add_step), window_multiple=1.0, intersend=intersend)

    # Congestion response
    if rtt_ratio > 2.0:
        mult = 0.55 if rtt_ratio > 2.4 else 0.65
        return RemyAction(window_increment=0, window_multiple=mult, intersend=intersend)

    if rtt_ratio > 1.35:
        dec = max(1, int(round(0.05 * bdp_est)))
        return RemyAction(window_increment=-dec, window_multiple=0.96, intersend=intersend)

    # CRITICAL RULE: when rtt_ratio < 1.3 must increase cwnd
    if rtt_ratio < 1.3:
        if st["util_ewma"] > 0.98 and recv_slowdown <= 1.03:
            inc = add_step
        elif st["util_ewma"] > 0.92:
            inc = max(1, int(round(0.6 * add_step)))
        else:
            inc = 1
        return RemyAction(window_increment=max(1, inc), window_multiple=1.0, intersend=intersend)

    return RemyAction(window_increment=1, window_multiple=1.0, intersend=intersend)