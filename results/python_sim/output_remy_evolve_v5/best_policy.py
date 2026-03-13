def evolved_policy(memory):
    from alphacc.remy_eval import RemyAction
    import math

    if not hasattr(evolved_policy, "_state"):
        evolved_policy._state = {
            "max_bw": 0.0,          # pkts/ms
            "startup": True,
            "startup_rounds": 0,
            "last_intersend": 0.0,
            "eq_bw": 0.0
        }
    st = evolved_policy._state

    send_i = memory.send_ewma if memory.send_ewma > 1e-6 else 1e-6
    rec_i = memory.rec_ewma if memory.rec_ewma > 1e-6 else send_i
    slow_rec_i = memory.slow_rec_ewma if memory.slow_rec_ewma > 1e-6 else rec_i

    recv_bw = 1.0 / rec_i
    slow_bw = 1.0 / slow_rec_i
    send_bw = 1.0 / send_i

    st["max_bw"] = max(st["max_bw"] * 0.995, recv_bw)
    if st["eq_bw"] <= 0.0:
        st["eq_bw"] = recv_bw
    else:
        st["eq_bw"] = 0.9 * st["eq_bw"] + 0.1 * recv_bw

    bw_ref = max(st["eq_bw"], slow_bw, 1e-6)
    bdp_est = max(1.0, bw_ref * max(memory.min_rtt, 1e-3))
    scale = max(1.0, min(16.0, math.sqrt(bdp_est)))

    q = memory.rtt_ratio
    pacing_gain = 1.0

    if st["startup"]:
        st["startup_rounds"] += 1
        if q > 1.25 or recv_bw < 0.92 * st["max_bw"] or st["startup_rounds"] > 80:
            st["startup"] = False
        else:
            pacing_gain = 1.20 if q < 1.10 else 1.08
            target_bw = max(recv_bw, slow_bw) * pacing_gain
            intersend = 1.0 / max(target_bw, 1e-6)
            st["last_intersend"] = intersend
            inc = max(2, int(math.ceil(scale * 0.9)))
            return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    if q > 2.0:
        target_bw = max(0.55 * slow_bw, 0.45 * recv_bw, 1e-6)
        intersend = 1.0 / target_bw
        st["last_intersend"] = intersend
        return RemyAction(window_increment=0, window_multiple=0.70, intersend=intersend)

    if q > 1.5:
        pacing_gain = 0.92
        target_bw = max(min(recv_bw, slow_bw) * pacing_gain, 1e-6)
        intersend = 1.0 / target_bw
        st["last_intersend"] = intersend
        return RemyAction(window_increment=0, window_multiple=0.92, intersend=intersend)

    if q < 1.3:
        # Required positive increase in low-queue regime
        if recv_bw >= 0.97 * slow_bw:
            pacing_gain = 1.06
            inc = max(1, int(math.ceil(0.35 * scale)))
        else:
            pacing_gain = 1.01
            inc = max(1, int(math.ceil(0.18 * scale)))
        target_bw = max(recv_bw, slow_bw) * pacing_gain
        intersend = 1.0 / max(target_bw, 1e-6)
        st["last_intersend"] = intersend
        return RemyAction(window_increment=inc, window_multiple=1.0, intersend=intersend)

    # Transitional region 1.3 <= rtt_ratio <= 2.0
    pacing_gain = 0.98 if q > 1.4 else 1.0
    target_bw = max(min(recv_bw, slow_bw) * pacing_gain, 1e-6)
    intersend = 1.0 / target_bw
    st["last_intersend"] = intersend
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=intersend)