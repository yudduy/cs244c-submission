# Experiment Log

## E-001: Identify delay parameter bug
- **Date**: 2026-03-09
- **Method**: Read C++ network.cc, delay.hh, sender-runner.cc, configrange.hh to trace `rtt=` parameter through to Delay object
- **Result**: `rtt=150` → `config.delay=150` → `Delay(150)` → `tickno + 150`. Single one-way delay, no return path. Python had `rtt_ms / 2.0 = 75ms`.
- **Fix**: `self.delay_ms = rtt_ms` (not `/2.0`)
- **Impact**: Mean error 68-127% → 6.6%

## E-002: Identify send loop vs send-one discrepancy
- **Date**: 2026-03-09
- **Method**: Read rat-templates.cc, sendergang.cc. C++ `Rat::send()` is a single `if` (not `while`). Event loop re-enters at same tickno for multiple sends.
- **Result**: Python while-loop sent all window packets atomically. C++ interleaves network processing between sends.
- **Fix**: Restructured event loop to match C++ `run_simulation()`. `_sender_send_one()` sends one packet per tick.
- **Impact**: 10x tree at high links: 60-70% → 0.2-1.5%

## E-003: Fix ACK timing and utility accounting
- **Date**: 2026-03-09
- **Method**: Read receiver.cc (tick_received set on delivery), utility.hh (uses tick_received - tick_sent), sendergang.cc (receive_feedback before send, accumulate_sending_time after send)
- **Fixes**:
  - Added `tick_received` field to Packet, set in `_delay_tick`
  - Moved delay accounting to use `pkt.tick_received` not sender read time
  - Added receiver event source in `_next_event_time`
  - Added `accumulate_sending_time` after every send attempt
- **Impact**: Minor (<1% change), mostly correctness alignment

## E-004: Full 4-tree calibration
- **Date**: 2026-03-09
- **Method**: Run all 4 trees (1x, 10x, 20x, 100x) × 9 link rates, compare against C++ results
- **Result**: 3.0% mean |error|, 11.7% max. Max at 2.4 Mbps for 20x tree (out-of-range, PRNG-dominated).
