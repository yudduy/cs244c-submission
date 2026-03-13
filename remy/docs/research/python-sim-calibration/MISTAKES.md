# Mistakes

## M-001: Assumed RTT = 2 × one-way delay
- **What happened**: Set `delay_ms = rtt_ms / 2.0`, assuming RTT splits into forward+return
- **Why**: Intuition from real networks where RTT = 2× propagation. Remy's model has no return path.
- **Fix**: Read the actual C++ (delay.hh, network.cc, configrange.hh) — single Delay object, full RTT as one-way

## M-002: Used while-loop for sending
- **What happened**: Sent all window-fitting packets in a while loop per tick
- **Why**: Assumed event-driven sim processes all pending events at a tickno atomically
- **Fix**: Read rat-templates.cc — single `if` send, event loop handles repetition

## Recurring Patterns
| Pattern | Count | Mitigation |
|---------|-------|------------|
| Assumed model from real-network intuition | 1 | Always read the C++ first |
| Assumed atomic event processing | 1 | Event loops can re-enter at same time |
