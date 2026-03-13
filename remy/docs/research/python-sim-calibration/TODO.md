# TODO

## Done
- [x] Fix delay parameter (rtt_ms, not rtt_ms/2)
- [x] Restructure event loop (send-one-per-tick)
- [x] Fix ACK timing (tick_received on delivery)
- [x] Add receiver event source
- [x] Fix sending-duration accounting
- [x] Codex review
- [x] Full 4-tree calibration

## Optional (diminishing returns)
- [ ] Add Fisher-Yates shuffle for sender ordering (C++ shuffles each tick)
- [ ] Match C++ PRNG (mt19937 with same seed) for exact bitwise matching
- [ ] Run calibration on GCP with C++ sender-runner for live comparison
