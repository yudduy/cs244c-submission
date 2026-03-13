# Literature

## Key Papers
### Winstein & Balakrishnan 2013 — TCP ex Machina
- **Finding**: Automated CCA design via whisker trees over EWMA signal space
- **Relevance**: The simulator we're reimplementing
- **Key model detail**: Single bottleneck, one-way delay, no return path, on/off Poisson sources

## Key Implementations
### Remy C++ (armaan-abraham/remy fork)
- **Files**: src/network.cc, sendergang.cc, rat-templates.cc, utility.hh, delay.hh
- **Key behaviors**:
  - Event-driven: advance to min(events), call tick() once per iteration
  - Rat::send sends ONE packet per call (single if, not while)
  - SenderGang shuffles sender order each tick (Fisher-Yates)
  - Delay uses full RTT param as one-way delay (no ACK path)
  - Utility uses tick_received (set by Receiver::accept) for delay accounting
  - score = sum over senders of: log2(tput_norm / link_ppt) - log2(avg_delay / rtt_param)
