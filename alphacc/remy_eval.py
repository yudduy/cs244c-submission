"""
Remy-compatible evaluation framework for CCA generalization experiments.

Faithful reimplementation of Remy's C++ event-driven simulator.
Validated against C++ sender-runner output (see scripts/calibrate.py).

Calibration: 3.0% mean |error| across 4 whisker trees × 9 link rates.
  - 1x (179 iter): 2.4% mean, 5.6% max
  - 10x (36 iter): 2.7% mean, 7.3% max
  - 20x (19 iter): 3.8% mean, 11.7% max (out-of-range link rate outlier)
  - 100x (published): 3.0% mean, 6.4% max

Key fidelity choices matching C++ (src/network.cc, sendergang.cc, rat-templates.cc):
- Event-driven: advance to min(next_event_times), process all components once per tick
- Sender sends at most ONE packet per tick (not a while loop)
- Per-tick order: senders(receive_feedback + send_one) → link → loss → delay
- Link: service-then-enqueue, service_time = 1/link_ppt
- Delay: full RTT param used as one-way delay (no return path in Remy's model)
- tick_received set at delay delivery, not at sender read time
- Fair-share time: accumulate_sending_time after every send attempt
- Score: log2(throughput/capacity) - log2(delay/rtt_param), summed across senders
- EWMA: alpha=1/8 (fast), alpha=1/256 (slow), rtt_ratio=instantaneous

Reference: "TCP ex Machina" (Winstein & Balakrishnan, SIGCOMM 2013)
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

BYTES_PER_PACKET = 1500


# ── Remy Observation/Action ────────────────────────────────────────

@dataclass
class RemyMemory:
    """Observation state matching Remy's Memory class (memory.hh/cc).

    EWMA signals with exact C++ alpha values.
    rtt_ratio is instantaneous rtt/min_rtt (NOT EWMA'd).
    """
    send_ewma: float = 0.0
    rec_ewma: float = 0.0
    rtt_ratio: float = 0.0
    slow_rec_ewma: float = 0.0
    rtt_diff: float = 0.0
    queueing_delay: float = 0.0
    min_rtt: float = float('inf')
    _last_tick_sent: float = -1.0
    _last_tick_received: float = -1.0
    _is_first: bool = True

    def packets_received(self, packets, flow_id, largest_ack):
        """Process a batch of received packets (matches C++ memory.cc).

        packets: list of (tick_sent, tick_received, seq_num, pkt_flow_id)
        """
        alpha = 1.0 / 8.0
        slow_alpha = 1.0 / 256.0

        for tick_sent, tick_received, seq_num, pkt_flow_id in packets:
            if pkt_flow_id != flow_id:
                continue

            rtt = tick_received - tick_sent
            pkt_outstanding = max(1, seq_num - largest_ack)

            if self._is_first:
                self._last_tick_sent = tick_sent
                self._last_tick_received = tick_received
                self.min_rtt = rtt
                self._is_first = False
            else:
                self.send_ewma = (1 - alpha) * self.send_ewma + alpha * (tick_sent - self._last_tick_sent)
                self.rec_ewma = (1 - alpha) * self.rec_ewma + alpha * (tick_received - self._last_tick_received)
                self.slow_rec_ewma = (1 - slow_alpha) * self.slow_rec_ewma + slow_alpha * (tick_received - self._last_tick_received)

                self._last_tick_sent = tick_sent
                self._last_tick_received = tick_received

                self.min_rtt = min(self.min_rtt, rtt)
                self.rtt_ratio = rtt / self.min_rtt
                self.rtt_diff = rtt - self.min_rtt
                self.queueing_delay = self.rec_ewma * pkt_outstanding

    # Legacy single-packet interface (used by evolve_remy.py policies)
    def packet_sent(self, tick: float):
        pass  # send_ewma is updated from receiver-side tick_sent deltas

    def packet_received(self, tick: float, rtt: float):
        """Single-packet update for backward compatibility."""
        alpha = 1.0 / 8.0
        slow_alpha = 1.0 / 256.0
        if self._is_first:
            self.min_rtt = rtt
            self._last_tick_received = tick
            self._is_first = False
        else:
            if self._last_tick_received >= 0:
                inter_recv = tick - self._last_tick_received
                self.rec_ewma = (1 - alpha) * self.rec_ewma + alpha * inter_recv
                self.slow_rec_ewma = (1 - slow_alpha) * self.slow_rec_ewma + slow_alpha * inter_recv
            self._last_tick_received = tick
            self.min_rtt = min(self.min_rtt, rtt)
            self.rtt_ratio = rtt / self.min_rtt
            self.rtt_diff = rtt - self.min_rtt

    def as_vector(self) -> Tuple[float, float, float, float]:
        return (self.send_ewma, self.rec_ewma, self.rtt_ratio, self.slow_rec_ewma)

    def reset(self):
        self.__init__()


@dataclass
class RemyAction:
    """Action space matching Remy's Whisker output."""
    window_increment: int = 1      # additive change to cwnd
    window_multiple: float = 1.0   # multiplicative change to cwnd
    intersend: float = 0.0         # inter-send time (ms), 0 = no pacing

    def __post_init__(self):
        import math
        if not math.isfinite(self.window_increment):
            self.window_increment = 0
        else:
            self.window_increment = int(self.window_increment)
        if not math.isfinite(self.window_multiple):
            self.window_multiple = 1.0
        if not math.isfinite(self.intersend):
            self.intersend = 0.0


# ── Packet ─────────────────────────────────────────────────────────

@dataclass
class Packet:
    sender_id: int
    flow_id: int
    tick_sent: float
    seq_num: int
    tick_received: float = -1.0  # set by receiver (delay layer delivery time)


# ── Tick-driven simulator matching Remy's Network class ────────────

class RemySimulator:
    """Tick-driven packet simulator faithfully matching Remy's C++ Network.

    Key differences from the old event-based sim:
    - Tick-driven: advances to next event time each step
    - Per-tick order: senders → link → loss → delay → receiver
    - ACKs delivered in tick N are read by senders in tick N+1
    - Link uses service-then-enqueue (not enqueue-then-serve)
    - Delay added AFTER link service, not before
    - Throughput uses fair-share time accounting
    - Score uses log2 (not ln)
    """

    def __init__(
        self,
        policies: List,
        link_ppt: float = 1.0,
        rtt_ms: float = 150.0,
        num_senders: int = 2,
        buffer_pkts: int = None,
        mean_on_ms: float = 5000.0,
        mean_off_ms: float = 5000.0,
        stochastic_loss: float = 0.0,
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.link_rate = link_ppt
        self.delay_ms = rtt_ms  # Remy uses the full RTT param as ONE-WAY delay (no return path)
        self.rtt_ms = rtt_ms
        self.buffer_max = buffer_pkts
        self.mean_on = mean_on_ms
        self.mean_off = mean_off_ms
        self.loss_rate = stochastic_loss

        self.num_senders = num_senders
        self.policies = [policies[i % len(policies)] for i in range(num_senders)]

        # --- Link state (matches link.hh) ---
        self._link_pending: Optional[Tuple[Packet, float]] = None  # (pkt, release_time)
        self._link_buffer: deque = deque()  # FIFO buffer
        self._link_service_time = 1.0 / link_ppt  # ms per packet

        # --- Delay state (matches delay.hh) ---
        self._delay_queue: deque = deque()  # (release_time, packet)

        # --- Receiver state ---
        self._receiver: List[List[Packet]] = [[] for _ in range(num_senders)]

        # --- Per-sender state (matches rat.hh) ---
        self._memory = [RemyMemory() for _ in range(num_senders)]
        self._window = [0] * num_senders
        self._intersend_time = [0.0] * num_senders
        self._packets_sent = [0] * num_senders
        self._largest_ack = [-1] * num_senders
        self._flow_id = [0] * num_senders
        self._last_send_time = [0.0] * num_senders

        # --- On/off state (matches sendergang.cc) ---
        self._is_sending = [False] * num_senders
        self._next_switch_tick = [0.0] * num_senders
        self._internal_tick = [0.0] * num_senders

        # --- Utility accounting (matches utility.hh) ---
        self._tick_share_sending = [0.0] * num_senders
        self._pkts_received = [0] * num_senders
        self._total_delay = [0.0] * num_senders

        # Initialize on/off schedule
        self.prelaunch = (mean_on_ms >= 1e8)
        for i in range(num_senders):
            if self.prelaunch:
                # Pantheon-style: all senders start ON, stay on forever
                self._is_sending[i] = True
                self._next_switch_tick[i] = float('inf')
                self._sender_reset(i, 0.0)
                self._internal_tick[i] = 0.0
            else:
                self._next_switch_tick[i] = self.rng.expovariate(1.0 / max(1.0, self.mean_off))

    def run(self, duration_ms: float):
        """Event loop matching C++ Network::run_simulation().

        C++ advances to min(all next_event_times), then calls tick() once.
        Each tick: senders(read_acks + send_one_each) → link → loss → delay.
        The loop can run multiple times at the same tickno if senders need
        to send multiple packets (each gets one send per tick iteration).
        """
        tickno = 0.0
        while tickno < duration_ms:
            # Find next event time (>= tickno, matching C++ which uses min directly)
            next_tick = self._next_event_time(tickno)

            if next_tick > duration_ms:
                break

            tickno = next_tick

            # tick(): senders(read+send_one) → link → loss → delay
            self._tick(tickno, duration_ms)

    def _next_event_time(self, tickno):
        """Find earliest event time >= tickno (matches C++ run_simulation)."""
        result = float('inf')
        for i in range(self.num_senders):
            t = self._next_switch_tick[i]
            if t >= tickno:
                result = min(result, t)
            t = self._sender_next_event(i)
            if t >= tickno:
                result = min(result, t)
        if self._link_pending is not None:
            t = self._link_pending[1]
            if t >= tickno:
                result = min(result, t)
        if self._delay_queue:
            t = self._delay_queue[0][0]
            if t >= tickno:
                result = min(result, t)
        # Receiver: if any sender has packets waiting, process now
        # (matches C++ Receiver::next_event_time)
        for i in range(self.num_senders):
            if self._receiver[i]:
                result = min(result, tickno)
                break
        return result

    def _tick(self, tickno, duration_ms):
        """One tick: senders(read+send_one) → link → delay. Matches C++ tick()."""
        self._senders_tick(tickno, duration_ms)
        self._link_tick(tickno)
        self._delay_tick(tickno)

    def _sender_next_event(self, sid):
        """Next time this sender wants to act."""
        if not self._is_sending[sid]:
            return float('inf')
        # Window check: can only send if window has room
        if self._packets_sent[sid] >= self._largest_ack[sid] + 1 + self._window[sid]:
            return float('inf')  # window full, wait for ACKs
        # Pacing constraint
        next_send = self._last_send_time[sid] + self._intersend_time[sid]
        return next_send

    def _senders_tick(self, tickno, duration_ms):
        """Handle on/off switching, ACK reading, and ONE send per sender.

        Matches C++ SenderGang::tick():
        1. switch_senders(num_sending, tickno) — handle on/off
        2. run_senders(next, rec, num_sending, tickno) — for each sender:
           a. receive_feedback(rec) — read packets from receiver
           b. sender.send() — send at most ONE packet
        """
        num_sending = sum(1 for s in self._is_sending if s)

        # Phase 1: switch_senders (on/off switching)
        for sid in range(self.num_senders):
            while self._next_switch_tick[sid] <= tickno:
                if self._is_sending[sid]:
                    self._accumulate_sending_time(sid, tickno, num_sending)
                    self._is_sending[sid] = False
                    num_sending = sum(1 for s in self._is_sending if s)
                    self._next_switch_tick[sid] += self.rng.expovariate(1.0 / max(1.0, self.mean_off))
                else:
                    self._is_sending[sid] = True
                    num_sending = sum(1 for s in self._is_sending if s)
                    self._sender_reset(sid, tickno)
                    self._internal_tick[sid] = tickno
                    self._next_switch_tick[sid] += self.rng.expovariate(1.0 / max(1.0, self.mean_on))

        num_sending = sum(1 for s in self._is_sending if s)

        # Phase 2: run_senders — read ACKs + send one packet each
        # C++ shuffles sender order (Fisher-Yates); we use sequential for determinism
        for sid in range(self.num_senders):
            # receive_feedback: read packets from receiver
            self._sender_receive_feedback(sid, tickno, num_sending)
            # send at most ONE packet, then accumulate sending time
            # (matches C++ TimeSwitchedSender::tick: send then accumulate_sending_time_until)
            if self._is_sending[sid]:
                self._sender_send_one(sid, tickno)
                self._accumulate_sending_time(sid, tickno, num_sending)

    def _sender_reset(self, sid, tickno):
        """Reset sender state (matches rat.cc reset)."""
        self._memory[sid].reset()
        self._last_send_time[sid] = 0.0
        self._window[sid] = 0
        self._intersend_time[sid] = 0.0
        self._flow_id[sid] += 1
        self._largest_ack[sid] = self._packets_sent[sid] - 1
        # Immediately query policy for initial window (matches C++)
        action = self.policies[sid](self._memory[sid])
        self._apply_action(sid, action)

    def _apply_action(self, sid, action):
        """Apply whisker action to sender state (matches whisker.hh)."""
        import math
        inc = action.window_increment if math.isfinite(action.window_increment) else 0
        mult = action.window_multiple if math.isfinite(action.window_multiple) else 1.0
        raw = self._window[sid] * mult + inc
        new_window = int(min(max(raw, 0), 1000000))
        self._window[sid] = new_window
        intersend = action.intersend if math.isfinite(action.intersend) else 0.0
        self._intersend_time[sid] = max(0.0, intersend)

    def _sender_send_one(self, sid, tickno):
        """Sender sends at most ONE packet (matches rat-templates.cc Rat::send).

        C++ sends exactly one packet per tick call — no while loop.
        The outer event loop handles multiple sends at the same tickno.
        """
        # If window is 0, query policy first (matches C++ Rat::send)
        if self._window[sid] == 0:
            action = self.policies[sid](self._memory[sid])
            self._apply_action(sid, action)

        # Check window: packets_sent < largest_ack + 1 + window
        if (self._packets_sent[sid] < self._largest_ack[sid] + 1 + self._window[sid]
                and self._last_send_time[sid] + self._intersend_time[sid] <= tickno):

            pkt = Packet(
                sender_id=sid,
                flow_id=self._flow_id[sid],
                tick_sent=tickno,
                seq_num=self._packets_sent[sid],
            )
            self._packets_sent[sid] += 1
            self._last_send_time[sid] = tickno

            # Enter link
            self._link_accept(pkt, tickno)

    def _link_accept(self, pkt, tickno):
        """Link accepts a packet (matches link.hh accept)."""
        if self._link_pending is None:
            # Link idle — start service immediately
            release_time = tickno + self._link_service_time
            self._link_pending = (pkt, release_time)
        elif self.buffer_max is None or len(self._link_buffer) < self.buffer_max:
            # Buffer has room
            self._link_buffer.append(pkt)
        else:
            # Buffer full — drop (packet is lost)
            pass

    def _link_tick(self, tickno):
        """Link services packets (matches link-templates.cc tick)."""
        while self._link_pending is not None and self._link_pending[1] <= tickno:
            pkt = self._link_pending[0]
            # Deliver to loss layer / delay
            self._link_deliver(pkt, tickno)
            # Dequeue next from buffer
            if self._link_buffer:
                next_pkt = self._link_buffer.popleft()
                release_time = tickno + self._link_service_time
                self._link_pending = (next_pkt, release_time)
            else:
                self._link_pending = None

    def _link_deliver(self, pkt, tickno):
        """Deliver packet from link to loss/delay layers."""
        # Stochastic loss
        if self.loss_rate > 0 and self.rng.random() < self.loss_rate:
            return  # dropped
        # Enter delay layer
        release_time = tickno + self.delay_ms
        self._delay_queue.append((release_time, pkt))

    def _delay_tick(self, tickno):
        """Delay layer releases packets to receiver (matches delay.hh tick).

        Just delivers to receiver buffer. Utility accounting happens in
        _sender_receive_feedback (matching C++ where utility.packets_received
        is called during sender tick, not delay tick).
        """
        while self._delay_queue and self._delay_queue[0][0] <= tickno:
            _, pkt = self._delay_queue.popleft()
            pkt.tick_received = tickno  # matches C++ Receiver::accept
            self._receiver[pkt.sender_id].append(pkt)

    def _sender_receive_feedback(self, sid, tickno, num_sending):
        """Read packets from receiver for one sender (matches receive_feedback + packets_received).

        C++ order in SwitchedSender::tick():
        1. receive_feedback(rec) — read packets, update utility + sender memory
        2. sender.send() — send one packet

        In receive_feedback: utility.packets_received(packets) then sender.packets_received(packets).
        In Rat::packets_received: updates memory, queries whisker, applies action.
        """
        packets = self._receiver[sid]
        if not packets:
            return

        # utility.packets_received — delay accounting (uses tick_received, not tickno)
        for pkt in packets:
            self._pkts_received[pkt.sender_id] += 1
            self._total_delay[pkt.sender_id] += (pkt.tick_received - pkt.tick_sent)

        # sender.packets_received — update memory, query policy
        if self._is_sending[sid]:
            self._accumulate_sending_time(sid, tickno, num_sending)

        # Build packet list for batch memory update
        pkt_data = []
        for pkt in packets:
            pkt_data.append((pkt.tick_sent, pkt.tick_received, pkt.seq_num, pkt.flow_id))
            if pkt.seq_num > self._largest_ack[sid]:
                self._largest_ack[sid] = pkt.seq_num

        # Batch update memory (matches C++ memory.cc)
        self._memory[sid].packets_received(
            pkt_data, self._flow_id[sid], self._largest_ack[sid])

        # Query policy and apply action (matches C++ Rat::packets_received)
        action = self.policies[sid](self._memory[sid])
        self._apply_action(sid, action)

        # Clear receiver buffer
        self._receiver[sid] = []

    def _accumulate_sending_time(self, sid, tickno, num_sending):
        """Accumulate fair-share sending time (matches sendergang.cc)."""
        if num_sending > 0 and tickno > self._internal_tick[sid]:
            duration = tickno - self._internal_tick[sid]
            self._tick_share_sending[sid] += duration / num_sending
            self._internal_tick[sid] = tickno

    def results(self, duration_ms: float) -> Dict:
        """Compute results matching C++ sender-runner output."""
        per_sender = []
        total_score = 0.0

        for i in range(self.num_senders):
            if self._pkts_received[i] > 0 and self._tick_share_sending[i] > 0:
                tput_norm = self._pkts_received[i] / self._tick_share_sending[i]
                avg_delay = self._total_delay[i] / self._pkts_received[i]
                # Per-sender score (matches sender-runner.cc)
                score = math.log2(tput_norm / self.link_rate) - math.log2(avg_delay / self.rtt_ms)
            else:
                tput_norm = 0.0
                avg_delay = float('inf')
                score = -100.0

            tput = self._pkts_received[i] / duration_ms if duration_ms > 0 else 0
            total_score += score

            per_sender.append({
                'id': i,
                'pkts_sent': self._packets_sent[i],
                'pkts_acked': self._pkts_received[i],
                'pkts_lost': 0,
                'throughput_ppt': tput,
                'throughput_normalized': tput_norm,
                'avg_delay_ms': avg_delay,
                'utility': score,
                'on_time_ms': self._tick_share_sending[i],
            })

        total_acked = sum(s['pkts_acked'] for s in per_sender)
        total_delay_sum = sum(self._total_delay)
        total_tput = sum(s['throughput_ppt'] for s in per_sender)
        avg_delay = total_delay_sum / total_acked if total_acked > 0 else float('inf')

        return {
            'throughput_ppt': total_tput,
            'avg_delay_ms': avg_delay,
            'total_utility': total_score,
            'normalized_score': total_score / self.num_senders,
            'per_sender': per_sender,
            'link_ppt': self.link_rate,
            'rtt_ms': self.rtt_ms,
            'num_senders': self.num_senders,
        }


# ── High-level API ──────────────────────────────────────────────────

def run_remy_sim(
    policies: List,
    link_ppt: float = 1.0,
    rtt_ms: float = 150.0,
    num_senders: int = 2,
    buffer_pkts: int = None,
    duration_ms: float = 30_000.0,
    mean_on_ms: float = 1000.0,
    mean_off_ms: float = 1000.0,
    delta: float = 1.0,
    seed: int = 42,
    stochastic_loss: float = 0.0,
) -> Dict:
    """Run Remy-style simulation with multiple on/off senders."""
    sim = RemySimulator(
        policies=policies,
        link_ppt=link_ppt,
        rtt_ms=rtt_ms,
        num_senders=num_senders,
        buffer_pkts=buffer_pkts,
        mean_on_ms=mean_on_ms,
        mean_off_ms=mean_off_ms,
        stochastic_loss=stochastic_loss,
        seed=seed,
    )
    sim.run(duration_ms)
    return sim.results(duration_ms)


# ── ConfigRange and Evaluation ──────────────────────────────────────

@dataclass
class ConfigRange:
    """Remy-style config range for training/evaluation."""
    link_ppt_range: List[float] = field(default_factory=lambda: [1.0])
    rtt_range: List[float] = field(default_factory=lambda: [150.0])
    nsrc_range: List[int] = field(default_factory=lambda: [2])
    buffer_range: List[Optional[int]] = field(default_factory=lambda: [None])
    mean_on_ms: float = 5000.0
    mean_off_ms: float = 5000.0
    duration_ms: float = 100_000.0
    stochastic_loss: float = 0.0
    delta: float = 1.0

    def configs(self) -> List[Dict]:
        """Generate Cartesian product of all parameter ranges."""
        cfgs = []
        for link in self.link_ppt_range:
            for rtt in self.rtt_range:
                for nsrc in self.nsrc_range:
                    for buf in self.buffer_range:
                        cfgs.append({
                            'link_ppt': link,
                            'rtt_ms': rtt,
                            'num_senders': nsrc,
                            'buffer_pkts': buf,
                            'mean_on_ms': self.mean_on_ms,
                            'mean_off_ms': self.mean_off_ms,
                            'duration_ms': self.duration_ms,
                            'stochastic_loss': self.stochastic_loss,
                            'delta': self.delta,
                        })
        return cfgs


def make_training_range(multiplier: float = 1.0, steps: int = 5) -> ConfigRange:
    """Create a Remy-style training ConfigRange.

    The multiplier controls the breadth of link rates:
    - 1x: single point (link=1 ppt, RTT=150ms)
    - 10x: link ∈ [0.316, 3.16] ppt (log-spaced)
    - 100x: link ∈ [0.1, 10.0] ppt

    Matches Remy 2013/2014 paper setup.
    """
    if multiplier <= 1.0:
        links = [1.0]
    else:
        lo = 1.0 / math.sqrt(multiplier)
        hi = 1.0 * math.sqrt(multiplier)
        links = [lo * (hi / lo) ** (i / (steps - 1)) for i in range(steps)]

    return ConfigRange(
        link_ppt_range=links,
        rtt_range=[150.0],
        nsrc_range=[2],
        buffer_range=[None],  # infinite buffers (Remy default)
        mean_on_ms=5000.0,
        mean_off_ms=5000.0,
        duration_ms=100_000.0,  # 100s
        delta=1.0,
    )


def evaluate_policy(
    policy,
    config_range: ConfigRange,
    num_trials: int = 1,
    seed: int = 42,
) -> Dict:
    """Evaluate a CCA policy across a ConfigRange.

    Returns aggregate metrics including mean utility and normalized score.
    """
    configs = config_range.configs()
    all_results = []

    for trial in range(num_trials):
        for i, cfg in enumerate(configs):
            result = run_remy_sim(
                policies=[policy],
                link_ppt=cfg['link_ppt'],
                rtt_ms=cfg['rtt_ms'],
                num_senders=cfg['num_senders'],
                buffer_pkts=cfg['buffer_pkts'],
                duration_ms=cfg['duration_ms'],
                mean_on_ms=cfg['mean_on_ms'],
                mean_off_ms=cfg['mean_off_ms'],
                delta=cfg['delta'],
                seed=seed + trial * 1000 + i,
                stochastic_loss=cfg.get('stochastic_loss', 0.0),
            )
            all_results.append(result)

    if not all_results:
        return {'mean_utility': -100, 'mean_normalized': -100, 'results': []}

    mean_utility = sum(r['total_utility'] for r in all_results) / len(all_results)
    mean_normalized = sum(r['normalized_score'] for r in all_results) / len(all_results)
    mean_throughput = sum(r['throughput_ppt'] for r in all_results) / len(all_results)
    mean_delay = sum(r['avg_delay_ms'] for r in all_results) / len(all_results)

    return {
        'mean_utility': mean_utility,
        'mean_normalized': mean_normalized,
        'mean_throughput_ppt': mean_throughput,
        'mean_delay_ms': mean_delay,
        'num_configs': len(configs),
        'num_trials': num_trials,
        'results': all_results,
    }


# ── Standard CCA Policies ──────────────────────────────────────────

def aimd_policy(memory: RemyMemory) -> RemyAction:
    """Standard AIMD with proper congestion event detection.

    Additive increase by 1 per ACK. Multiplicative decrease (halve) on
    congestion — but at most once per RTT (matching real TCP behavior).
    Without cooldown, halving on every ACK collapses cwnd exponentially.
    """
    if not hasattr(aimd_policy, '_states'):
        aimd_policy._states = {}
    mid = id(memory)
    if mid not in aimd_policy._states:
        aimd_policy._states[mid] = {'last_halve_tick': -1e9}
    st = aimd_policy._states[mid]

    # Congestion signal: rtt_ratio > 1.5 (queuing delay = 50% of min_rtt)
    if memory.rtt_ratio > 1.5:
        # Only halve if we haven't halved recently (cooldown = ~1 RTT)
        # Use min_rtt as RTT estimate; rec_ewma as time proxy
        cooldown = max(1.0, memory.min_rtt) if memory.min_rtt < float('inf') else 150.0
        tick_now = memory._last_tick_received if memory._last_tick_received > 0 else 0
        if tick_now - st['last_halve_tick'] >= cooldown:
            st['last_halve_tick'] = tick_now
            return RemyAction(window_increment=0, window_multiple=0.5, intersend=0.0)
        # In cooldown — hold steady
        return RemyAction(window_increment=0, window_multiple=1.0, intersend=0.0)
    return RemyAction(window_increment=1, window_multiple=1.0, intersend=0.0)


def copa_policy(memory: RemyMemory) -> RemyAction:
    """Copa (NSDI 2018) ported from venkatarun95/genericCC (markoviancc.cc).

    Core logic:
    - target_window = rtt / (queuing_delay * delta)
    - Per RTT: if cwnd < target → update_dir++, else update_dir--
    - Per RTT: cwnd += update_amt / (delta * cwnd)  [or -= ]
    - Velocity: update_amt doubles when direction consistent, resets to 1 on reversal
    - Direction decision and cwnd adjustment gated to once per RTT
    - intersend (pacing) updated every ACK: cwnd / rtt

    Delta=0.5 (default from Copa paper).
    """
    if not hasattr(copa_policy, '_states'):
        copa_policy._states = {}
    mid = id(memory)
    if mid not in copa_policy._states:
        copa_policy._states[mid] = {
            'the_window': 4.0,
            'update_dir': 0,
            'prev_update_dir': 0,
            'update_amt': 1.0,
            'last_update_tick': -1e9,
            'intersend': 0.0,
        }
    st = copa_policy._states[mid]
    delta = 0.5

    # Bootstrap
    if memory._is_first:
        st['the_window'] = 4.0
        st['update_dir'] = 0
        st['prev_update_dir'] = 0
        st['update_amt'] = 1.0
        st['last_update_tick'] = -1e9
        st['intersend'] = 0.0
        return RemyAction(window_increment=4, window_multiple=1.0, intersend=0.0)

    min_rtt = memory.min_rtt if memory.min_rtt < float('inf') else 150.0
    cur_rtt = min_rtt * memory.rtt_ratio if memory.rtt_ratio > 0 else min_rtt
    queuing_delay = max(0.0, cur_rtt - min_rtt)
    tick_now = memory._last_tick_received

    # Gate direction + cwnd update to once per RTT
    if tick_now - st['last_update_tick'] >= cur_rtt:
        st['last_update_tick'] = tick_now

        # Copa target: target_window = rtt / (queuing_delay * delta)
        if queuing_delay > 0.01:
            target_window = cur_rtt / (queuing_delay * delta)
        else:
            target_window = float('inf')  # no queuing → always increase

        if st['the_window'] < target_window:
            st['update_dir'] += 1
            st['the_window'] += st['update_amt'] / (delta * st['the_window'])
        else:
            st['update_dir'] -= 1
            st['the_window'] = max(2.0, st['the_window'] - st['update_amt'] / (delta * st['the_window']))

        # Velocity: double if same direction, reset on reversal
        # Cap at 16 to prevent overshoot (Copa paper mentions bounded velocity)
        if st['prev_update_dir'] * st['update_dir'] > 0:
            if st['update_amt'] < 0.006:
                st['update_amt'] += 0.005
            else:
                st['update_amt'] = min(16, int(st['update_amt']) * 2)
        else:
            st['update_amt'] = 1.0
            if st['update_dir'] != 0:
                st['prev_update_dir'] = 1 if st['update_dir'] > 0 else -1

    # Pacing: intersend = rtt / cwnd (updated every ACK)
    if st['the_window'] > 0:
        st['intersend'] = cur_rtt / st['the_window']
    else:
        st['intersend'] = cur_rtt

    # Set cwnd directly using mult=0, inc=target (bypasses Remy's incremental model)
    target_int = max(1, round(st['the_window']))
    return RemyAction(window_increment=target_int, window_multiple=0.0, intersend=st['intersend'])


def bbr_policy(memory: RemyMemory) -> RemyAction:
    """BBR-inspired: estimate BW from delivery rate, pace at BtlBw, cwnd = BDP.

    Adapted for Remy's framework: uses rec_ewma for BW estimation,
    min_rtt for RTprop, and Copa's cwnd-setting trick (mult=0, inc=target).
    Conservative cwnd (1*BDP) to avoid queue inflation in shared links.

    Phases: Startup → Drain → ProbeBW (8-phase gain cycling).
    Includes periodic ProbeRTT to get accurate min_rtt.
    """
    if not hasattr(bbr_policy, '_states'):
        bbr_policy._states = {}
    mid = id(memory)
    if mid not in bbr_policy._states:
        bbr_policy._states[mid] = {
            'phase': 'startup',
            'btl_bw': 0.0,        # windowed-max delivery rate (pkts/ms)
            'bw_filter': [],       # sliding window of bw samples
            'ack_count': 0,
            'probe_cycle': 0,
            'filled_pipe': False,
            'full_bw': 0.0,
            'full_bw_count': 0,
            'last_cycle_tick': -1.0,
            'probe_rtt_timer': 0,  # ACKs until next ProbeRTT
        }
    st = bbr_policy._states[mid]
    st['ack_count'] += 1
    st['probe_rtt_timer'] += 1

    # Bootstrap
    if memory._is_first or memory.rec_ewma < 0.01:
        return RemyAction(window_increment=2, window_multiple=1.0, intersend=0.0)

    # --- BW estimation: windowed max of delivery rate ---
    cur_bw = 1.0 / memory.rec_ewma
    rtt_ms = memory.min_rtt if memory.min_rtt < float('inf') else 150.0

    # Sample BW once per RTT worth of ACKs
    acks_per_rtt = max(1, int(rtt_ms / max(0.01, memory.rec_ewma)))
    if st['ack_count'] >= acks_per_rtt:
        st['bw_filter'].append(cur_bw)
        if len(st['bw_filter']) > 10:
            st['bw_filter'] = st['bw_filter'][-10:]
        st['ack_count'] = 0

    # Use slow_rec_ewma for more stable BW estimate (avoids burst spikes)
    if memory.slow_rec_ewma > 0.01:
        stable_bw = 1.0 / memory.slow_rec_ewma
    else:
        stable_bw = cur_bw

    # BtlBw = max of windowed filter, but capped by slow estimate * 1.5
    # This prevents inflation from startup bursts
    raw_max = max(st['bw_filter']) if st['bw_filter'] else cur_bw
    st['btl_bw'] = min(raw_max, stable_bw * 1.5)

    bdp = max(2, st['btl_bw'] * rtt_ms)

    # --- STARTUP: grow until pipe full ---
    if st['phase'] == 'startup':
        if cur_bw > st['full_bw'] * 1.25:
            st['full_bw'] = cur_bw
            st['full_bw_count'] = 0
        else:
            st['full_bw_count'] += 1

        if st['full_bw_count'] >= 3 or memory.rtt_ratio > 1.5:
            st['filled_pipe'] = True
            st['phase'] = 'drain'
        else:
            target_cwnd = max(4, int(bdp))
            intersend = 1.0 / max(0.001, st['btl_bw'] * 1.5)
            return RemyAction(window_increment=target_cwnd, window_multiple=0.0,
                              intersend=intersend)

    # --- DRAIN: reduce queue ---
    if st['phase'] == 'drain':
        if memory.rtt_ratio < 1.1:
            st['phase'] = 'probebw'
            st['probe_cycle'] = 0
            st['probe_rtt_timer'] = 0

        target_cwnd = max(2, int(bdp * 0.75))
        intersend = 1.0 / max(0.001, st['btl_bw'] * 0.5)
        return RemyAction(window_increment=target_cwnd, window_multiple=0.0,
                          intersend=intersend)

    # --- ProbeRTT: every ~200 RTTs, reduce cwnd to drain queue ---
    probe_rtt_interval = acks_per_rtt * 200
    if st['probe_rtt_timer'] >= probe_rtt_interval:
        st['probe_rtt_timer'] = 0
        # 4 packets for 1 RTT to let queue drain
        return RemyAction(window_increment=4, window_multiple=0.0, intersend=0.0)

    # --- ProbeBW: 8-phase pacing gain cycle ---
    pacing_gains = [1.1, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    gain = pacing_gains[st['probe_cycle'] % 8]

    tick_now = memory._last_tick_received if memory._last_tick_received > 0 else 0
    if tick_now - st['last_cycle_tick'] >= rtt_ms:
        st['last_cycle_tick'] = tick_now
        st['probe_cycle'] = (st['probe_cycle'] + 1) % 8
        gain = pacing_gains[st['probe_cycle']]

    # cwnd = 1 * BDP (conservative; real BBR uses 2x but that queues in shared sim)
    target_cwnd = max(4, int(bdp))
    pacing_rate = gain * st['btl_bw']
    intersend = 1.0 / max(0.001, pacing_rate)

    return RemyAction(window_increment=target_cwnd, window_multiple=0.0,
                      intersend=intersend)


def constant_policy(memory: RemyMemory) -> RemyAction:
    """Constant window — no adaptation (for baseline comparison)."""
    return RemyAction(window_increment=0, window_multiple=1.0, intersend=0.0)


# ── Generalization Experiment ───────────────────────────────────────

def run_generalization_experiment(
    policies: Dict[str, callable],
    test_multipliers: List[float] = None,
    num_trials: int = 1,
    seed: int = 42,
    duration_ms: float = 50_000.0,
    verbose: bool = True,
) -> Dict:
    """Run the Remy generalization experiment.

    Evaluates each policy across a range of test multipliers.
    Returns throughput-delay pairs for plotting.
    """
    if test_multipliers is None:
        test_multipliers = [1, 2, 5, 10, 20, 50, 100]

    results = {}
    for name, policy in policies.items():
        results[name] = []
        for mult in test_multipliers:
            cfg = make_training_range(mult)
            cfg.duration_ms = duration_ms  # faster for experiments
            if verbose:
                print(f"  Evaluating {name} on {mult}x range ({len(cfg.configs())} configs)...")
            eval_result = evaluate_policy(policy, cfg, num_trials=num_trials, seed=seed)
            results[name].append({
                'multiplier': mult,
                'throughput': eval_result['mean_throughput_ppt'],
                'delay': eval_result['mean_delay_ms'],
                'utility': eval_result['mean_utility'],
                'normalized': eval_result['mean_normalized'],
            })
            if verbose:
                print(f"    utility={eval_result['mean_utility']:.2f} "
                      f"tput={eval_result['mean_throughput_ppt']:.4f} ppt "
                      f"delay={eval_result['mean_delay_ms']:.1f}ms "
                      f"norm={eval_result['mean_normalized']:.3f}")

    return results


# ── Quick smoke test ────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    print("=== Remy Evaluation Framework Smoke Test ===\n")

    print("Single config test (AIMD, 10s):")
    t0 = time.time()
    result = run_remy_sim(
        policies=[aimd_policy],
        link_ppt=1.0,
        rtt_ms=150.0,
        num_senders=2,
        duration_ms=10_000.0,
        seed=42,
    )
    elapsed = time.time() - t0
    print(f"  throughput={result['throughput_ppt']:.4f} ppt "
          f"(utilization={result['throughput_ppt']/result['link_ppt']:.1%}), "
          f"delay={result['avg_delay_ms']:.1f}ms, "
          f"utility={result['total_utility']:.2f}, "
          f"time={elapsed:.1f}s")
    for s in result['per_sender']:
        print(f"  sender {s['id']}: sent={s['pkts_sent']}, acked={s['pkts_acked']}, "
              f"lost={s['pkts_lost']}, tput={s['throughput_ppt']:.4f}, "
              f"on_time={s['on_time_ms']:.0f}ms")

    print("\nBaseline comparison (10s, single config):")
    for name, policy in [('AIMD', aimd_policy), ('Copa', copa_policy), ('Constant', constant_policy)]:
        result = run_remy_sim(
            policies=[policy],
            link_ppt=1.0,
            rtt_ms=150.0,
            num_senders=2,
            duration_ms=10_000.0,
            seed=42,
        )
        print(f"  {name:12s}: util={result['throughput_ppt']/result['link_ppt']:5.1%} "
              f"delay={result['avg_delay_ms']:6.1f}ms "
              f"utility={result['total_utility']:7.2f}")

    print("\nGeneralization experiment (quick, 20s sims):")
    t0 = time.time()
    gen_results = run_generalization_experiment(
        policies={'AIMD': aimd_policy, 'Copa': copa_policy},
        test_multipliers=[1, 5, 10],
        num_trials=1,
        seed=42,
        duration_ms=20_000.0,
    )
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
