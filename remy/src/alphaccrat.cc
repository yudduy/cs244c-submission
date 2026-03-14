#include <algorithm>

#include "alphaccrat.hh"

using namespace std;

WhiskerTree & AlphaCCRat::get_dummy_whiskers()
{
  static WhiskerTree dummy;
  return dummy;
}

AlphaCCRat::AlphaCCRat()
  : Rat( get_dummy_whiskers() ),
    _shared( make_shared<SharedState>() )
{}

AlphaCCRat::AlphaCCRat( const AlphaCCRat & other )
  : Rat( other ),
    _shared( other._shared )
{
}

AlphaCCRat & AlphaCCRat::operator=( const AlphaCCRat & other )
{
  if ( this != &other ) {
    _memory = other._memory;
    _packets_sent = other._packets_sent;
    _packets_received = other._packets_received;
    _track = other._track;
    _last_send_time = other._last_send_time;
    _the_window = other._the_window;
    _intersend_time = other._intersend_time;
    _flow_id = other._flow_id;
    _largest_ack = other._largest_ack;
    _shared = other._shared;
  }
  return *this;
}

void AlphaCCRat::apply_action( int window_increment, double window_multiple, double intersend )
{
  const int raw_window = static_cast<int>( _the_window * window_multiple + window_increment );
  _the_window = min( max( 0, raw_window ), 1000000 );
  _intersend_time = intersend;
}

void AlphaCCRat::update_window_and_intersend()
{
  auto & st = *_shared;

  const double send_ewma = _memory.field( 0 );
  const double rec_ewma = _memory.field( 1 );
  const double rtt_ratio = _memory.field( 2 );
  const double slow_rec_ewma = _memory.field( 3 );

  if ( !st.initialized ) {
    st.initialized = true;
    st.phase = SharedState::Phase::STARTUP;
    st.rtt_base = max( 1.0, rtt_ratio > 0 ? rtt_ratio : 1.0 );
    st.trend = 0.0;
    st.util_ema = 1.0;
    st.last_intersend = 0.0;
    st.cooldown = 0;
  }

  const double send = max( 1e-6, send_ewma > 0 ? send_ewma : 1e-6 );
  const double rec = max( 1e-6, rec_ewma > 0 ? rec_ewma : 1e-6 );
  const double slow_rec = max( rec, slow_rec_ewma > 0 ? slow_rec_ewma : rec );
  const double rtt = max( 1.0, rtt_ratio > 0 ? rtt_ratio : 1.0 );

  const double send_rate = 1.0 / send;
  const double rec_rate = 1.0 / rec;
  const double slow_rec_rate = 1.0 / slow_rec;

  const double util = send / rec;
  st.util_ema = 0.85 * st.util_ema + 0.15 * util;
  const double rtt_grad = rtt - st.rtt_base;
  st.trend = 0.8 * st.trend + 0.2 * rtt_grad;
  st.rtt_base = 0.995 * st.rtt_base + 0.005 * min( rtt, st.rtt_base );

  if ( st.cooldown > 0 ) {
    st.cooldown--;
  }

  if ( st.phase == SharedState::Phase::STARTUP ) {
    if ( rtt > 1.25 || util > 1.20 ) {
      st.phase = SharedState::Phase::DRAIN;
      st.cooldown = 2;
      const double target_rate = max( 1e-6, rec_rate * 0.95 );
      const double intersend = 1.0 / target_rate;
      st.last_intersend = intersend;
      apply_action( 1, 0.92, intersend );
      return;
    }

    const double target_rate = max( send_rate, rec_rate * 1.30 );
    const double intersend = 1.0 / max( 1e-6, target_rate );
    st.last_intersend = intersend;
    apply_action( 3, 1.0, intersend );
    return;
  }

  if ( st.phase == SharedState::Phase::DRAIN ) {
    if ( rtt < 1.12 && util < 1.08 ) {
      st.phase = SharedState::Phase::STEADY;
    }

    const double target_rate = max( 1e-6, slow_rec_rate * 0.92 );
    double intersend = 1.0 / target_rate;
    intersend = 0.7 * st.last_intersend + 0.3 * intersend;
    st.last_intersend = intersend;

    if ( rtt > 2.0 ) {
      apply_action( 0, 0.60, intersend );
      return;
    }

    apply_action( 1, 0.96, intersend );
    return;
  }

  const double congestion = (rtt - 1.0)
    + 0.5 * max( 0.0, st.trend )
    + 0.6 * max( 0.0, st.util_ema - 1.0 );

  if ( rtt > 2.0 ) {
    const double target_rate = max( 1e-6, slow_rec_rate * 0.80 );
    double intersend = 1.0 / target_rate;
    intersend = 0.6 * st.last_intersend + 0.4 * intersend;
    st.last_intersend = intersend;
    st.cooldown = 2;
    apply_action( 0, 0.55, intersend );
    return;
  }

  if ( rtt < 1.3 ) {
    int inc = 1;
    double gain = 1.02;
    if ( congestion < 0.10 ) {
      inc = 3;
      gain = 1.18;
    } else if ( congestion < 0.22 ) {
      inc = 2;
      gain = 1.08;
    }

    const double target_rate = max( 1e-6, rec_rate * gain );
    double intersend = 1.0 / target_rate;
    intersend = 0.75 * st.last_intersend + 0.25 * intersend;
    st.last_intersend = intersend;
    apply_action( inc, 1.0, intersend );
    return;
  }

  const double sev = min( 1.0, max( 0.0, (rtt - 1.3) / 0.7 ) );
  double mult = 1.0 - 0.22 * sev;
  if ( st.cooldown > 0 ) {
    mult = min( mult, 0.95 );
  }

  int inc = -1;
  if ( sev < 0.5 ) {
    inc = 1;
  } else if ( sev < 0.8 ) {
    inc = 0;
  }

  const double gain = 1.0 - 0.18 * sev;
  const double target_rate = max( 1e-6, slow_rec_rate * gain );
  double intersend = 1.0 / target_rate;
  intersend = 0.8 * st.last_intersend + 0.2 * intersend;
  st.last_intersend = intersend;
  apply_action( inc, mult, intersend );
}
