/* EvolvedRat: C++ translation of v5_best_policy.py (AlphaCC single-point best).
   Stateful CCA with BDP-scaled pacing, startup phase, and 5 rtt_ratio regimes. */

#include <cmath>
#include <algorithm>

#include "evolvedrat.hh"

using namespace std;

WhiskerTree & EvolvedRat::get_dummy_whiskers()
{
  static WhiskerTree dummy;
  return dummy;
}

EvolvedRat::EvolvedRat( EvolvedPolicy & /* policy */ )
  : Rat( get_dummy_whiskers() ),
    _max_bw( 0.0 ),
    _eq_bw( 0.0 ),
    _startup( true ),
    _startup_rounds( 0 ),
    _last_min_rtt( 0.0 )
{
}

EvolvedRat::EvolvedRat( const EvolvedRat & other )
  : Rat( other ),
    _max_bw( other._max_bw ),
    _eq_bw( other._eq_bw ),
    _startup( other._startup ),
    _startup_rounds( other._startup_rounds ),
    _last_min_rtt( other._last_min_rtt )
{
}

void EvolvedRat::update_window_and_intersend()
{
  /* Before EWMAs are initialized (takes 2+ ACKs), Memory EWMA fields are zero.
     The Python simulator only calls the policy after EWMAs are populated.
     C++ calls update_window_and_intersend() on reset() and on early ACKs
     before EWMAs have real values. Skip policy logic to avoid 1/0 → ∞.
     Use window=1, intersend=1ms (1 pkt/ms) — slightly faster than Remy's
     default of 3ms but conservative enough to avoid queue flooding. */
  if ( _memory.field(1) <= 0 ) {  /* rec_rec_ewma not yet populated */
    _the_window = max( 1, _the_window );
    if ( _intersend_time <= 0 ) _intersend_time = 1.0;
    return;
  }

  /* Read Memory fields (same indices as Python: send_ewma=0, rec_rec_ewma=1,
     rtt_ratio=2, slow_rec_rec_ewma=3). min_rtt via getter. */
  const double send_i   = max( _memory.field(0), 1e-6 );
  const double rec_i    = _memory.field(1) > 1e-6 ? _memory.field(1) : send_i;
  const double slow_rec_i = _memory.field(3) > 1e-6 ? _memory.field(3) : rec_i;

  const double recv_bw  = 1.0 / rec_i;      /* pkts/ms */
  const double slow_bw  = 1.0 / slow_rec_i;
  /* send_bw unused in non-startup paths but computed for completeness */

  /* Update persistent bandwidth estimates */
  _max_bw = max( _max_bw * 0.995, recv_bw );
  if ( _eq_bw <= 0.0 ) {
    _eq_bw = recv_bw;
  } else {
    _eq_bw = 0.9 * _eq_bw + 0.1 * recv_bw;
  }

  const double bw_ref   = max( { _eq_bw, slow_bw, 1e-6 } );
  const double min_rtt  = max( _memory.min_rtt(), 1e-3 );
  const double bdp_est  = max( 1.0, bw_ref * min_rtt );
  const double scale    = max( 1.0, min( 16.0, sqrt( bdp_est ) ) );

  const double q = _memory.field(2);  /* rtt_ratio */

  int    window_increment = 0;
  double window_multiple  = 1.0;
  double intersend        = 0.0;

  if ( _startup ) {
    _startup_rounds++;
    if ( q > 1.25 || recv_bw < 0.92 * _max_bw || _startup_rounds > 80 ) {
      _startup = false;
      /* Fall through to steady-state logic below */
    } else {
      const double pacing_gain = ( q < 1.10 ) ? 1.20 : 1.08;
      const double target_bw = max( recv_bw, slow_bw ) * pacing_gain;
      intersend = 1.0 / max( target_bw, 1e-6 );
      window_increment = max( 2, (int)ceil( scale * 0.9 ) );
      window_multiple = 1.0;

      _the_window = min( max( 0, (int)( _the_window * window_multiple + window_increment ) ), 1000000 );
      _intersend_time = intersend;
      return;
    }
  }

  /* Steady-state: 5 regimes based on rtt_ratio */
  if ( q > 2.0 ) {
    const double target_bw = max( 0.55 * slow_bw, max( 0.45 * recv_bw, 1e-6 ) );
    intersend = 1.0 / target_bw;
    window_increment = 0;
    window_multiple = 0.70;
  } else if ( q > 1.5 ) {
    const double pacing_gain = 0.92;
    const double target_bw = max( min( recv_bw, slow_bw ) * pacing_gain, 1e-6 );
    intersend = 1.0 / target_bw;
    window_increment = 0;
    window_multiple = 0.92;
  } else if ( q < 1.3 ) {
    double pacing_gain;
    if ( recv_bw >= 0.97 * slow_bw ) {
      pacing_gain = 1.06;
      window_increment = max( 1, (int)ceil( 0.35 * scale ) );
    } else {
      pacing_gain = 1.01;
      window_increment = max( 1, (int)ceil( 0.18 * scale ) );
    }
    const double target_bw = max( recv_bw, slow_bw ) * pacing_gain;
    intersend = 1.0 / max( target_bw, 1e-6 );
    window_multiple = 1.0;
  } else {
    /* Transitional: 1.3 <= q <= 2.0 (note: q <= 1.5 already handled above) */
    const double pacing_gain = ( q > 1.4 ) ? 0.98 : 1.0;
    const double target_bw = max( min( recv_bw, slow_bw ) * pacing_gain, 1e-6 );
    intersend = 1.0 / target_bw;
    window_increment = 1;
    window_multiple = 1.0;
  }

  /* Apply window formula (same as Rat::update_window_and_intersend) */
  _the_window = min( max( 0, (int)( _the_window * window_multiple + window_increment ) ), 1000000 );
  _intersend_time = intersend;
}
