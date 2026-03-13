#ifndef EVOLVEDRAT_HH
#define EVOLVEDRAT_HH

#include "rat.hh"
#include "evolvedpolicy.hh"

class EvolvedRat : private Rat {
private:
  /* Persistent state (matches v5_best_policy.py _state dict) */
  double _max_bw;         /* pkts/ms, decayed max of recv_bw */
  double _eq_bw;          /* EMA of recv_bw */
  bool   _startup;
  int    _startup_rounds;
  double _last_min_rtt;   /* reserved for future: min_rtt from prior on period */

  static WhiskerTree & get_dummy_whiskers();

  void update_window_and_intersend() override;

public:
  EvolvedRat( EvolvedPolicy & policy );
  EvolvedRat( const EvolvedRat & other );

  using Rat::packets_received;
  using Rat::reset;
  using Rat::send;
  using Rat::next_event_time;
  using Rat::packets_sent;
  using Rat::state_DNA;
};

#endif
