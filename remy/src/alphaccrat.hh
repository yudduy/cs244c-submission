#ifndef ALPHACCRAT_HH
#define ALPHACCRAT_HH

#include <memory>

#include "rat.hh"

class AlphaCCRat : private Rat {
private:
  struct SharedState {
    enum class Phase { STARTUP, DRAIN, STEADY };

    bool initialized = false;
    Phase phase = Phase::STARTUP;
    double rtt_base = 1.0;
    double trend = 0.0;
    double util_ema = 1.0;
    double last_intersend = 0.0;
    int cooldown = 0;
  };

  std::shared_ptr<SharedState> _shared;

  static WhiskerTree & get_dummy_whiskers();
  void update_window_and_intersend() override;
  void apply_action( int window_increment, double window_multiple, double intersend );

public:
  AlphaCCRat();
  AlphaCCRat( const AlphaCCRat & other );
  AlphaCCRat & operator=( const AlphaCCRat & other );

  using Rat::reset;
  using Rat::packets_received;
  using Rat::send;
  using Rat::next_event_time;
  using Rat::packets_sent;
  using Rat::state_DNA;

  std::string str() const { return "AlphaCC/OpenEvolve sender-runner port"; }
};

#endif
