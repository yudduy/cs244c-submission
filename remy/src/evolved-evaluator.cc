/* Evaluator<EvolvedPolicy> template specialization.
   Mirrors neural-evaluator.cc but uses EvolvedRat senders. */

#include "evolvedpolicy.hh"
#include "evolvedrat.hh"
#include "evaluator.hh"

/* Include template implementation files for EvolvedRat network instantiation */
#include "network.cc"
#include "rat-templates.cc"

using namespace std;

template <>
Evaluator<EvolvedPolicy>::Evaluator( const ConfigRange & range )
  : _prng_seed( global_PRNG()() ),
    _tick_count( range.simulation_ticks ),
    _configs( get_config_outer_product( range ) )
{
}

template <>
typename Evaluator<EvolvedPolicy>::Outcome Evaluator<EvolvedPolicy>::score(
    EvolvedPolicy & policy,
    const unsigned int prng_seed,
    const vector<NetConfig> & configs,
    const bool trace __attribute((unused)),
    const unsigned int ticks_to_run )
{
  PRNG run_prng( prng_seed );

  Evaluator::Outcome the_outcome;
  for ( auto &x : configs ) {
    Network< SenderGang<EvolvedRat, TimeSwitchedSender<EvolvedRat>>,
             SenderGang<EvolvedRat, TimeSwitchedSender<EvolvedRat>> >
      network1( EvolvedRat( policy ), run_prng, x );
    network1.run_simulation( ticks_to_run );

    the_outcome.score += network1.senders().utility();
    the_outcome.throughputs_delays.emplace_back( x, network1.senders().throughputs_delays() );
  }

  return the_outcome;
}

template <>
typename Evaluator<EvolvedPolicy>::Outcome Evaluator<EvolvedPolicy>::score(
    EvolvedPolicy & run_actions,
    const bool trace,
    const double carefulness ) const
{
  return score( run_actions, _prng_seed, _configs, trace, _tick_count * carefulness );
}
