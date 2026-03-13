#include <torch/torch.h>

#include "ratbrain.hh"
#include "neuralrat.hh"
#include "evaluator.hh"

/* Include template implementation files for NeuralRat network instantiation */
#include "network.cc"
#include "rat-templates.cc"

using namespace std;

/* ---- Evaluator<RatBrain> full specializations ----
   These mirror the WhiskerTree/FinTree specializations in evaluator.cc
   but use NeuralRat senders driven by the loaded RatBrain policy. */

template <>
Evaluator<RatBrain>::Evaluator( const ConfigRange & range )
  : _prng_seed( global_PRNG()() ),
    _tick_count( range.simulation_ticks ),
    _configs( get_config_outer_product( range ) )
{
  torch::set_num_threads( 1 );
}

template <>
typename Evaluator<RatBrain>::Outcome Evaluator<RatBrain>::score(
    RatBrain & brain,
    const unsigned int prng_seed,
    const vector<NetConfig> & configs,
    const bool trace __attribute((unused)),
    const unsigned int ticks_to_run )
{
  PRNG run_prng( prng_seed );

  Evaluator::Outcome the_outcome;
  for ( auto &x : configs ) {
    Network< SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>>,
             SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>> >
      network1( NeuralRat( brain ), run_prng, x );
    network1.run_simulation( ticks_to_run );

    the_outcome.score += network1.senders().utility();
    the_outcome.throughputs_delays.emplace_back( x, network1.senders().throughputs_delays() );
  }

  return the_outcome;
}

template <>
typename Evaluator<RatBrain>::Outcome Evaluator<RatBrain>::score(
    RatBrain & run_actions,
    const bool trace,
    const double carefulness ) const
{
  return score( run_actions, _prng_seed, _configs, trace, _tick_count * carefulness );
}
