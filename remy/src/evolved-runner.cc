/* Standalone evaluator for AlphaCC evolved policy.
   Does NOT require LibTorch — only protobuf + Remy core.
   Usage: ./evolved-runner link=0.946 rtt=100 on=1000 off=1000 nsrc=2 */

#include <cstdio>
#include <cmath>
#include <string>
#include <limits>

#include "evaluator.hh"
#include "configrange.hh"
#include "evolvedpolicy.hh"
#include "evolvedrat.hh"

using namespace std;

int main( int argc, char *argv[] )
{
  unsigned int num_senders = 2;
  double link_ppt = 1.0;
  double delay = 100.0;
  double mean_on_duration = 5000.0;
  double mean_off_duration = 5000.0;
  double buffer_size = numeric_limits<unsigned int>::max();
  double stochastic_loss_rate = 0;
  unsigned int simulation_ticks = 100000;

  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[ i ] );
    if ( arg.substr( 0, 5 ) == "nsrc=" ) {
      num_senders = atoi( arg.substr( 5 ).c_str() );
    } else if ( arg.substr( 0, 5 ) == "link=" ) {
      link_ppt = atof( arg.substr( 5 ).c_str() );
    } else if ( arg.substr( 0, 4 ) == "rtt=" ) {
      delay = atof( arg.substr( 4 ).c_str() );
    } else if ( arg.substr( 0, 3 ) == "on=" ) {
      mean_on_duration = atof( arg.substr( 3 ).c_str() );
    } else if ( arg.substr( 0, 4 ) == "off=" ) {
      mean_off_duration = atof( arg.substr( 4 ).c_str() );
    } else if ( arg.substr( 0, 4 ) == "buf=" ) {
      if (arg.substr( 4 ) == "inf") {
        buffer_size = numeric_limits<unsigned int>::max();
      } else {
        buffer_size = atoi( arg.substr( 4 ).c_str() );
      }
    } else if ( arg.substr( 0, 6 ) == "sloss=" ) {
      stochastic_loss_rate = atof( arg.substr( 6 ).c_str() );
    }
  }

  fprintf( stderr, "AlphaCC v5 evolved policy: link=%.3f ppt (%.1f Mbps), rtt=%.0f, on=%.0f, off=%.0f, nsrc=%u\n",
           link_ppt, link_ppt * 10, delay, mean_on_duration, mean_off_duration, num_senders );

  ConfigRange configuration_range;
  configuration_range.link_ppt = Range( link_ppt, link_ppt, 0 );
  configuration_range.rtt = Range( delay, delay, 0 );
  configuration_range.num_senders = Range( num_senders, num_senders, 0 );
  configuration_range.mean_on_duration = Range( mean_on_duration, mean_on_duration, 0 );
  configuration_range.mean_off_duration = Range( mean_off_duration, mean_off_duration, 0 );
  configuration_range.buffer_size = Range( buffer_size, buffer_size, 0 );
  configuration_range.stochastic_loss_rate = Range( stochastic_loss_rate, stochastic_loss_rate, 0 );
  configuration_range.simulation_ticks = simulation_ticks;

  EvolvedPolicy policy;
  Evaluator< EvolvedPolicy > eval( configuration_range );
  auto outcome = eval.score( policy, false, 10 );

  /* Output format matches sender-runner for direct comparison */
  printf( "score = %f\n", outcome.score );
  double norm_score = 0;

  for ( auto &run : outcome.throughputs_delays ) {
    printf( "===\nconfig: %s\n", run.first.str().c_str() );
    for ( auto &x : run.second ) {
      printf( "sender: [tp=%f, del=%f]\n", x.first / run.first.link_ppt, x.second / run.first.delay );
      norm_score += log2( x.first / run.first.link_ppt ) - log2( x.second / run.first.delay );
    }
  }

  printf( "normalized_score = %f\n", norm_score );

  return 0;
}
