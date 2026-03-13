#include <cstdio>
#include <vector>
#include <string>
#include <limits>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "evaluator.hh"
#include "configrange.hh"
#include "ratbrain.hh"
#include "neuralrat.hh"
#include "evolvedpolicy.hh"
#include "evolvedrat.hh"
using namespace std;

enum class SenderType { WHISKER, POISSON, NEURAL, EVOLVED };

template <typename T>
void print_tree(T & tree)
{
  if ( tree.has_config() ) {
    printf( "Prior assumptions:\n%s\n\n", tree.config().DebugString().c_str() );
  }

  if ( tree.has_optimizer() ) {
    printf( "Remy optimization settings:\n%s\n\n", tree.optimizer().DebugString().c_str() );
  }
}

template <typename T>
void parse_outcome( T & outcome )
{
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

  printf( "Rules: %s\n", outcome.used_actions.str().c_str() );
}

int main( int argc, char *argv[] )
{
  SenderType sender_type = SenderType::WHISKER;
  string input_filename;
  string config_filename;
  int hidden_size = 128;
  int num_hidden_layers = 2;
  unsigned int num_senders = 2;
  double link_ppt = 1.0;
  double delay = 100.0;
  double mean_on_duration = 5000.0;
  double mean_off_duration = 5000.0;
  double buffer_size = numeric_limits<unsigned int>::max();
  double stochastic_loss_rate = 0;
  unsigned int simulation_ticks = 100000;

  /* Parse all arguments */
  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[ i ] );
    if ( arg.substr( 0, 7 ) == "sender=" ) {
      string type_str( arg.substr( 7 ) );
      if ( type_str == "poisson" ) {
        sender_type = SenderType::POISSON;
        fprintf( stderr, "Running poisson sender\n" );
      } else if ( type_str == "neural" ) {
        sender_type = SenderType::NEURAL;
        fprintf( stderr, "Running neural sender\n" );
      } else if ( type_str == "evolved" ) {
        sender_type = SenderType::EVOLVED;
        fprintf( stderr, "Running evolved (AlphaCC v5) sender\n" );
      }
    } else if ( arg.substr( 0, 3 ) == "if=" ) {
      input_filename = arg.substr( 3 );
    } else if ( arg.substr( 0, 3 ) == "cf=" ) {
      config_filename = arg.substr( 3 );
    } else if ( arg.substr( 0, 12 ) == "hidden_size=" ) {
      hidden_size = atoi( arg.substr( 12 ).c_str() );
    } else if ( arg.substr( 0, 18 ) == "num_hidden_layers=" ) {
      num_hidden_layers = atoi( arg.substr( 18 ).c_str() );
    } else if ( arg.substr( 0, 5 ) == "nsrc=" ) {
      num_senders = atoi( arg.substr( 5 ).c_str() );
      fprintf( stderr, "Setting num_senders to %d\n", num_senders );
    } else if ( arg.substr( 0, 5 ) == "link=" ) {
      link_ppt = atof( arg.substr( 5 ).c_str() );
      fprintf( stderr, "Setting link packets per ms to %f\n", link_ppt );
    } else if ( arg.substr( 0, 4 ) == "rtt=" ) {
      delay = atof( arg.substr( 4 ).c_str() );
      fprintf( stderr, "Setting delay to %f ms\n", delay );
    } else if ( arg.substr( 0, 3 ) == "on=" ) {
      mean_on_duration = atof( arg.substr( 3 ).c_str() );
      fprintf( stderr, "Setting mean_on_duration to %f ms\n", mean_on_duration );
    } else if ( arg.substr( 0, 4 ) == "off=" ) {
      mean_off_duration = atof( arg.substr( 4 ).c_str() );
      fprintf( stderr, "Setting mean_off_duration to %f ms\n", mean_off_duration );
    } else if ( arg.substr( 0, 4 ) == "buf=" ) {
      if (arg.substr( 4 ) == "inf") {
        buffer_size = numeric_limits<unsigned int>::max();
      } else {
        buffer_size = atoi( arg.substr( 4 ).c_str() );
      }
    } else if ( arg.substr( 0, 6 ) == "sloss=" ) {
      stochastic_loss_rate = atof( arg.substr( 6 ).c_str() );
      fprintf( stderr, "Setting stochastic loss rate to %f\n", stochastic_loss_rate );
    }
  }

  ConfigRange configuration_range;
  configuration_range.link_ppt = Range( link_ppt,link_ppt, 0 ); /* 1 Mbps to 10 Mbps */
  configuration_range.rtt = Range( delay, delay, 0 ); /* ms */
  configuration_range.num_senders = Range( num_senders, num_senders, 0 );
  configuration_range.mean_on_duration = Range( mean_on_duration, mean_on_duration, 0 );
  configuration_range.mean_off_duration = Range( mean_off_duration, mean_off_duration, 0 );
  configuration_range.buffer_size = Range( buffer_size, buffer_size, 0 );
  configuration_range.stochastic_loss_rate = Range( stochastic_loss_rate, stochastic_loss_rate, 0);
  configuration_range.simulation_ticks = simulation_ticks;

  if ( sender_type == SenderType::EVOLVED ) {
    /* No input file needed — policy is hardcoded in EvolvedRat */
    EvolvedPolicy policy;
    Evaluator< EvolvedPolicy > eval( configuration_range );
    auto outcome = eval.score( policy, false, 10 );
    parse_outcome< Evaluator< EvolvedPolicy >::Outcome >( outcome );
  } else if ( sender_type == SenderType::NEURAL ) {
    /* Print prior assumptions from training config file */
    if ( !config_filename.empty() ) {
      int cfd = open( config_filename.c_str(), O_RDONLY );
      if ( cfd < 0 ) {
        perror( "open config file" );
        exit( 1 );
      }
      RemyBuffers::ConfigRange input_config;
      if ( !input_config.ParseFromFileDescriptor( cfd ) ) {
        fprintf( stderr, "Could not parse config from %s.\n", config_filename.c_str() );
        exit( 1 );
      }
      if ( close( cfd ) < 0 ) {
        perror( "close" );
        exit( 1 );
      }
      printf( "Prior assumptions:\n%s\n\n", input_config.DebugString().c_str() );
    }

    /* Load neural model */
    TrainingConfig tc;
    tc.hidden_size = hidden_size;
    tc.num_hidden_layers = num_hidden_layers;
    tc.replay_buffer_size = 1; /* inference only */

    RatBrain brain( tc );
    brain.load( input_filename );

    Evaluator< RatBrain > eval( configuration_range );
    auto outcome = eval.score( brain, false, 10 );
    parse_outcome< Evaluator< RatBrain >::Outcome >( outcome );
  } else {
    /* Load protobuf input file (whisker or poisson) */
    RemyBuffers::WhiskerTree whisker_tree_proto;
    RemyBuffers::FinTree fin_tree_proto;

    int fd = open( input_filename.c_str(), O_RDONLY );
    if ( fd < 0 ) {
      perror( "open" );
      exit( 1 );
    }

    bool parsed = ( sender_type == SenderType::POISSON )
      ? fin_tree_proto.ParseFromFileDescriptor( fd )
      : whisker_tree_proto.ParseFromFileDescriptor( fd );
    if ( !parsed ) {
      fprintf( stderr, "Could not parse %s.\n", input_filename.c_str() );
      exit( 1 );
    }

    if ( close( fd ) < 0 ) {
      perror( "close" );
      exit( 1 );
    }

    if ( sender_type == SenderType::POISSON ) {
      print_tree< RemyBuffers::FinTree >( fin_tree_proto );
      FinTree fins( fin_tree_proto );

      Evaluator< FinTree > eval( configuration_range );
      auto outcome = eval.score( fins, false, 10 );
      parse_outcome< Evaluator< FinTree >::Outcome >( outcome );
    } else {
      print_tree< RemyBuffers::WhiskerTree >( whisker_tree_proto );
      WhiskerTree whiskers( whisker_tree_proto );

      Evaluator< WhiskerTree > eval( configuration_range );
      auto outcome = eval.score( whiskers, false, 10 );
      parse_outcome< Evaluator< WhiskerTree >::Outcome >( outcome );
    }
  }

  return 0;
}
