#include <cstdio>
#include <vector>
#include <string>
#include <future>
#include <mutex>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <chrono>

#include "CLI11.hpp"
#include "ratbrain.hh"
#include "neuralrat.hh"
#include "configrange.hh"
#include "evaluator.hh"

/* Include template implementation files for NeuralRat network instantiation */
#include "network.cc"
#include "rat-templates.cc"

using namespace std;

/* ---- Experience collection and main loop ---- */

double collect_experience( RatBrain & brain,
                           const unsigned int prng_seed,
                           const vector<NetConfig> & configs,
                           const unsigned int tick_count )
{
  mutex brain_mutex;

  /* Generate deterministic per-config PRNG seeds (each thread needs its own) */
  PRNG seed_prng( prng_seed );
  vector<unsigned int> seeds;
  for ( size_t i = 0; i < configs.size(); i++ ) {
    seeds.push_back( seed_prng() );
  }

  /* Launch a parallel async task for each config (mirrors breeder.cc pattern) */
  size_t total_events = 0;
  vector<future<double>> futures;

  for ( size_t i = 0; i < configs.size(); i++ ) {
    futures.push_back(
      async( launch::async,
        [&brain, &configs, &brain_mutex, &total_events, tick_count]
        ( unsigned int seed, size_t idx ) -> double {
          PRNG run_prng( seed );

          /* Run simulation with NeuralRat senders */
          Network< SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>>,
                   SenderGang<NeuralRat, TimeSwitchedSender<NeuralRat>> >
            network( NeuralRat( brain ), run_prng, configs[idx] );

          network.run_simulation( tick_count );

          double sim_utility = network.senders().utility();

          /* Safely record experience into the shared replay buffer.
             Inference uses the NeuralRat's local network clone under NoGradGuard
             and is safe to call concurrently, but remember_episode writes to the
             shared buffer and needs mutex protection. */
          {
            lock_guard<mutex> lock( brain_mutex );
            auto & gang = network.mutable_senders().mutable_gang1();
            unsigned int num_senders = gang.count_senders();

            /* First pass: count total events across all senders in this rollout */
            size_t rollout_events = 0;
            for ( unsigned int j = 0; j < num_senders; j++ ) {
              rollout_events += gang.mutable_sender( j ).mutable_inner_sender().observation_count();
            }

            /* Second pass: store experience with the total rollout event count */
            for ( unsigned int j = 0; j < num_senders; j++ ) {
              gang.mutable_sender( j ).mutable_inner_sender().episode_done( sim_utility, rollout_events );
            }
            total_events += rollout_events;
          }

          return sim_utility;
        },
        seeds[i], i )
    );
  }

  /* Collect results from all futures (blocks until each completes) */
  double total_score = 0;
  double total_futures = 0;
  for ( auto & f : futures ) {
    total_score += f.get();
    total_futures += 1;
  }

  printf( "total_events = %zu\n", total_events );

  return total_score / total_futures;
}

int main( int argc, char *argv[] )
{
  CLI::App app{ "Neural Rat Trainer" };

  /* ---- Simulation / IO options ---- */
  string config_filename;
  app.add_option( "--cf", config_filename, "Input configuration protobuf file" )->required();

  string output_filename;
  app.add_option( "--of", output_filename, "Output checkpoint prefix" );

  unsigned int save_every = 0;
  app.add_option( "--save-every", save_every, "Save checkpoint every N runs (0 = disabled)" );

  unsigned int num_config_evals = 8;
  app.add_option( "--num-config-evals", num_config_evals, "Times to replicate config grid per iteration" );

  /* ---- Training hyperparameters ---- */
  TrainingConfig tc;

  app.add_option( "--replay-buffer-size", tc.replay_buffer_size, "Replay buffer capacity" );
  app.add_option( "--batch-size",         tc.batch_size,         "Effective batch size" );
  app.add_option( "--lr",                 tc.learning_rate,      "Adam learning rate" );
  app.add_option( "--ppo-epsilon",        tc.ppo_epsilon,        "PPO clipping epsilon" );
  app.add_option( "--utd-ratio",          tc.utd_ratio,          "Update-to-data ratio" );
  app.add_option( "--entropy-coeff",      tc.entropy_coeff,      "Entropy bonus coefficient" );
  app.add_option( "--weight-decay",       tc.weight_decay,       "AdamW weight decay coefficient" );
  app.add_option( "--max-grad-norm",      tc.max_grad_norm,      "Max gradient norm for clipping" );
  app.add_option( "--accumulation-steps", tc.accumulation_steps, "Gradient accumulation steps" );
  app.add_option( "--hidden-size",        tc.hidden_size,        "Hidden layer width" );
  app.add_option( "--num-hidden-layers",  tc.num_hidden_layers,  "Number of hidden layers" );

  CLI11_PARSE( app, argc, argv );

  /* ---- Load config protobuf ---- */
  RemyBuffers::ConfigRange input_config;
  int cfd = open( config_filename.c_str(), O_RDONLY );
  if ( cfd < 0 ) {
    perror( "open config file error" );
    exit( 1 );
  }
  if ( !input_config.ParseFromFileDescriptor( cfd ) ) {
    fprintf( stderr, "Could not parse input config from file %s.\n", config_filename.c_str() );
    exit( 1 );
  }
  if ( close( cfd ) < 0 ) {
    perror( "close" );
    exit( 1 );
  }

  ConfigRange config_range( input_config );
  vector<NetConfig> base_configs = get_config_outer_product( config_range );
  unsigned int tick_count = config_range.simulation_ticks;

  vector<NetConfig> configs;
  configs.reserve( base_configs.size() * num_config_evals );
  for ( unsigned int r = 0; r < num_config_evals; r++ ) {
    configs.insert( configs.end(), base_configs.begin(), base_configs.end() );
  }

  printf( "#######################\n" );
  printf( "Neural Rat Trainer\n" );
  printf( "  config file:         %s\n", config_filename.c_str() );
  printf( "  output prefix:       %s\n", output_filename.empty() ? "(none)" : output_filename.c_str() );
  printf( "  save_every:          %u\n", save_every );
  printf( "  num_config_evals:    %u\n", num_config_evals );
  printf( "  base configs:        %zu\n", base_configs.size() );
  printf( "  total configs:       %zu\n", configs.size() );
  printf( "  tick_count:          %u\n", tick_count );
  printf( "  replay_buffer_size:  %zu\n", tc.replay_buffer_size );
  printf( "  batch_size:          %zu\n", tc.batch_size );
  printf( "  lr:                  %g\n",  tc.learning_rate );
  printf( "  ppo_epsilon:         %g\n",  tc.ppo_epsilon );
  printf( "  utd_ratio:           %zu\n", tc.utd_ratio );
  printf( "  entropy_coeff:       %g\n",  tc.entropy_coeff );
  printf( "  weight_decay:        %g\n",  tc.weight_decay );
  printf( "  max_grad_norm:       %g\n",  tc.max_grad_norm );
  printf( "  accumulation_steps:  %zu\n", tc.accumulation_steps );
  printf( "  hidden_size:         %d\n",  tc.hidden_size );
  printf( "  num_hidden_layers:   %d\n",  tc.num_hidden_layers );
  printf( "#######################\n" );

  /* Create checkpoint directory if it does not exist */
  if ( !output_filename.empty() ) {
    string dir = output_filename.substr( 0, output_filename.rfind( '/' ) );
    if ( !dir.empty() && dir != output_filename ) {
      mkdir( dir.c_str(), 0755 );
    }
  }

  RatBrain brain( tc );

  unsigned int run = 0;

  const int num_cpus = thread::hardware_concurrency();

  while ( 1 ) {
    unsigned int prng_seed = global_PRNG()();
    auto t0 = chrono::steady_clock::now();
    torch::set_num_threads( 1 );
    double score = collect_experience( brain, prng_seed, configs, tick_count );
    auto t1 = chrono::steady_clock::now();
    double collect_secs = chrono::duration<double>( t1 - t0 ).count();
    printf( "run = %u, score = %f, collect_time = %.2fs\n", run, score, collect_secs );

    torch::set_num_threads( num_cpus );
    brain.learn();

    if ( save_every > 0 && !output_filename.empty() && ( run % save_every == 0 ) ) {
      char of[ 256 ];
      snprintf( of, 256, "%s.%u", output_filename.c_str(), run );
      brain.save( string( of ) );
    }

    fflush( NULL );
    run++;
  }

  return 0;
}
