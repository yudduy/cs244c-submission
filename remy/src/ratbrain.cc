#include "ratbrain.hh"
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

/* ---- PolicyNet implementation ---- */

PolicyNetImpl::PolicyNetImpl( int hidden_size, int num_hidden_layers )
  : _hidden_size( hidden_size ), _num_hidden_layers( num_hidden_layers )
{
  obs_mean  = register_buffer( "obs_mean",  torch::zeros( {INPUT_DIM} ) );
  obs_var   = register_buffer( "obs_var",   torch::ones( {INPUT_DIM} ) );
  obs_count = register_buffer( "obs_count", torch::zeros( {1} ) );

  input_proj = register_module( "input_proj", torch::nn::Linear( INPUT_DIM, hidden_size ) );

  for ( int i = 0; i < num_hidden_layers; i++ ) {
    hidden_layers.push_back( register_module( "fc" + to_string( i ),
      torch::nn::Linear( hidden_size, hidden_size ) ) );
  }

  policy_wi = register_module( "policy_wi", torch::nn::Linear( hidden_size, NUM_WINDOW_INCREMENT ) );
  policy_wm = register_module( "policy_wm", torch::nn::Linear( hidden_size, NUM_WINDOW_MULTIPLE ) );
  policy_is = register_module( "policy_is", torch::nn::Linear( hidden_size, NUM_INTERSEND ) );
}

void PolicyNetImpl::update_obs_stats( torch::Tensor batch )
{
  /* Welford's parallel/batch update: merge batch stats into running stats */
  auto batch_mean  = batch.mean( 0 );
  auto batch_var   = batch.var( 0, /*unbiased=*/false );
  auto batch_count = static_cast<double>( batch.size( 0 ) );
  auto old_count   = obs_count.item<double>();
  auto new_count   = old_count + batch_count;

  auto delta = batch_mean - obs_mean;
  obs_mean.add_( delta * ( batch_count / new_count ) );
  obs_var.copy_( ( obs_var * old_count + batch_var * batch_count
                   + delta * delta * ( old_count * batch_count / new_count ) ) / new_count );
  obs_count.fill_( new_count );

}

tuple<torch::Tensor, torch::Tensor, torch::Tensor>
PolicyNetImpl::forward( torch::Tensor x )
{
  /* Normalize input by running mean/std (per-dimension) */
  x = ( x - obs_mean ) / ( obs_var.sqrt() + 1e-8 );

  x = torch::gelu( input_proj->forward( x ) );

  for ( size_t i = 0; i < hidden_layers.size(); i++ ) {
    auto residual = x;
    x = torch::gelu( hidden_layers[i]->forward( x ) );
    x = x + residual;
  }

  return make_tuple(
    policy_wi->forward( x ),
    policy_wm->forward( x ),
    policy_is->forward( x )
  );
}

std::shared_ptr<PolicyNetImpl> PolicyNetImpl::clone_network() const
{
  auto net = std::make_shared<PolicyNetImpl>( _hidden_size, _num_hidden_layers );
  torch::NoGradGuard no_grad;
  /* Copy parameters */
  auto src_params = parameters();
  auto dst_params = net->parameters();
  for ( size_t i = 0; i < src_params.size(); i++ ) {
    dst_params[i].copy_( src_params[i] );
  }
  /* Copy buffers (running observation stats) */
  auto src_bufs = buffers();
  auto dst_bufs = net->buffers();
  for ( size_t i = 0; i < src_bufs.size(); i++ ) {
    dst_bufs[i].copy_( src_bufs[i] );
  }
  net->eval();
  return net;
}

/* ---- RatBrain implementation ---- */

RatBrain::RatBrain( const TrainingConfig & config )
  : _config( config ),
    _device( torch::cuda::is_available() ? torch::kCUDA : torch::kCPU ),
    _network( config.hidden_size, config.num_hidden_layers ),
    _optimizer( nullptr ),
    _buf_obs( torch::zeros( {static_cast<long>(config.replay_buffer_size), static_cast<long>(INPUT_DIM)} ) ),
    _buf_utility( torch::zeros( {static_cast<long>(config.replay_buffer_size)} ) ),
    _buf_old_log_prob( torch::zeros( {static_cast<long>(config.replay_buffer_size)} ) ),
    _buf_action_wi( torch::zeros( {static_cast<long>(config.replay_buffer_size)}, torch::kLong ) ),
    _buf_action_wm( torch::zeros( {static_cast<long>(config.replay_buffer_size)}, torch::kLong ) ),
    _buf_action_is( torch::zeros( {static_cast<long>(config.replay_buffer_size)}, torch::kLong ) ),
    _buf_total_events( torch::zeros( {static_cast<long>(config.replay_buffer_size)} ) ),
    _write_pos( 0 ),
    _buffer_count( 0 )
{
  assert(_device == torch::kCPU);
  _network->to( _device );
  _optimizer = make_shared<torch::optim::AdamW>(
    _network->parameters(),
    torch::optim::AdamWOptions( config.learning_rate ).weight_decay( config.weight_decay ) );
  cerr << "RatBrain using device: " << _device << endl;
}

ActionResult infer_action( PolicyNet & net, const Memory & memory, int current_window )
{
  torch::NoGradGuard no_grad;

  /* Convert active memory fields to tensor. Whisker tree uses first 4 features of memory. */
  float obs[INPUT_DIM];
  for ( int i = 0; i < INPUT_DIM; i++ ) {
    obs[i] = static_cast<float>( memory.field( ACTIVE_AXES[i] ) );
  }
  auto obs_tensor = torch::from_blob( obs, {1, static_cast<long>(INPUT_DIM)} );

  /* Forward pass */
  auto output = net->forward( obs_tensor );
  auto logits_wi = get<0>( output );
  auto logits_wm = get<1>( output );
  auto logits_is = get<2>( output );

  /* Sample from categorical distributions */
  auto probs_wi = torch::softmax( logits_wi, 1 );
  auto probs_wm = torch::softmax( logits_wm, 1 );
  auto probs_is = torch::softmax( logits_is, 1 );

  int action_wi = torch::multinomial( probs_wi, 1 ).item<int64_t>();
  int action_wm = torch::multinomial( probs_wm, 1 ).item<int64_t>();
  int action_is = torch::multinomial( probs_is, 1 ).item<int64_t>();

  /* Compute log probabilities */
  auto log_probs_wi = torch::log_softmax( logits_wi, 1 );
  auto log_probs_wm = torch::log_softmax( logits_wm, 1 );
  auto log_probs_is = torch::log_softmax( logits_is, 1 );

  float log_prob = log_probs_wi[0][action_wi].item<float>()
                 + log_probs_wm[0][action_wm].item<float>()
                 + log_probs_is[0][action_is].item<float>();

  /* Convert indices to actual parameter values */
  int window_increment = WINDOW_INCREMENT_MIN + action_wi * WINDOW_INCREMENT_STEP;
  double window_multiple = WINDOW_MULTIPLE_MIN + action_wm * WINDOW_MULTIPLE_STEP;
  double intersend = INTERSEND_MIN + action_is * INTERSEND_STEP;

  /* Compute new window (same formula as Whisker::window) */
  int new_window = min( max( 0, static_cast<int>( current_window * window_multiple + window_increment ) ), 1000000 );

  /* Build result */
  ActionResult result;
  result.the_window = new_window;
  result.intersend_time = intersend;

  std::copy( obs, obs + INPUT_DIM, result.obs_action.observation.begin() );
  result.obs_action.action_wi_idx = action_wi;
  result.obs_action.action_wm_idx = action_wm;
  result.obs_action.action_is_idx = action_is;
  result.obs_action.old_log_prob = log_prob;

  return result;
}


void RatBrain::remember_episode( double utility, const vector<ObsAction> & observations, size_t total_rollout_events )
{
  cerr << "remember_episode: " << observations.size() << " steps, utility=" << utility << ", total_rollout_events=" << total_rollout_events << endl;

  /* Build observation tensor for batch stats update */
  auto obs_tensor = torch::zeros( {static_cast<long>( observations.size() ), INPUT_DIM} );
  for ( size_t i = 0; i < observations.size(); i++ ) {
    for ( int j = 0; j < INPUT_DIM; j++ ) {
      obs_tensor[i][j] = static_cast<float>( observations[i].observation[j] );
    }
  }
  _network->update_obs_stats( obs_tensor );

  for ( const auto & obs : observations ) {
    for ( int j = 0; j < INPUT_DIM; j++ ) {
      _buf_obs[_write_pos][j] = static_cast<float>( obs.observation[j] );
    }
    _buf_utility[_write_pos] = static_cast<float>( utility );
    _buf_old_log_prob[_write_pos] = obs.old_log_prob;
    _buf_action_wi[_write_pos] = static_cast<int64_t>( obs.action_wi_idx );
    _buf_action_wm[_write_pos] = static_cast<int64_t>( obs.action_wm_idx );
    _buf_action_is[_write_pos] = static_cast<int64_t>( obs.action_is_idx );
    _buf_total_events[_write_pos] = static_cast<float>( total_rollout_events );

    _write_pos = ( _write_pos + 1 ) % _config.replay_buffer_size;
    if ( _buffer_count < _config.replay_buffer_size ) _buffer_count++;
  }
}

void RatBrain::learn()
{
  if ( _buffer_count < _config.batch_size ) return;

  const long mini_batch_size = static_cast<long>( _config.batch_size / _config.accumulation_steps );

  for ( size_t train_iter = 0; train_iter < _config.utd_ratio; train_iter++ ) {
    /* Sample random batch indices for the full effective batch */
    auto indices = torch::randint( 0, static_cast<long>(_buffer_count),
                                   {static_cast<long>(_config.batch_size)}, torch::kLong );

    /* GRPO: advantages are just normalized utilities across the full batch */
    auto all_utilities = _buf_utility.index_select( 0, indices );
    auto all_advantages = ( all_utilities - all_utilities.mean() ) / all_utilities.std().clamp_min( 1e-8 );

    /* Compute average total_events across the full batch for reweighting */
    auto all_total_events = _buf_total_events.index_select( 0, indices );
    auto avg_total_events = all_total_events.mean();

    /* Compute losses with gradient accumulation */
    _optimizer->zero_grad();

    float accum_loss = 0, accum_entropy = 0, accum_policy_loss = 0;
    long accum_clipped = 0;

    for ( size_t accum_step = 0; accum_step < _config.accumulation_steps; accum_step++ ) {
      /* Slice indices for this mini-batch and load to GPU */
      auto mb_indices = indices.slice( 0,
        accum_step * mini_batch_size, ( accum_step + 1 ) * mini_batch_size );

      auto obs_batch = _buf_obs.index_select( 0, mb_indices ).to( _device );
      auto old_log_prob_batch = _buf_old_log_prob.index_select( 0, mb_indices ).to( _device );
      auto action_wi_batch = _buf_action_wi.index_select( 0, mb_indices ).to( _device );
      auto action_wm_batch = _buf_action_wm.index_select( 0, mb_indices ).to( _device );
      auto action_is_batch = _buf_action_is.index_select( 0, mb_indices ).to( _device );
      auto total_events_batch = _buf_total_events.index_select( 0, mb_indices ).to( _device );

      /* Pre-computed normalized advantages for this mini-batch */
      auto advantage = all_advantages.slice( 0,
        accum_step * mini_batch_size, ( accum_step + 1 ) * mini_batch_size ).to( _device );

      /* Forward pass */
      auto output = _network->forward( obs_batch );
      auto logits_wi = get<0>( output );
      auto logits_wm = get<1>( output );
      auto logits_is = get<2>( output );

      /* Compute new log probabilities for taken actions */
      auto log_probs_wi = torch::log_softmax( logits_wi, 1 );
      auto log_probs_wm = torch::log_softmax( logits_wm, 1 );
      auto log_probs_is = torch::log_softmax( logits_is, 1 );

      auto new_log_prob = log_probs_wi.gather( 1, action_wi_batch.unsqueeze( 1 ) ).squeeze( 1 )
                        + log_probs_wm.gather( 1, action_wm_batch.unsqueeze( 1 ) ).squeeze( 1 )
                        + log_probs_is.gather( 1, action_is_batch.unsqueeze( 1 ) ).squeeze( 1 );

      /* PPO clipped surrogate loss: divide by total_events (per-rollout normalization)
         and by num_senders (per-sender normalization), multiply by avg_num_senders
         to keep gradient scale consistent across configs */
      auto ratio = torch::exp( new_log_prob - old_log_prob_batch );
      auto surr1 = ratio * advantage;
      auto surr2 = torch::clamp( ratio, 1.0 - _config.ppo_epsilon, 1.0 + _config.ppo_epsilon ) * advantage;
      auto per_sample_weight = avg_total_events / total_events_batch;
      auto policy_loss = -( torch::min( surr1, surr2 ) * per_sample_weight ).mean();
      accum_clipped += (ratio.lt( 1.0 - _config.ppo_epsilon ) | ratio.gt( 1.0 + _config.ppo_epsilon )).sum().item<long>();

      /* Entropy bonus */
      auto entropy_wi = -( torch::softmax( logits_wi, 1 ) * log_probs_wi ).sum( 1 );
      auto entropy_wm = -( torch::softmax( logits_wm, 1 ) * log_probs_wm ).sum( 1 );
      auto entropy_is = -( torch::softmax( logits_is, 1 ) * log_probs_is ).sum( 1 );
      auto entropy = ( ( entropy_wi + entropy_wm + entropy_is ) * per_sample_weight ).mean();

      /* Scale loss by accumulation steps so gradients average correctly */
      auto loss = ( policy_loss - _config.entropy_coeff * entropy )
                  / static_cast<double>( _config.accumulation_steps );
      loss.backward();

      /* Track unscaled metrics for logging */
      accum_loss += loss.item<float>() * _config.accumulation_steps;
      accum_entropy += entropy.item<float>();
      accum_policy_loss += policy_loss.item<float>();
    }

    double grad_norm;
    if ( _config.max_grad_norm > 0 ) {
      grad_norm = torch::nn::utils::clip_grad_norm_( _network->parameters(), _config.max_grad_norm );
    } else {
      // compute grad norm for logging without clipping
      double total = 0.0;
      for ( const auto & p : _network->parameters() ) {
        if ( p.grad().defined() ) {
          total += p.grad().norm().item<double>() * p.grad().norm().item<double>();
        }
      }
      grad_norm = std::sqrt( total );
    }
    _optimizer->step();

    auto obs_std = _network->obs_var.sqrt();
    cerr << "learn: loss=" << accum_loss / _config.accumulation_steps
         << " entropy=" << accum_entropy / _config.accumulation_steps
         << " policy_loss=" << accum_policy_loss / _config.accumulation_steps
         << " clip_frac=" << static_cast<double>(accum_clipped) / _config.batch_size
         << " grad_norm=" << grad_norm
         << " obs_mean=[" << _network->obs_mean[0].item<float>() << ", " << _network->obs_mean[1].item<float>()
         << ", " << _network->obs_mean[2].item<float>() << ", " << _network->obs_mean[3].item<float>()
         << "] obs_std=[" << obs_std[0].item<float>() << ", " << obs_std[1].item<float>()
         << ", " << obs_std[2].item<float>() << ", " << obs_std[3].item<float>() << "]" << endl;
  }
}

void RatBrain::save( const string & filename ) const
{
  torch::serialize::OutputArchive archive;
  _network->save( archive );
  archive.save_to( filename );
  fprintf( stderr, "Saved model to %s\n", filename.c_str() );
}

void RatBrain::load( const string & filename )
{
  torch::serialize::InputArchive archive;
  archive.load_from( filename );
  _network->load( archive );
  fprintf( stderr, "Loaded model from %s\n", filename.c_str() );
}
