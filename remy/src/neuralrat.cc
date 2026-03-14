#include "neuralrat.hh"

WhiskerTree & NeuralRat::get_dummy_whiskers()
{
  static WhiskerTree dummy;
  return dummy;
}

NeuralRat::NeuralRat( RatBrain & brain )
  : Rat( get_dummy_whiskers() ),
    _brain( brain ),
    _local_network( brain.network()->clone_network() ),
    _episode_observations()
{
}

NeuralRat::NeuralRat( const NeuralRat & other )
  : Rat( other ),
    _brain( other._brain ),
    _local_network( other._local_network->clone_network() ),
    _episode_observations()  /* each copy starts with fresh observations */
{
}

void NeuralRat::update_window_and_intersend()
{
  ActionResult result = infer_action( _local_network, _memory, _the_window );
  _the_window = result.the_window;
  _intersend_time = result.intersend_time;
  _episode_observations.push_back( result.obs_action );
}

void NeuralRat::episode_done( double utility, size_t total_rollout_events )
{
  _brain.remember_episode( utility, _episode_observations, total_rollout_events );
  _episode_observations.clear();
}
