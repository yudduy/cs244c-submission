#ifndef RATBRAIN_HH
#define RATBRAIN_HH

#include <vector>
#include <array>
#include <tuple>
#include <algorithm>
#include <torch/torch.h>

#include "memory.hh"
#include "memoryrange.hh"

/* Observation space: same active axes as the WhiskerTree default for Rat */
constexpr std::array<Axis, 4> ACTIVE_AXES = {
  RemyBuffers::MemoryRange::SEND_EWMA,
  RemyBuffers::MemoryRange::REC_EWMA,
  RemyBuffers::MemoryRange::RTT_RATIO,
  RemyBuffers::MemoryRange::SLOW_REC_EWMA
};
constexpr int INPUT_DIM = ACTIVE_AXES.size();

/* Training hyperparameters — runtime-configurable via TrainingConfig */
struct TrainingConfig {
  size_t replay_buffer_size = 3000000;
  size_t batch_size         = 262144;
  double learning_rate      = 3e-4;
  double ppo_epsilon        = 0.2;
  size_t utd_ratio          = 8;      /* training iterations per experience collection */

  double entropy_coeff      = 0.005;
  double weight_decay       = 0.0;
  double max_grad_norm      = -1.0;  /* -1 for no clipping */
  size_t accumulation_steps = 16;
  int    hidden_size        = 128;
  int    num_hidden_layers  = 2;
};

/* Action space ranges (matching Whisker optimization settings in whisker.hh) */
constexpr int    WINDOW_INCREMENT_MIN  = 0;
constexpr int    WINDOW_INCREMENT_MAX  = 256;
constexpr int    WINDOW_INCREMENT_STEP = 1;

constexpr double WINDOW_MULTIPLE_MIN  = 0.0;
constexpr double WINDOW_MULTIPLE_MAX  = 1.0;
constexpr double WINDOW_MULTIPLE_STEP = 0.01;

constexpr double INTERSEND_MIN  = 0.25;
constexpr double INTERSEND_MAX  = 3.0;
constexpr double INTERSEND_STEP = 0.05;

/* Derived action space dimensions */
constexpr int NUM_WINDOW_INCREMENT = static_cast<int>((WINDOW_INCREMENT_MAX - WINDOW_INCREMENT_MIN) / WINDOW_INCREMENT_STEP) + 1;
constexpr int NUM_WINDOW_MULTIPLE  = static_cast<int>((WINDOW_MULTIPLE_MAX - WINDOW_MULTIPLE_MIN) / WINDOW_MULTIPLE_STEP) + 1;
constexpr int NUM_INTERSEND        = static_cast<int>((INTERSEND_MAX - INTERSEND_MIN) / INTERSEND_STEP) + 1;

struct ObsAction {
  std::array<double, INPUT_DIM> observation;
  int action_wi_idx;
  int action_wm_idx;
  int action_is_idx;
  float old_log_prob;
};

struct ActionResult {
  int the_window;
  double intersend_time;
  ObsAction obs_action;
};

struct PolicyNetImpl : torch::nn::Module {
  int _hidden_size;
  int _num_hidden_layers;

  /* Running observation normalization (registered buffers — saved/loaded automatically) */
  torch::Tensor obs_mean;
  torch::Tensor obs_var;
  torch::Tensor obs_count;

  torch::nn::Linear input_proj{nullptr};
  std::vector<torch::nn::Linear> hidden_layers;
  torch::nn::Linear policy_wi{nullptr}, policy_wm{nullptr}, policy_is{nullptr};

  PolicyNetImpl( int hidden_size, int num_hidden_layers );

  /* Update running mean/var with a batch of observations [N, INPUT_DIM] */
  void update_obs_stats( torch::Tensor batch );

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  forward( torch::Tensor x );

  std::shared_ptr<PolicyNetImpl> clone_network() const;
};

TORCH_MODULE(PolicyNet);

/* Free function: run inference on any PolicyNet (used by both RatBrain and NeuralRat) */
ActionResult infer_action( PolicyNet & net, const Memory & memory, int current_window );

class RatBrain {
private:
  TrainingConfig _config;
  torch::Device _device;
  PolicyNet _network;
  std::shared_ptr<torch::optim::AdamW> _optimizer;

  /* Replay buffer stored as flat tensors for vectorized batch indexing */
  torch::Tensor _buf_obs;          /* [replay_buffer_size, INPUT_DIM] float */
  torch::Tensor _buf_utility;      /* [replay_buffer_size] float */
  torch::Tensor _buf_old_log_prob; /* [replay_buffer_size] float */
  torch::Tensor _buf_action_wi;    /* [replay_buffer_size] long */
  torch::Tensor _buf_action_wm;    /* [replay_buffer_size] long */
  torch::Tensor _buf_action_is;    /* [replay_buffer_size] long */
  torch::Tensor _buf_total_events;     /* [replay_buffer_size] float — total events across all senders in the rollout */

  size_t _write_pos;
  size_t _buffer_count;

public:
  RatBrain( const TrainingConfig & config = TrainingConfig() );

  const PolicyNet & network() const { return _network; }
  void remember_episode( double utility, const std::vector<ObsAction> & observations, size_t total_rollout_events );
  void learn();
  void save( const std::string & filename ) const;
  void load( const std::string & filename );

  std::string str() const { return "(neural network)"; }
};

#endif
