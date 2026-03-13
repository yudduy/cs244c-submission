#ifndef EVOLVEDPOLICY_HH
#define EVOLVEDPOLICY_HH

#include <string>

/* Placeholder type for the Evaluator<EvolvedPolicy> template.
   The actual policy logic lives inside EvolvedRat::update_window_and_intersend(). */
struct EvolvedPolicy {
  std::string str() const { return std::string( "EvolvedPolicy(v5_best)" ); }
};

#endif
