#include <ctime>
#include <sys/types.h>
#include <unistd.h>

#include "random.hh"

PRNG & global_PRNG( void )
{
  static PRNG generator( time( NULL ) ^ getpid() );
  return generator;
}

void seed_global_PRNG( const unsigned int seed )
{
  global_PRNG().seed( seed );
}
