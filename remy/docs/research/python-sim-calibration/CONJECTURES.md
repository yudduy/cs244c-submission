# Conjectures

## Confirmed
### C-001: Delay parameter is the dominant error source
- **Statement**: The delay_ms = rtt_ms/2 bug causes >50% of the calibration error
- **Kill criterion**: Fixing it reduces mean error by <10% (relative)
- **Result**: Fixing reduced error from 68-127% to 6.6% — confirmed dominant
- **Status**: CONFIRMED

### C-002: Send interleaving matters at high link rates
- **Statement**: Sending one packet at a time with network processing between sends will fix the 60-70% error at high link rates for the 10x tree
- **Kill criterion**: Error at 37.7/59.8/94.9 Mbps remains >20% after fix
- **Result**: Error dropped from 60-70% to 0.2-1.5%
- **Status**: CONFIRMED

## Killed
### C-003: PRNG differences explain remaining ~3% error
- **Statement**: Using the same PRNG (mt19937 with same seed) would reduce error to <1%
- **Kill criterion**: Not tested yet (would require C++ PRNG reimplementation)
- **Status**: UNTESTED — likely true but not worth implementing
