# UTF-8 Validation Benchmark Handoff

## Required checkpoint

Before results are integrated into the report, record the benchmark script or
commands, implementations, compiler and flags, CPU and SIMD features, operating
system, Parabix block size, and all raw results.

## Implementations

- `utf8validate --algorithm=direct`: byte-oriented IDISA implementation.
- `utf8validate --algorithm=pablo`: validation-only logic derived from `u8u16`.
- Keiser-Lemire: external hand-written implementation, with version and license recorded.
- Scalar strict UTF-8 validator: non-SIMD baseline.

Run `tests/run_utf8_validation_tests.py` before accepting timing results.

## Measurement contract

1. Use the same machine, corpora, compiler optimization level, and validity
   semantics for every implementation.
2. Report compilation/JIT and file I/O separately from steady-state validation.
3. Use large ASCII-heavy, multilingual, continuation-heavy, and invalid corpora.
4. Warm up each implementation, then collect at least 10 trials.
5. Measure one thread first; then 2, 4, and 8 threads, capped at physical cores.
6. Save raw CSV columns:
   `implementation,corpus,bytes,threads,trial,elapsed_seconds,gb_per_second`.
7. Report median GB/s, interquartile range, and speedup relative to one thread.
8. Deliver the CSV, exact commands/scripts, two charts, and a short factual
   interpretation. Do not discard outliers without documenting the rule.

## Report-ready outputs

- Grouped single-thread throughput chart by corpus.
- Thread-scaling chart with 1/2/4/8-thread points.
- Compact table of median throughput, variability, and relative speedup.
- Notes distinguishing algorithm cost, S2P/transposition cost, JIT/setup, I/O,
  and synchronization overhead.
