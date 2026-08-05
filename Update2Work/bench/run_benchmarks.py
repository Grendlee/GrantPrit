#!/usr/bin/env python3
"""UTF-8 validation benchmark harness.

Implements the measurement contract in Update2Work/BENCHMARK_HANDOFF.md:
identical machine/corpora/semantics for every implementation, fixed costs
reported separately from steady-state validation, warmup plus repeated
trials, median and IQR, raw CSV.

Method.  Every implementation is invoked the same way -- one process, read a
file, validate, print a verdict -- so all of them carry the same fixed costs
(process start, dynamic linking, Parabix pipeline setup and cached-object
load).  Each configuration is timed at several corpus sizes and a line is
fitted to time against bytes:

    elapsed = fixed + bytes / throughput

The intercept is the fixed cost and the slope gives steady-state throughput.
This is what separates JIT/setup/I-O from validation work without needing an
in-process timer that Parabix's CLI cannot expose.  A "readonly" implementation
that reads and sums the bytes prices I/O on the same footing.

Corpora are generated one family at a time and deleted after use: the full set
at every size would be several GB.

Usage:
    python3 Update2Work/bench/run_benchmarks.py --out results.csv
    python3 Update2Work/bench/run_benchmarks.py --quick     # smaller, faster
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024

# Families named by the handoff doc: ASCII-heavy, multilingual,
# continuation-heavy, and invalid.
FAMILY_UNITS: dict[str, bytes] = {
    "ascii": b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen jugs. ",
    "multilingual": "The quick brown fox. Le renard brun rapide. café naïve façade "
                    "世界 مرحبا \U0001f30d ".encode(),
    "continuation": "世界こんにちは你好مرحبا".encode(),
    "invalid": b"",  # built from multilingual, corrupted near the end
}


@dataclass(frozen=True)
class Config:
    implementation: str
    threads: int
    argv: tuple[str, ...]


@dataclass
class Row:
    implementation: str
    corpus: str
    nbytes: int
    threads: int
    trial: int
    elapsed_seconds: float
    gb_per_second: float


def build_configs(utf8validate: Path, driver: Path, block_bits: int,
                  thread_counts: list[int]) -> list[Config]:
    configs: list[Config] = []
    for algorithm in ("direct", "pablo", "lookup"):
        for threads in thread_counts:
            configs.append(Config(
                implementation=f"parabix-{algorithm}",
                threads=threads,
                argv=(str(utf8validate), f"--algorithm={algorithm}",
                      f"-BlockSize={block_bits}", f"-thread-num={threads}"),
            ))
    for impl in ("simdjson", "scalar", "readonly"):
        configs.append(Config(
            implementation=("keiser-lemire-simdjson" if impl == "simdjson" else impl),
            threads=1,
            argv=(str(driver), f"--impl={impl}"),
        ))
    return configs


def make_corpus(family: str, target_bytes: int, path: Path) -> int:
    """Write a deterministic corpus of at least target_bytes, ending on a
    character boundary.  The invalid family is corrupted in its final bytes so
    that every implementation still scans the whole input -- a validator that
    bails at the first bad byte would otherwise be timed doing no work."""
    if family == "invalid":
        unit = FAMILY_UNITS["multilingual"]
    else:
        unit = FAMILY_UNITS[family]
    reps = target_bytes // len(unit) + 1
    data = bytearray(unit * reps)
    if family == "invalid":
        # a lone continuation byte a few characters from the end
        data[-6:-5] = b"\x80"
    path.write_bytes(bytes(data))
    return len(data)


def verify_agreement(configs: list[Config], corpus: Path, expected_valid: bool,
                     family: str) -> list[str]:
    """The contract requires identical validity semantics before any timing is
    accepted."""
    problems: list[str] = []
    for config in configs:
        if config.implementation == "readonly":
            continue
        completed = subprocess.run([*config.argv, str(corpus)],
                                   capture_output=True, text=True)
        out = completed.stdout
        if ": VALID " in out:
            observed = True
        elif ": INVALID " in out:
            observed = False
        else:
            problems.append(f"{config.implementation}/t{config.threads} on {family}: "
                            f"no verdict (stderr: {completed.stderr.strip()[:200]})")
            continue
        if observed != expected_valid:
            problems.append(f"{config.implementation}/t{config.threads} on {family}: "
                            f"expected valid={expected_valid}, got {observed}")
    return problems


def time_config(config: Config, corpus: Path, nbytes: int, warmup: int,
                trials: int) -> list[float]:
    command = [*config.argv, str(corpus)]
    for _ in range(warmup):
        subprocess.run(command, capture_output=True)
    times: list[float] = []
    for _ in range(trials):
        start = time.perf_counter()
        subprocess.run(command, capture_output=True)
        times.append(time.perf_counter() - start)
    return times


def iqr(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    lower = ordered[:mid]
    upper = ordered[-mid:]
    return statistics.median(upper) - statistics.median(lower)


def fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares fit of elapsed = fixed + bytes/throughput.

    Returns (fixed_seconds, gb_per_second).  Throughput is inf if the slope is
    non-positive, which means the sizes were too small to separate signal from
    fixed cost."""
    n = len(points)
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    denom = sum((p[0] - mean_x) ** 2 for p in points)
    if denom == 0:
        return (mean_y, float("inf"))
    slope = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denom
    fixed = mean_y - slope * mean_x
    throughput = (1.0 / slope / 1e9) if slope > 0 else float("inf")
    return (fixed, throughput)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utf8validate", type=Path, default=Path("build/bin/utf8validate"))
    parser.add_argument("--driver", type=Path, default=Path("Update2Work/bench/lemire_bench"))
    parser.add_argument("--out", type=Path, default=Path("Update2Work/bench/results.csv"))
    parser.add_argument("--summary", type=Path, default=None,
                        help="where to write the summary JSON (default: alongside --out)")
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/utf8-bench-corpora"))
    parser.add_argument("--block-bits", type=int, default=128,
                        help="128 is the only width with a byte shuffle on NEON")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--families", nargs="+", default=list(FAMILY_UNITS))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--quick", action="store_true",
                        help="smaller corpora and fewer trials, for a smoke test")
    args = parser.parse_args()

    if args.quick:
        args.sizes_mb = [4, 16, 48]
        args.trials = 5
    if args.summary is None:
        args.summary = args.out.with_suffix(".summary.json")

    utf8validate = args.utf8validate.resolve()
    driver = args.driver.resolve()
    for tool, label in ((utf8validate, "utf8validate"), (driver, "lemire_bench")):
        if not tool.exists():
            print(f"error: {label} not found at {tool}", file=sys.stderr)
            return 2

    configs = build_configs(utf8validate, driver, args.block_bits, args.threads)
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip(),
        "block_bits": args.block_bits,
        "trials": args.trials,
        "warmup": args.warmup,
        "sizes_mb": args.sizes_mb,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    print(json.dumps(machine, indent=2))

    rows: list[Row] = []
    # implementation -> threads -> family -> [(bytes, median_seconds)]
    curves: dict[tuple[str, int, str], list[tuple[int, float]]] = {}
    problems: list[str] = []

    try:
        for family in args.families:
            expected_valid = family != "invalid"
            print(f"\n=== family: {family} ===", flush=True)
            for size_mb in args.sizes_mb:
                corpus = args.scratch / f"{family}_{size_mb}mb.bin"
                nbytes = make_corpus(family, size_mb * MB, corpus)
                if size_mb == args.sizes_mb[0]:
                    problems.extend(verify_agreement(configs, corpus, expected_valid, family))
                for config in configs:
                    times = time_config(config, corpus, nbytes, args.warmup, args.trials)
                    median = statistics.median(times)
                    for trial, elapsed in enumerate(times, 1):
                        rows.append(Row(config.implementation, family, nbytes,
                                        config.threads, trial, elapsed,
                                        nbytes / elapsed / 1e9))
                    key = (config.implementation, config.threads, family)
                    curves.setdefault(key, []).append((nbytes, median))
                    print(f"  {config.implementation:24s} t={config.threads} "
                          f"{size_mb:4d}MB median={median*1000:8.2f}ms "
                          f"raw={nbytes/median/1e9:6.3f} GB/s", flush=True)
                corpus.unlink(missing_ok=True)
    finally:
        shutil.rmtree(args.scratch, ignore_errors=True)

    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["implementation", "corpus", "bytes", "threads", "trial",
                         "elapsed_seconds", "gb_per_second"])
        for row in rows:
            writer.writerow([row.implementation, row.corpus, row.nbytes, row.threads,
                             row.trial, f"{row.elapsed_seconds:.6f}",
                             f"{row.gb_per_second:.4f}"])

    summary: dict = {"machine": machine, "steady_state": {}, "problems": problems}
    print("\n=== steady-state throughput (fixed cost removed by regression) ===")
    print(f"{'implementation':26s} {'thr':>3} {'family':14s} {'fixed_ms':>9} {'GB/s':>8}")
    for (impl, threads, family), points in sorted(curves.items()):
        fixed, throughput = fit(points)
        summary["steady_state"].setdefault(impl, {}).setdefault(str(threads), {})[family] = {
            "fixed_ms": round(fixed * 1000, 3),
            "gb_per_second": (None if throughput == float("inf") else round(throughput, 4)),
            "points": [[n, round(t, 6)] for n, t in points],
        }
        shown = "n/a" if throughput == float("inf") else f"{throughput:8.3f}"
        print(f"{impl:26s} {threads:3d} {family:14s} {fixed*1000:9.2f} {shown:>8}")

    # thread scaling relative to one thread, per implementation and family
    scaling: dict = {}
    for (impl, threads, family), points in curves.items():
        _, throughput = fit(points)
        if throughput != float("inf"):
            scaling.setdefault(impl, {}).setdefault(family, {})[threads] = throughput
    summary["scaling"] = {
        impl: {fam: {str(t): round(v / by_t[1], 4) for t, v in sorted(by_t.items())}
               for fam, by_t in fams.items() if 1 in by_t}
        for impl, fams in scaling.items()
    }
    if summary["scaling"]:
        print("\n=== thread scaling (speedup vs 1 thread) ===")
        for impl, fams in sorted(summary["scaling"].items()):
            for family, speedups in sorted(fams.items()):
                pretty = "  ".join(f"{t}t={s:.2f}x" for t, s in sorted(speedups.items()))
                print(f"{impl:26s} {family:14s} {pretty}")

    args.summary.write_text(json.dumps(summary, indent=2))
    print(f"\nraw CSV : {args.out}  ({len(rows)} rows)")
    print(f"summary : {args.summary}")
    if problems:
        print("\n!! validity-semantics problems (timings are not trustworthy):")
        for problem in problems:
            print("   " + problem)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
