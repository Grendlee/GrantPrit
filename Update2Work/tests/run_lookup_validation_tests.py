#!/usr/bin/env python3
"""Differential test for the Keiser-Lemire lookup validator.

Reuses the case generator and runner from run_utf8_validation_tests.py so the
lookup kernel is judged against exactly the same corpus as direct and Pablo,
with Python's strict UTF-8 decoder as the oracle.

Block width 64 is excluded by default: the lookup algorithm needs a byte-shuffle
primitive, and mvmd_shuffle(8, ...) has no 64-bit-block implementation (see
IDISA_Builder::mvmd_shuffle).  128/256/512 cover pshufb, AVX2, NEON tbl1, and
AVX-512 VBMI.
"""

from __future__ import annotations

import argparse
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_utf8_validation_tests import Case, cases_for_block, run_algorithm

ALGORITHMS = ("direct", "pablo", "lookup")


def shuffle_widths() -> list[int]:
    """Block widths whose byte shuffle the current machine can actually build.

    NEON is a 128-bit ISA, so the ARM builder implements mvmd_shuffle(8, ...)
    only at that width.  x86 covers 128 (pshufb), 256 (AVX2), and 512
    (AVX-512 VBMI/BW).
    """
    if platform.machine().lower() in ("arm64", "aarch64"):
        return [128]
    return [128, 256, 512]


def stride_crossing_cases(block_bits: int) -> list[Case]:
    """Valid sequences straddling a stride boundary (8 blocks), not just a block.

    The generated suite in run_utf8_validation_tests.py tops out at three blocks,
    so it never reaches a stride boundary.  A kernel that drops its pending-block
    carry between invocations passes that suite and still misreports every
    multibyte sequence landing on a stride edge, which is most of a real corpus.
    """
    stride = block_bits  # bytes: a stride is 8 blocks of block_bits/8 bytes
    cases: list[Case] = []
    for sequence, label in ((b"\xC2\xA2", "2byte"),
                            (b"\xE4\xB8\x96", "3byte"),
                            (b"\xF0\x9F\x8C\x8D", "4byte")):
        for boundary in (stride, 2 * stride):
            for offset in range(1, len(sequence)):
                start = boundary - offset
                data = b"A" * start + sequence
                data += b"A" * (3 * stride - len(data))
                cases.append(Case(f"{label}_crossing_stride_{boundary}_off{offset}", data))
    # a long run of a sequence whose period does not divide the stride
    cases.append(Case("long_3byte_run", "世".encode() * (stride * 4)))
    cases.append(Case("long_mixed_run", ("A" + "\U0001f30d").encode() * (stride * 2)))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/bin/utf8validate"))
    parser.add_argument("--block-bits", type=int, nargs="+", default=None)
    parser.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    parser.add_argument("--stride-crossing", action="store_true",
                        help="only run the stride-boundary regression cases")
    args = parser.parse_args()
    executable = args.executable.resolve()
    if args.block_bits is None:
        args.block_bits = shuffle_widths()
        print(f"block widths for {platform.machine()}: {args.block_bits}")

    failures: list[str] = []
    total = 0
    skipped: list[int] = []
    per_algorithm_failures = {algorithm: 0 for algorithm in args.algorithms}

    with tempfile.TemporaryDirectory(prefix="utf8-lookup-") as directory:
        root = Path(directory)
        for block_bits in args.block_bits:
            generated = (stride_crossing_cases(block_bits) if args.stride_crossing
                         else cases_for_block(block_bits // 8) + stride_crossing_cases(block_bits))
            case_paths: list[Path] = []
            expected: dict[str, bool] = {}
            for number, case in enumerate(generated):
                path = root / f"b{block_bits}_{number:03d}_{case.name}.bin"
                path.write_bytes(case.data)
                case_paths.append(path)
                expected[str(path)] = case.valid

            try:
                outputs = {
                    algorithm: run_algorithm(executable, algorithm, block_bits, case_paths)
                    for algorithm in args.algorithms
                }
            except RuntimeError as error:
                if "mvmd_shuffle" in str(error):
                    print(f"SKIP block={block_bits}: no byte shuffle at this width "
                          f"on {platform.machine()}")
                    skipped.append(block_bits)
                    continue
                raise
            for path in case_paths:
                key = str(path)
                total += 1
                observed = {a: outputs[a][key] for a in args.algorithms}
                wrong = [a for a, v in observed.items() if v != expected[key]]
                for algorithm in wrong:
                    per_algorithm_failures[algorithm] += 1
                if wrong:
                    detail = " ".join(f"{a}={observed[a]}" for a in args.algorithms)
                    failures.append(
                        f"block={block_bits} case={path.name} "
                        f"expected={expected[key]} {detail}"
                    )

    tested = [w for w in args.block_bits if w not in skipped]
    cases_per_width = total // len(tested) if tested else 0
    if failures:
        print("FAIL")
        print("\n".join(failures[:40]))
        if len(failures) > 40:
            print(f"... and {len(failures) - 40} more")
        print("\nmismatches per algorithm: " + ", ".join(
            f"{a}={per_algorithm_failures[a]}/{total}" for a in args.algorithms))
        return 1

    if not tested:
        print("ERROR: every requested block width was skipped")
        return 1
    print(f"PASS: {total} cases ({cases_per_width} per width) across block sizes "
          f"{tested}; {', '.join(args.algorithms)} and Python strict "
          "decoding all agree")
    if skipped:
        print(f"skipped (no byte shuffle at this width): {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
