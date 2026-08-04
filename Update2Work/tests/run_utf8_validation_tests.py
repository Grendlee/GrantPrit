#!/usr/bin/env python3
"""Generate UTF-8 boundary cases and compare both Parabix validators.

The trusted result is Python's strict UTF-8 decoder.  Test files live only in
a temporary directory so the repository does not accumulate opaque binaries.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    data: bytes

    @property
    def valid(self) -> bool:
        try:
            self.data.decode("utf-8", errors="strict")
            return True
        except UnicodeDecodeError:
            return False


def cases_for_block(block_bytes: int) -> list[Case]:
    cases = [
        Case("empty", b""),
        Case("one_ascii", b"A"),
        Case("valid_ascii", b"Hello, Parabix!"),
        Case("valid_multilingual", "cafe\u0301 \u4e16\u754c \U0001f30d".encode()),
        Case("stray_continuation", b"A\x80B"),
        Case("c0_leader", b"\xC0\xAF"),
        Case("c1_leader", b"\xC1\xBF"),
        Case("f5_leader", b"\xF5\x80\x80\x80"),
        Case("ff_byte", b"\xFF"),
        Case("overlong_3", b"\xE0\x80\x80"),
        Case("surrogate", b"\xED\xA0\x80"),
        Case("overlong_4", b"\xF0\x80\x80\x80"),
        Case("above_unicode", b"\xF4\x90\x80\x80"),
    ]
    valid_sequences = (b"\xC2\xA2", b"\xE2\x82\xAC", b"\xF0\x9F\x8C\x8D")
    truncated = (b"\xC2", b"\xE2", b"\xE2\x82", b"\xF0", b"\xF0\x9F", b"\xF0\x9F\x8C")

    for boundary in (block_bytes, 2 * block_bytes, 3 * block_bytes):
        for index, sequence in enumerate(valid_sequences, 2):
            prefix = b"A" * (boundary - 1)
            cases.append(Case(f"valid_{index}byte_crossing_{boundary}", prefix + sequence))
        for index, tail in enumerate(truncated):
            cases.append(Case(f"truncated_exact_{boundary}_{index}",
                              b"A" * (boundary - len(tail)) + tail))

    for boundary in (block_bytes, 2 * block_bytes):
        for length in range(boundary - 3, boundary + 4):
            for index, tail in enumerate(truncated):
                if length >= len(tail):
                    cases.append(Case(f"truncated_len_{length}_{index}",
                                      b"A" * (length - len(tail)) + tail))
    return cases


def run_algorithm(executable: Path, algorithm: str, block_bits: int,
                  paths: list[Path]) -> dict[str, bool]:
    command = [str(executable), f"--algorithm={algorithm}",
               f"-BlockSize={block_bits}", "-thread-num=1", *map(str, paths)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    results: dict[str, bool] = {}
    for line in completed.stdout.splitlines():
        if ": VALID " in line:
            results[line.split(": ", 1)[0]] = True
        elif ": INVALID " in line:
            results[line.split(": ", 1)[0]] = False
    if len(results) != len(paths):
        raise RuntimeError(
            f"{algorithm}/{block_bits}: expected {len(paths)} results, got {len(results)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, default=Path("build/bin/utf8validate"))
    parser.add_argument("--block-bits", type=int, nargs="+", default=[64, 128, 256, 512])
    args = parser.parse_args()
    executable = args.executable.resolve()

    failures: list[str] = []
    total = 0
    with tempfile.TemporaryDirectory(prefix="utf8-validation-") as directory:
        root = Path(directory)
        for block_bits in args.block_bits:
            generated = cases_for_block(block_bits // 8)
            case_paths: list[Path] = []
            expected: dict[str, bool] = {}
            for number, case in enumerate(generated):
                path = root / f"b{block_bits}_{number:03d}_{case.name}.bin"
                path.write_bytes(case.data)
                case_paths.append(path)
                expected[str(path)] = case.valid

            outputs = {
                algorithm: run_algorithm(executable, algorithm, block_bits, case_paths)
                for algorithm in ("direct", "pablo")
            }
            for path in case_paths:
                key = str(path)
                total += 1
                direct = outputs["direct"][key]
                pablo = outputs["pablo"][key]
                if direct != expected[key] or pablo != expected[key] or direct != pablo:
                    failures.append(
                        f"block={block_bits} case={path.name} expected={expected[key]} "
                        f"direct={direct} pablo={pablo}"
                    )

    if failures:
        print("FAIL")
        print("\n".join(failures))
        return 1
    print(f"PASS: {total} cases across {len(args.block_bits)} block sizes; "
          "direct, Pablo, and Python strict decoding agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
