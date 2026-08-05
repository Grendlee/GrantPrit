#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

// Lookup-table version of the validator.  Same job as UTF8ValidationKernel,
// but instead of a long chain of range compares it does the Keiser-Lemire
// trick: three 16-entry table lookups on byte nibbles whose AND names every
// special-case error at once.  The point is to see how close a portable
// Parabix kernel gets to the hand-written implementation.
//
// MultiBlockKernel, not BlockOrientedKernel, and that choice is load-bearing.
// The previous block is carried in a "pending" scalar so a sequence split
// across a block boundary is still checked.  Reading that scalar once at entry
// and writing it once at exit keeps the carry alive across kernel invocations;
// doing the same get/set per block inside a BlockOrientedKernel does not, and
// silently drops the carry at every stride boundary.  See
// tests/run_lookup_validation_tests.py --stride-crossing for the regression
// that pins this down.
//
// Requires a byte-shuffle primitive, so mvmd_shuffle(8, ...) must exist for
// the target block width: 128 (pshufb / NEON tbl1), 256 (AVX2), or 512
// (AVX-512 VBMI/BW).  There is no 64-bit-block byte shuffle.
class UTF8ValidationLookupKernel final : public MultiBlockKernel {
public:
    UTF8ValidationLookupKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}
