#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

// Byte-oriented compare-chain validator.
//
// MultiBlockKernel rather than BlockOrientedKernel: the previous block is
// carried in a "pending" scalar, and that carry has to be read once at entry
// and written once at exit to survive across kernel invocations.  Reading and
// writing it per block inside a BlockOrientedKernel loses it at every stride
// boundary, which silently misreports any multibyte sequence landing there.
// See tests/run_lookup_validation_tests.py --stride-crossing.
class UTF8ValidationKernel final : public MultiBlockKernel {
public:
    UTF8ValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}
