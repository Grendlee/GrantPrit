#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class UTF8ValidationKernel final : public BlockOrientedKernel {
public:
    UTF8ValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * remainingItems) override;
};

}
