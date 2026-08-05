#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class UTF8ValidationLookupKernel final : public MultiBlockKernel {
public:
    UTF8ValidationLookupKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}
