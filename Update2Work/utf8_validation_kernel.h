#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class UTF8ValidationKernel final : public MultiBlockKernel {
public:
    UTF8ValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}
