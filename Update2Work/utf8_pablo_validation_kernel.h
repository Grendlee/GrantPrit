#pragma once

#include <pablo/pablo_kernel.h>

namespace kernel {

class UTF8PabloValidationKernel final : public pablo::PabloKernel {
public:
    UTF8PabloValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * basisBits,
                              StreamSet * errors);
private:
    void generatePabloMethod() override;
};

// UNUSED.  Kept for reference only -- do not add back to a pipeline.  Its
// Add1() output stream needs one item more than the input, which the pipeline
// only accounts for within a single segment; past roughly 192 KiB it overruns
// its buffer and corrupts memory.  EOF truncation is now detected by
// UTF8EOFDeficitKernel, which uses in-file counts only.
class UTF8EOFKernel final : public pablo::PabloKernel {
public:
    UTF8EOFKernel(LLVMTypeSystemInterface & ts, StreamSet * bytes, StreamSet * errors);
private:
    void generatePabloMethod() override;
};

class UTF8ErrorCountKernel final : public pablo::PabloKernel {
public:
    UTF8ErrorCountKernel(LLVMTypeSystemInterface & ts, StreamSet * errors,
                         Scalar * countResult);
private:
    void generatePabloMethod() override;
};

}
