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
