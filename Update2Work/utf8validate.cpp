#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/io/source_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include "utf8_validation_kernel.h"
#include "utf8_pablo_validation_kernel.h"
#include <llvm/Support/CommandLine.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>

using namespace llvm;
using namespace kernel;

static cl::OptionCategory u8vOptions("utf8validate Options", "UTF-8 validation options.");
static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<input file ...>"), cl::OneOrMore, cl::cat(u8vOptions));
static cl::opt<std::string> algorithm("algorithm", cl::desc("Validation algorithm: direct or pablo"),
                                     cl::init("direct"), cl::cat(u8vOptions));

static std::vector<uint64_t> errorCounts;
static std::vector<bool> opened;

extern "C" {
    void record_result(uint64_t errorCount, uint64_t fileIdx) {
        errorCounts[fileIdx] = errorCount;
    }
    void record_direct_result(uint64_t errorCount, uint64_t eofErrorCount, uint64_t fileIdx) {
        errorCounts[fileIdx] = errorCount + eofErrorCount;
    }
}

typedef void (*ValidateFunctionType)(uint32_t fd, uint32_t fileIdx);

auto pipelineGen(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>("fd"), Input<uint32_t>("fileIdx"));
    Scalar * const fileDescriptor = P.getInputScalar("fd");
    Scalar * const fileIdx = P.getInputScalar("fileIdx");
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    Scalar * errorCount = nullptr;
    if (algorithm == "direct") {
        Kernel * const validator = P.CreateKernelCall<UTF8ValidationKernel>(ByteStream);
        errorCount = validator->getOutputScalarAt(0);
        StreamSet * const EOFErrors = P.CreateStreamSet(1);
        P.CreateKernelCall<UTF8EOFKernel>(ByteStream, EOFErrors);
        Scalar * const eofErrorCount = P.CreateScalar(P.getInt64Ty());
        P.CreateKernelCall<UTF8ErrorCountKernel>(EOFErrors, eofErrorCount);
        P.CreateCall("record_direct_result", record_direct_result,
                     {errorCount, eofErrorCount, fileIdx});
    } else if (algorithm == "pablo") {
        StreamSet * const BasisBits = P.CreateStreamSet(8);
        Selected_S2P(P, ByteStream, BasisBits);
        StreamSet * const Errors = P.CreateStreamSet(1);
        P.CreateKernelCall<UTF8PabloValidationKernel>(BasisBits, Errors);
        errorCount = P.CreateScalar(P.getInt64Ty());
        P.CreateKernelCall<UTF8ErrorCountKernel>(Errors, errorCount);
    } else {
        llvm::report_fatal_error("utf8validate: --algorithm must be direct or pablo");
    }
    if (algorithm == "pablo") {
        P.CreateCall("record_result", record_result, {errorCount, fileIdx});
    }
    return P.compile();
}

void validate(ValidateFunctionType fn_ptr, const uint32_t fileIdx) {
    const std::string & fileName = inputFiles[fileIdx];
    const int fd = open(fileName.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        std::cerr << "utf8validate: cannot open " << fileName << "\n";
        return;
    }
    opened[fileIdx] = true;
    fn_ptr(fd, fileIdx);
    close(fd);
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&u8vOptions, codegen::codegen_flags()});
    CPUDriver driver("utf8validate");
    auto fn_ptr = pipelineGen(driver);
    errorCounts.assign(inputFiles.size(), 0);
    opened.assign(inputFiles.size(), false);
    for (unsigned i = 0; i < inputFiles.size(); ++i) {
        validate(fn_ptr, i);
    }
    int rc = 0;
    for (unsigned i = 0; i < inputFiles.size(); ++i) {
        if (!opened[i]) {
            std::cout << inputFiles[i] << ": ERROR (could not open)\n";
            rc = 1;
            continue;
        }
        const bool valid = (errorCounts[i] == 0);
        std::cout << inputFiles[i] << ": " << (valid ? "VALID" : "INVALID")
                  << " (errors=" << errorCounts[i] << ")\n";
        if (!valid) rc = 1;
    }
    return rc;
}
