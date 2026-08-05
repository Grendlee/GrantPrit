#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/io/source_kernel.h>
#include <chrono>
#include <algorithm>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include "utf8_validation_kernel.h"
#include "utf8_pablo_validation_kernel.h"
#include "utf8_validation_lookup_kernel.h"
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

static cl::opt<unsigned> benchIters("bench", cl::desc("Benchmark: re-run the pipeline N times per file over an in-memory buffer and report median ns/byte."), cl::init(0), cl::cat(u8vOptions));
static cl::opt<bool> benchSamples("bench-samples", cl::desc("With -bench, print every individual trial so median and IQR can be recomputed from raw data."), cl::init(false), cl::cat(u8vOptions));
static cl::opt<std::string> algorithm("algorithm", cl::desc("Validation algorithm: direct, pablo, or lookup"),
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

static uint64_t endsMidSequence(const unsigned char * tail, size_t tailLen, uint64_t fileLength) {
    for (size_t back = 1; back <= tailLen && back <= 4; ++back) {
        const unsigned char c = tail[tailLen - back];
        if (c < 0x80) return 0;
        if (c < 0xC0) continue;
        size_t needed;
        if (c < 0xE0) needed = 2;
        else if (c < 0xF0) needed = 3;
        else needed = 4;

        return (back < needed) ? 1 : 0;
    }
    (void) fileLength;
    return 0;
}

static uint64_t eofErrorForBuffer(const char * data, size_t length) {
    if (length == 0) return 0;
    const size_t tailLen = (length < 4) ? length : 4;
    return endsMidSequence(reinterpret_cast<const unsigned char *>(data) + (length - tailLen),
                           tailLen, length);
}

template <typename PipelineT>
static void buildValidation(PipelineT & P, StreamSet * ByteStream, Scalar * fileIdx) {
    Scalar * errorCount = nullptr;
    if (algorithm == "direct") {
        Kernel * const validator = P.template CreateKernelCall<UTF8ValidationKernel>(ByteStream);
        errorCount = validator->getOutputScalarAt(0);
        P.CreateCall("record_result", record_result, {errorCount, fileIdx});
    } else if (algorithm == "lookup") {

        Kernel * const validator = P.template CreateKernelCall<UTF8ValidationLookupKernel>(ByteStream);
        errorCount = validator->getOutputScalarAt(0);
        P.CreateCall("record_result", record_result, {errorCount, fileIdx});
    } else if (algorithm == "pablo") {
        StreamSet * const BasisBits = P.CreateStreamSet(8);
        Selected_S2P(P, ByteStream, BasisBits);
        StreamSet * const Errors = P.CreateStreamSet(1);
        P.template CreateKernelCall<UTF8PabloValidationKernel>(BasisBits, Errors);
        errorCount = P.CreateScalar(P.getInt64Ty());
        P.template CreateKernelCall<UTF8ErrorCountKernel>(Errors, errorCount);
        P.CreateCall("record_result", record_result, {errorCount, fileIdx});
    } else {
        llvm::report_fatal_error("utf8validate: --algorithm must be direct, pablo, or lookup");
    }
}

auto pipelineGen(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>("fd"), Input<uint32_t>("fileIdx"));
    Scalar * const fileDescriptor = P.getInputScalar("fd");
    Scalar * const fileIdx = P.getInputScalar("fileIdx");
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    buildValidation(P, ByteStream, fileIdx);
    return P.compile();
}

auto pipelineGenMem(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<const char *>{"buffer"}, Input<size_t>{"length"},
                            Input<uint32_t>("fileIdx"));
    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");
    Scalar * const fileIdx = P.getInputScalar("fileIdx");
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);
    buildValidation(P, ByteStream, fileIdx);
    return P.compile();
}

typedef void (*ValidateMemFn)(const char * buffer, size_t length, uint32_t fileIdx);

static std::vector<char> slurp(const std::string & path) {
    std::vector<char> buf;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) return buf;
    struct stat st;
    if (fstat(fd, &st) == 0) {
        buf.resize(st.st_size);
        ssize_t got = 0;
        while (got < st.st_size) {
            const ssize_t n = read(fd, buf.data() + got, st.st_size - got);
            if (n <= 0) break;
            got += n;
        }
        buf.resize(got);
    }
    close(fd);

    const size_t block = 4096;
    const size_t padded = ((buf.size() + block - 1) / block) * block;
    buf.resize(padded, '\0');
    return buf;
}

static int runBench(CPUDriver & driver) {
    auto memFn = pipelineGenMem(driver);
    for (unsigned i = 0; i < inputFiles.size(); ++i) {
        struct stat st;
        const bool haveSize = (stat(inputFiles[i].c_str(), &st) == 0);
        std::vector<char> data = slurp(inputFiles[i]);
        if (data.empty()) {
            std::cout << inputFiles[i] << ": ERROR (could not read)\n";
            continue;
        }

        const double nbytes = haveSize ? double(st.st_size) : double(data.size());
        std::vector<double> samples;
        samples.reserve(benchIters);
        for (unsigned k = 0; k < benchIters; ++k) {
            const auto t0 = std::chrono::steady_clock::now();
            memFn(data.data(), data.size(), i);
            const auto t1 = std::chrono::steady_clock::now();
            samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / nbytes);
        }
        errorCounts[i] += eofErrorForBuffer(data.data(), size_t(nbytes));
        if (benchSamples) {
            for (unsigned k = 0; k < samples.size(); ++k) {
                std::cout << inputFiles[i] << "  trial=" << (k + 1)
                          << "  ns_per_byte=" << samples[k]
                          << "  gb_per_s=" << (1.0 / samples[k]) << "\n";
            }
        }
        std::sort(samples.begin(), samples.end());
        const double med = samples[samples.size() / 2];
        std::cout << inputFiles[i] << "  bytes=" << size_t(nbytes)
                  << "  " << (errorCounts[i] == 0 ? "VALID" : "INVALID")
                  << "  median_ns/byte=" << med << "  GB/s=" << (1.0 / med)
                  << "  (iters=" << benchIters << ")\n";
    }
    return 0;
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

    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        const size_t tailLen = (st.st_size < 4) ? size_t(st.st_size) : size_t(4);
        unsigned char tail[4];
        if (pread(fd, tail, tailLen, st.st_size - tailLen) == ssize_t(tailLen)) {
            errorCounts[fileIdx] += endsMidSequence(tail, tailLen, uint64_t(st.st_size));
        }
    }
    close(fd);
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&u8vOptions, codegen::codegen_flags()});
    CPUDriver driver("utf8validate");
    errorCounts.assign(inputFiles.size(), 0);
    opened.assign(inputFiles.size(), false);
    if (benchIters > 0) return runBench(driver);
    auto fn_ptr = pipelineGen(driver);
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
