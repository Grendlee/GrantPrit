#include "utf8_validation_kernel.h"
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>

using namespace llvm;

namespace kernel {

UTF8ValidationKernel::UTF8ValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream)
: BlockOrientedKernel(ts, "UTF8Validation",
{Binding{"byteStream", byteStream}},
{},
{},
{Binding{ts.getSizeTy(), "errorCount"}},
{}) {
    addInternalScalar(ts.getBitBlockType(), "pending");
}

void UTF8ValidationKernel::generateDoBlockMethod(KernelBuilder & b) {

    Type * const sizeTy = b.getSizeTy();
    Value * const i32Zero = b.getInt32(0);

    Value * const v80 = b.simd_fill(8, b.getInt8(0x80));
    Value * const v8F = b.simd_fill(8, b.getInt8(0x8F));
    Value * const v90 = b.simd_fill(8, b.getInt8(0x90));
    Value * const v9F = b.simd_fill(8, b.getInt8(0x9F));
    Value * const vA0 = b.simd_fill(8, b.getInt8(0xA0));
    Value * const vC0 = b.simd_fill(8, b.getInt8(0xC0));
    Value * const vC1 = b.simd_fill(8, b.getInt8(0xC1));
    Value * const vE0 = b.simd_fill(8, b.getInt8(0xE0));
    Value * const vED = b.simd_fill(8, b.getInt8(0xED));
    Value * const vF0 = b.simd_fill(8, b.getInt8(0xF0));
    Value * const vF4 = b.simd_fill(8, b.getInt8(0xF4));
    Value * const vF5 = b.simd_fill(8, b.getInt8(0xF5));

    Value * const currBlock = b.loadInputStreamBlock("byteStream", i32Zero);
    Value * const curr = b.fwCast(8, currBlock);
    Value * const prev = b.fwCast(8, b.getScalarField("pending"));

    // each pos gets the byte 1/2/3 back, carrying in the tail of the last block
    Value * const prev1 = b.mvmd_dslli(8, curr, prev, 1);
    Value * const prev2 = b.mvmd_dslli(8, curr, prev, 2);
    Value * const prev3 = b.mvmd_dslli(8, curr, prev, 3);

    Value * const isCont = b.simd_and(b.simd_uge(8, curr, v80), b.simd_ult(8, curr, vC0));
    Value * mustCont = b.simd_uge(8, prev1, vC0);
    mustCont = b.simd_or(mustCont, b.simd_uge(8, prev2, vE0));
    mustCont = b.simd_or(mustCont, b.simd_uge(8, prev3, vF0));

    Value * err = b.simd_xor(isCont, mustCont);

    // Tight second-byte ranges reject overlong encodings, UTF-16 surrogates,
    // and code points above U+10FFFF. prev1 carries across SIMD blocks.
    Value * rangeErr = b.simd_and(b.simd_eq(8, prev1, vE0), b.simd_ult(8, curr, vA0));
    Value * const surrogateErr = b.simd_and(b.simd_eq(8, prev1, vED), b.simd_ugt(8, curr, v9F));
    rangeErr = b.simd_or(rangeErr, surrogateErr);
    Value * const f0OverlongErr = b.simd_and(b.simd_eq(8, prev1, vF0), b.simd_ult(8, curr, v90));
    rangeErr = b.simd_or(rangeErr, f0OverlongErr);
    Value * const outOfRangeErr = b.simd_and(b.simd_eq(8, prev1, vF4), b.simd_ugt(8, curr, v8F));
    rangeErr = b.simd_or(rangeErr, outOfRangeErr);
    err = b.simd_or(err, rangeErr);

    // bytes that can never appear in valid utf-8
    Value * illegal = b.simd_eq(8, curr, vC0);
    illegal = b.simd_or(illegal, b.simd_eq(8, curr, vC1));
    illegal = b.simd_or(illegal, b.simd_uge(8, curr, vF5));
    err = b.simd_or(err, illegal);

    Value * const mask = b.hsimd_signmask(8, err);
    Value * const blockErrs = b.CreateZExtOrTrunc(b.CreatePopcount(mask), sizeTy);
    b.setScalarField("errorCount", b.CreateAdd(b.getScalarField("errorCount"), blockErrs));
    b.setScalarField("pending", currBlock);
}

void UTF8ValidationKernel::generateFinalBlockMethod(KernelBuilder & b, Value * /* remainingItems */) {
    // Always process the zero-padded terminal block.  When EOF is exactly on a
    // block boundary this is a synthetic all-zero block; pending carries the
    // preceding bytes into it, exposing any required continuations after EOF.
    RepeatDoBlockLogic(b);
}

}
