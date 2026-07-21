#include "utf8_validation_kernel.h"
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>

using namespace llvm;

namespace kernel {

UTF8ValidationKernel::UTF8ValidationKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream)
: MultiBlockKernel(ts, "UTF8Validation",
{Binding{"byteStream", byteStream}},
{},
{},
{Binding{ts.getSizeTy(), "errorCount"}},
{}) {
    addInternalScalar(ts.getBitBlockType(), "pending");
}

void UTF8ValidationKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const loop = b.CreateBasicBlock("u8v_loop");
    BasicBlock * const exit = b.CreateBasicBlock("u8v_exit");

    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();
    Value * const i32Zero = b.getInt32(0);
    ConstantInt * const ZERO = b.getSize(0);
    ConstantInt * const ONE = b.getSize(1);
    ConstantInt * const THREE = b.getSize(3);
    ConstantInt * const SEVEN = b.getSize(7);

    const unsigned blocksPerStride = (getStride() * 8) / b.getBitBlockWidth();

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

    Value * const numBlocks = b.CreateMul(numOfStrides, b.getSize(blocksPerStride));
    Value * const startPending = b.getScalarField("pending");
    Value * const startCount = b.getScalarField("errorCount");

    b.CreateBr(loop);
    b.SetInsertPoint(loop);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(ZERO, entry);
    PHINode * const pending = b.CreatePHI(blockTy, 2);
    pending->addIncoming(startPending, entry);
    PHINode * const count = b.CreatePHI(sizeTy, 2);
    count->addIncoming(startCount, entry);

    // loadInputStreamBlock strides by 8 blocks, so index sub-blocks by pack
    Value * const packIndex = b.CreateAnd(blockNo, SEVEN);
    Value * const blockOffset = b.CreateLShr(blockNo, THREE);
    Value * const currBlock = b.loadInputStreamPack("byteStream", i32Zero, packIndex, blockOffset);
    Value * const curr = b.fwCast(8, currBlock);
    Value * const prev = b.fwCast(8, pending);

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
    Value * const nextCount = b.CreateAdd(count, blockErrs);

    Value * const nextBlock = b.CreateAdd(blockNo, ONE);
    blockNo->addIncoming(nextBlock, loop);
    pending->addIncoming(currBlock, loop);
    count->addIncoming(nextCount, loop);
    b.CreateCondBr(b.CreateICmpULT(nextBlock, numBlocks), loop, exit);

    b.SetInsertPoint(exit);
    PHINode * const finalPending = b.CreatePHI(blockTy, 1);
    finalPending->addIncoming(currBlock, loop);
    PHINode * const finalCount = b.CreatePHI(sizeTy, 1);
    finalCount->addIncoming(nextCount, loop);
    b.setScalarField("pending", finalPending);
    b.setScalarField("errorCount", finalCount);
}

}
