#include "utf8_validation_lookup_kernel.h"
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Constants.h>
#include <array>

using namespace llvm;

namespace kernel {

UTF8ValidationLookupKernel::UTF8ValidationLookupKernel(LLVMTypeSystemInterface & ts, StreamSet * byteStream)
: MultiBlockKernel(ts, "UTF8ValidationLookup",
{Binding{"byteStream", byteStream}},
{},
{},
{Binding{ts.getSizeTy(), "errorCount"}},
{}) {
    // same carry trick as the compare kernel - remember the last block so a
    // character split across the boundary still gets checked
    addInternalScalar(ts.getBitBlockType(), "pending");
}

// The three Keiser-Lemire lookup tables. Each entry is the set of error-flag
// bits that a given nibble is consistent with; ANDing the three lookups leaves
// only the bits that ALL three agree on, which is exactly the real errors.
// Flags: TOO_SHORT=0x01 TOO_LONG=0x02 OVERLONG_3=0x04 TOO_LARGE=0x08
//        SURROGATE=0x10 OVERLONG_2=0x20 TOO_LARGE_1000/OVERLONG_4=0x40 TWO_CONTS=0x80
// CARRY = TOO_SHORT|TOO_LONG|TWO_CONTS = 0x83.
static const std::array<uint8_t, 16> T1_byte1_high = {
    0x02,0x02,0x02,0x02, 0x02,0x02,0x02,0x02,   // 0-7 ASCII lead -> TOO_LONG
    0x80,0x80,0x80,0x80,                         // 8-B continuation -> TWO_CONTS
    0x21,                                        // C  TOO_SHORT|OVERLONG_2
    0x01,                                        // D  TOO_SHORT
    0x15,                                        // E  TOO_SHORT|OVERLONG_3|SURROGATE
    0x49                                         // F  TOO_SHORT|TOO_LARGE|TOO_LARGE_1000|OVERLONG_4
};
static const std::array<uint8_t, 16> T2_byte1_low = {
    0xE7,0xA3,0x83,0x83, 0x8B,0xCB,0xCB,0xCB,
    0xCB,0xCB,0xCB,0xCB, 0xCB,0xDB,0xCB,0xCB
};
static const std::array<uint8_t, 16> T3_byte2_high = {
    0x01,0x01,0x01,0x01, 0x01,0x01,0x01,0x01,   // 0-7 ASCII -> TOO_SHORT
    0xE6,0xAE,0xBA,0xBA,                         // 8/9/A/B continuation subranges
    0x01,0x01,0x01,0x01                          // C-F lead -> TOO_SHORT
};

void UTF8ValidationLookupKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const loop = b.CreateBasicBlock("u8vl_loop");
    BasicBlock * const exit = b.CreateBasicBlock("u8vl_exit");

    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();
    Value * const i32Zero = b.getInt32(0);
    ConstantInt * const ZERO = b.getSize(0);
    ConstantInt * const ONE = b.getSize(1);
    ConstantInt * const THREE = b.getSize(3);
    ConstantInt * const SEVEN = b.getSize(7);

    const unsigned blocksPerStride = (getStride() * 8) / b.getBitBlockWidth();
    const unsigned lanes = b.getBitBlockWidth() / 8;

    // Build a table constant as a full-width byte vector. The 16-entry table is
    // repeated across each 128-bit lane, which is what NEON tbl / SSE pshufb want.
    // Wider builders (AVX2, AVX-512) index the whole vector rather than per lane,
    // but every index we produce is a nibble, so only the first 16 entries are
    // ever selected and the replication is harmless either way.
    auto makeTable = [&](const std::array<uint8_t, 16> & t) -> Value * {
        SmallVector<Constant *, 64> vals(lanes);
        for (unsigned i = 0; i < lanes; ++i) {
            vals[i] = ConstantInt::get(b.getInt8Ty(), t[i % 16]);
        }
        return ConstantVector::get(vals);
    };
    Value * const T1 = makeTable(T1_byte1_high);
    Value * const T2 = makeTable(T2_byte1_low);
    Value * const T3 = makeTable(T3_byte2_high);

    Value * const v0F = b.simd_fill(8, b.getInt8(0x0F));
    Value * const v80 = b.simd_fill(8, b.getInt8(0x80));
    Value * const vE0 = b.simd_fill(8, b.getInt8(0xE0));
    Value * const vF0 = b.simd_fill(8, b.getInt8(0xF0));
    Value * const zeroVec = Constant::getNullValue(b.fwVectorType(8));

    // Read the carry once here and write it once at exit.  Doing this per block
    // instead loses the carry between kernel invocations, so every sequence
    // straddling a stride boundary is misreported.
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

    Value * const packIndex = b.CreateAnd(blockNo, SEVEN);
    Value * const blockOffset = b.CreateLShr(blockNo, THREE);
    Value * const currBlock = b.loadInputStreamPack("byteStream", i32Zero, packIndex, blockOffset);
    Value * const curr = b.fwCast(8, currBlock);
    Value * const prev = b.fwCast(8, pending);

    Value * const prev1 = b.mvmd_dslli(8, curr, prev, 1);
    Value * const prev2 = b.mvmd_dslli(8, curr, prev, 2);
    Value * const prev3 = b.mvmd_dslli(8, curr, prev, 3);

    // three nibble lookups, ANDed together -> all the special-case errors
    Value * const hiPrev1 = b.simd_srli(8, prev1, 4);
    Value * const loPrev1 = b.simd_and(prev1, v0F);
    Value * const hiCurr = b.simd_srli(8, curr, 4);
    Value * const b1h = b.mvmd_shuffle(8, T1, hiPrev1);
    Value * const b1l = b.mvmd_shuffle(8, T2, loPrev1);
    Value * const b2h = b.mvmd_shuffle(8, T3, hiCurr);
    Value * const sc = b.simd_and(b.simd_and(b1h, b1l), b2h);

    // a byte that must be the 3rd/4th of a sequence: prev2 is a 3-byte lead
    // (>=E0) or prev3 is a 4-byte lead (>=F0). Keep only the continuation bit.
    Value * must23 = b.simd_or(b.simd_uge(8, prev2, vE0), b.simd_uge(8, prev3, vF0));
    must23 = b.simd_and(must23, v80);

    // XOR: cancels the expected two-continuation bit, leaves any real error
    Value * const errVec = b.simd_xor(must23, sc);

    // a lane is bad if any error bit is set; turn that into a per-byte boolean
    Value * const errBool = b.simd_not(b.simd_eq(8, errVec, zeroVec));
    Value * const mask = b.hsimd_signmask(8, errBool);
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
