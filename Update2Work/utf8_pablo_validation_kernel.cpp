#include "utf8_pablo_validation_kernel.h"

#include <pablo/builder.hpp>
#include <pablo/pe_count.h>
#include <re/alphabet/alphabet.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>

using namespace pablo;
using namespace re;

namespace kernel {

UTF8PabloValidationKernel::UTF8PabloValidationKernel(
        LLVMTypeSystemInterface & ts, StreamSet * basisBits, StreamSet * errors)
: PabloKernel(ts, "UTF8PabloValidation",
              {Binding{"basis", basisBits}},
              {Binding{"errors", errors, FixedRate(), Add1()}}) {
}

void UTF8PabloValidationKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    const std::vector<PabloAST *> u8 = getInputStreamSet("basis");
    cc::Parabix_CC_Compiler_Builder ccc(u8);

    PabloAST * const suffix = ccc.compileCC(makeByte(0x80, 0xBF), pb);
    PabloAST * const pfx2 = ccc.compileCC(makeByte(0xC0, 0xDF), pb);
    PabloAST * const pfx3 = ccc.compileCC(makeByte(0xE0, 0xEF), pb);
    PabloAST * const pfx4 = ccc.compileCC(makeByte(0xF0, 0xFF), pb);

    PabloAST * const scope22 = pb.createAdvance(pfx2, 1);
    PabloAST * const scope32 = pb.createAdvance(pfx3, 1);
    PabloAST * const scope33 = pb.createAdvance(scope32, 1);
    PabloAST * const scope42 = pb.createAdvance(pfx4, 1);
    PabloAST * const scope43 = pb.createAdvance(scope42, 1);
    PabloAST * const scope44 = pb.createAdvance(scope43, 1);
    PabloAST * const anyScope = pb.createOr(scope22,
        pb.createOr(scope32, pb.createOr(scope33,
        pb.createOr(scope42, pb.createOr(scope43, scope44)))));

    // Compare only real byte positions.  A separate Add1-position error marks
    // an expected continuation exactly at EOF.
    PabloAST * errors = pb.createXor(pb.createInFile(anyScope), suffix);
    errors = pb.createOr(errors, ccc.compileCC(makeByte(0xC0, 0xC1), pb));
    errors = pb.createOr(errors, ccc.compileCC(makeByte(0xF5, 0xFF), pb));

    PabloAST * const e0 = ccc.compileCC(makeByte(0xE0), pb);
    PabloAST * const ed = ccc.compileCC(makeByte(0xED), pb);
    PabloAST * const f0 = ccc.compileCC(makeByte(0xF0), pb);
    PabloAST * const f4 = ccc.compileCC(makeByte(0xF4), pb);
    errors = pb.createOr(errors, pb.createAnd(pb.createAdvance(e0, 1),
                                               ccc.compileCC(makeByte(0x80, 0x9F), pb)));
    errors = pb.createOr(errors, pb.createAnd(pb.createAdvance(ed, 1),
                                               ccc.compileCC(makeByte(0xA0, 0xBF), pb)));
    errors = pb.createOr(errors, pb.createAnd(pb.createAdvance(f0, 1),
                                               ccc.compileCC(makeByte(0x80, 0x8F), pb)));
    errors = pb.createOr(errors, pb.createAnd(pb.createAdvance(f4, 1),
                                               ccc.compileCC(makeByte(0x90, 0xBF), pb)));
    errors = pb.createOr(errors, pb.createAtEOF(anyScope), "errorsWithEOF");

    Var * const output = getOutputStreamVar("errors");
    pb.createAssign(pb.createExtract(output, pb.getInteger(0)), errors);
}

UTF8EOFKernel::UTF8EOFKernel(LLVMTypeSystemInterface & ts, StreamSet * bytes,
                             StreamSet * errors)
: PabloKernel(ts, "UTF8EOF",
              {Binding{"bytes", bytes}},
              {Binding{"errors", errors, FixedRate(), Add1()}}) {
}

void UTF8EOFKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * const bytes = pb.createExtract(getInputStreamVar("bytes"), pb.getInteger(0));
    cc::Direct_CC_Compiler ccc(bytes);
    PabloAST * const pfx2 = ccc.compileCC(makeByte(0xC2, 0xDF), pb);
    PabloAST * const pfx3 = ccc.compileCC(makeByte(0xE0, 0xEF), pb);
    PabloAST * const pfx4 = ccc.compileCC(makeByte(0xF0, 0xF4), pb);
    PabloAST * const scope22 = pb.createAdvance(pfx2, 1);
    PabloAST * const scope32 = pb.createAdvance(pfx3, 1);
    PabloAST * const scope33 = pb.createAdvance(scope32, 1);
    PabloAST * const scope42 = pb.createAdvance(pfx4, 1);
    PabloAST * const scope43 = pb.createAdvance(scope42, 1);
    PabloAST * const scope44 = pb.createAdvance(scope43, 1);
    PabloAST * const anyScope = pb.createOr(scope22,
        pb.createOr(scope32, pb.createOr(scope33,
        pb.createOr(scope42, pb.createOr(scope43, scope44)))));
    Var * const output = getOutputStreamVar("errors");
    pb.createAssign(pb.createExtract(output, pb.getInteger(0)), pb.createAtEOF(anyScope));
}

UTF8ErrorCountKernel::UTF8ErrorCountKernel(LLVMTypeSystemInterface & ts,
        StreamSet * errors, Scalar * countResult)
: PabloKernel(ts, "UTF8ErrorCount",
              {Binding{"errors", errors, FixedRate(), Add1()}}, {}, {},
              {Binding{"countResult", countResult}}) {
}

void UTF8ErrorCountKernel::generatePabloMethod() {
    PabloBlock * const pb = getEntryScope();
    PabloAST * const errors = pb->createExtract(getInputStreamVar("errors"), pb->getInteger(0));
    pb->createAssign(getOutputScalarVar("countResult"), pb->createCount(errors));
}

}
