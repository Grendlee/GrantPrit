# Small byte-oriented baseline for the UTF-8/Parabix application study.
# It intentionally has no Parabix dependencies so its cost can later be
# compared with an S2P + bit-stream implementation.
add_executable(utf8-classify utf8_classify.cpp)

add_test(NAME utf8-classify-tests COMMAND utf8-classify --self-test)
