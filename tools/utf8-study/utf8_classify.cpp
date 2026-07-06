/*
 * Byte-oriented UTF-8 classification baseline for the Parabix application
 * study.  This is deliberately not a full Parabix implementation:
 *
 *  - Direct byte classification gives us a simple baseline.
 *  - Parabix bit streams may express sequence scopes and mismatch rules well.
 *  - Blindly translating every byte to eight basis streams may cost more than
 *    it saves, so a later prototype should measure S2P separately and consider
 *    a hybrid that leaves cheap classification at the byte/SIMD level.
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

enum class ByteClass : std::size_t {
    ASCII,
    Continuation,
    Leader2,
    Leader3,
    Leader4,
    Invalid,
    Count
};

struct Result {
    std::array<std::size_t, static_cast<std::size_t>(ByteClass::Count)> counts{};
    bool structurallyValid = true;
    std::size_t firstError = 0;
};

static ByteClass classify(const std::uint8_t byte) {
    if ((byte & 0x80U) == 0) return ByteClass::ASCII;            // 0xxxxxxx
    if ((byte & 0xC0U) == 0x80U) return ByteClass::Continuation; // 10xxxxxx
    if (byte >= 0xC2U && byte <= 0xDFU) return ByteClass::Leader2;
    if (byte >= 0xE0U && byte <= 0xEFU) return ByteClass::Leader3;
    if (byte >= 0xF0U && byte <= 0xF4U) return ByteClass::Leader4;
    return ByteClass::Invalid; // C0/C1 (overlong leaders) and F5--FF
}

static bool isContinuation(const std::uint8_t byte) {
    return classify(byte) == ByteClass::Continuation;
}

static Result inspect(const std::vector<std::uint8_t> &bytes) {
    Result result;

    // Count classes independently from validation so malformed inputs still
    // produce useful classification data for a future implementation.
    for (const std::uint8_t byte : bytes) {
        ++result.counts[static_cast<std::size_t>(classify(byte))];
    }

    for (std::size_t i = 0; i < bytes.size();) {
        const ByteClass kind = classify(bytes[i]);
        std::size_t width = 1;
        if (kind == ByteClass::Leader2) width = 2;
        else if (kind == ByteClass::Leader3) width = 3;
        else if (kind == ByteClass::Leader4) width = 4;
        else if (kind == ByteClass::Continuation || kind == ByteClass::Invalid) {
            result.structurallyValid = false;
            result.firstError = i;
            break;
        }

        if (i + width > bytes.size()) {
            result.structurallyValid = false;
            result.firstError = i; // truncated sequence
            break;
        }
        for (std::size_t offset = 1; offset < width; ++offset) {
            if (!isContinuation(bytes[i + offset])) {
                result.structurallyValid = false;
                result.firstError = i + offset;
                return result;
            }
        }

        // Minimal UTF-8 range rules: reject overlong forms, UTF-16 surrogates,
        // and code points above U+10FFFF. This remains a structural prototype.
        if ((kind == ByteClass::Leader3 && bytes[i] == 0xE0U && bytes[i + 1] < 0xA0U) ||
            (kind == ByteClass::Leader3 && bytes[i] == 0xEDU && bytes[i + 1] > 0x9FU) ||
            (kind == ByteClass::Leader4 && bytes[i] == 0xF0U && bytes[i + 1] < 0x90U) ||
            (kind == ByteClass::Leader4 && bytes[i] == 0xF4U && bytes[i + 1] > 0x8FU)) {
            result.structurallyValid = false;
            result.firstError = i;
            break;
        }
        i += width;
    }
    return result;
}

static std::vector<std::uint8_t> bytes(const std::string_view text) {
    return {text.begin(), text.end()};
}

static void printResult(const std::string_view name, const Result &result,
                        const std::size_t byteCount, const double microseconds) {
    const auto count = [&](const ByteClass c) { return result.counts[static_cast<std::size_t>(c)]; };
    std::cout << name << ": bytes=" << byteCount
              << " ascii=" << count(ByteClass::ASCII)
              << " continuation=" << count(ByteClass::Continuation)
              << " leader2=" << count(ByteClass::Leader2)
              << " leader3=" << count(ByteClass::Leader3)
              << " leader4=" << count(ByteClass::Leader4)
              << " invalid=" << count(ByteClass::Invalid)
              << " structural=" << (result.structurallyValid ? "PASS" : "FAIL");
    if (!result.structurallyValid) std::cout << " error_offset=" << result.firstError;
    std::cout << " time_us=" << std::fixed << std::setprecision(2) << microseconds << '\n';
}

struct TestCase {
    const char *name;
    std::vector<std::uint8_t> input;
    bool expectedValid;
};

static int runTests() {
    const std::array<TestCase, 5> tests{{
        {"valid ASCII", bytes("Hello, Parabix!"), true},
        {"valid multilingual", bytes("caf\xC3\xA9 \xE4\xB8\x96\xE7\x95\x8C \xF0\x9F\x8C\x8D"), true},
        {"invalid continuation", {0x41, 0x80, 0x42}, false},
        {"truncated sequence", {0xE2, 0x82}, false},
        {"overlong-style sequence", {0xC0, 0xAF}, false},
    }};

    bool allPassed = true;
    for (const TestCase &test : tests) {
        const auto start = std::chrono::steady_clock::now();
        const Result result = inspect(test.input);
        const auto stop = std::chrono::steady_clock::now();
        const double us = std::chrono::duration<double, std::micro>(stop - start).count();
        printResult(test.name, result, test.input.size(), us);
        if (result.structurallyValid != test.expectedValid) {
            std::cerr << "unexpected result for " << test.name << '\n';
            allPassed = false;
        }
    }
    std::cout << "tests=" << (allPassed ? "PASS" : "FAIL") << '\n';
    return allPassed ? 0 : 1;
}

static int inspectFile(const char *path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "could not open: " << path << '\n';
        return 2;
    }
    const std::vector<std::uint8_t> input{std::istreambuf_iterator<char>(file), {}};
    const auto start = std::chrono::steady_clock::now();
    const Result result = inspect(input);
    const auto stop = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(stop - start).count();
    printResult(path, result, input.size(), us);
    return result.structurallyValid ? 0 : 1;
}

int main(const int argc, char **argv) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--self-test")) {
        return runTests();
    }
    if (argc == 3 && std::string_view(argv[1]) == "--file") {
        return inspectFile(argv[2]);
    }
    std::cerr << "usage: utf8-classify [--self-test | --file PATH]\n";
    return 2;
}
