

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace {

bool scalar_validate(const unsigned char * data, size_t length) {
    size_t i = 0;
    while (i < length) {
        const unsigned char c = data[i];
        if (c < 0x80) { i += 1; continue; }
        size_t needed;
        unsigned char lo = 0x80, hi = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) {
            needed = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            needed = 2;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            needed = 3;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        } else {
            return false;
        }
        if (i + needed >= length) return false;
        for (size_t k = 1; k <= needed; ++k) {
            const unsigned char cc = data[i + k];
            const unsigned char lowBound = (k == 1) ? lo : 0x80;
            const unsigned char highBound = (k == 1) ? hi : 0xBF;
            if (cc < lowBound || cc > highBound) return false;
        }
        i += needed + 1;
    }
    return true;
}

bool read_file(const char * path, std::vector<char> & out) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return false; }
    out.resize(static_cast<size_t>(st.st_size));
    size_t got = 0;
    while (got < out.size()) {
        const ssize_t n = read(fd, out.data() + got, out.size() - got);
        if (n <= 0) break;
        got += static_cast<size_t>(n);
    }
    close(fd);
    return got == out.size();
}

}

int run_bench(const std::string & impl, const char * path, unsigned iters, bool emitSamples) {
    std::vector<char> buffer;
    if (!read_file(path, buffer)) {
        std::printf("%s: ERROR (could not read)\n", path);
        return 1;
    }
    const double nbytes = static_cast<double>(buffer.size());
    const unsigned char * bytes = reinterpret_cast<const unsigned char *>(buffer.data());
    std::vector<double> samples;
    samples.reserve(iters);
    bool valid = true;
    for (unsigned k = 0; k < iters; ++k) {
        const auto t0 = std::chrono::steady_clock::now();
        if (impl == "simdjson") {
            valid = simdjson::validate_utf8(buffer.data(), buffer.size());
        } else if (impl == "scalar") {
            valid = scalar_validate(bytes, buffer.size());
        } else {
            unsigned long long sum = 0;
            for (size_t i = 0; i < buffer.size(); i += 4096) sum += bytes[i];
            valid = (sum != 0xFFFFFFFFFFFFFFFFULL);
        }
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / nbytes);
    }
    if (emitSamples) {
        for (size_t k = 0; k < samples.size(); ++k) {
            std::printf("%s  trial=%zu  ns_per_byte=%g  gb_per_s=%g\n",
                        path, k + 1, samples[k], 1.0 / samples[k]);
        }
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    std::printf("%s  bytes=%zu  %s  median_ns/byte=%g  GB/s=%g  (iters=%u)\n",
                path, buffer.size(), valid ? "VALID" : "INVALID",
                median, 1.0 / median, iters);
    return 0;
}

int main(int argc, char ** argv) {
    std::string impl = "simdjson";
    unsigned iters = 0;
    bool emitSamples = false;
    std::vector<const char *> files;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--impl=", 7) == 0) {
            impl = argv[i] + 7;
        } else if (std::strcmp(argv[i], "--samples") == 0) {
            emitSamples = true;
        } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
            iters = static_cast<unsigned>(std::atoi(argv[i] + 8));
        } else {
            files.push_back(argv[i]);
        }
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: lemire_bench [--impl=simdjson|scalar|readonly] <file ...>\n");
        return 2;
    }

    if (iters > 0) {
        int rc = 0;
        for (const char * path : files) rc |= run_bench(impl, path, iters, emitSamples);
        return rc;
    }

    int rc = 0;
    for (const char * path : files) {
        std::vector<char> buffer;
        if (!read_file(path, buffer)) {
            std::printf("%s: ERROR (could not read)\n", path);
            rc = 1;
            continue;
        }
        bool valid;
        if (impl == "simdjson") {
            valid = simdjson::validate_utf8(buffer.data(), buffer.size());
        } else if (impl == "scalar") {
            valid = scalar_validate(reinterpret_cast<const unsigned char *>(buffer.data()),
                                    buffer.size());
        } else if (impl == "readonly") {

            unsigned long long sum = 0;
            for (size_t i = 0; i < buffer.size(); ++i) sum += static_cast<unsigned char>(buffer[i]);
            valid = (sum != 0xFFFFFFFFFFFFFFFFULL);
        } else {
            std::fprintf(stderr, "lemire_bench: unknown --impl=%s\n", impl.c_str());
            return 2;
        }
        std::printf("%s: %s (errors=%d)\n", path, valid ? "VALID" : "INVALID", valid ? 0 : 1);
        if (!valid) rc = 1;
    }
    return rc;
}
