// SPDX-License-Identifier: Apache-2.0
// Validate the C++ byte-level BPE tokenizer against HuggingFace reference IDs
// using the real Penguin-VL tokenizer extracted by tools/extract_tokenizer.py.
#include <cstdio>
#include <string>
#include <vector>

#include "bpe_tokenizer.h"

static bool check(const BpeTokenizer& t, const std::string& s, const std::vector<int>& expect) {
    std::vector<int> got = t.encode(s, false, false);
    bool ok = (got == expect);
    std::printf("%-32s -> [", s.c_str());
    for (size_t i = 0; i < got.size(); ++i) std::printf("%s%d", i ? ", " : "", got[i]);
    std::printf("]  %s\n", ok ? "OK" : "MISMATCH");
    if (!ok) {
        std::printf("    expected [");
        for (size_t i = 0; i < expect.size(); ++i) std::printf("%s%d", i ? ", " : "", expect[i]);
        std::printf("]\n");
    }
    return ok;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/pvl_tok";
    BpeTokenizer t = BpeTokenizer::LoadFromFiles(
        dir + "/vocab.txt", dir + "/merges.txt", SpecialTokensConfig{},
        /*add_special_if_missing=*/false, /*fallback_to_chars=*/true,
        /*use_byte_encoder=*/true);
    std::printf("vocab_size = %zu\n", t.vocab_size());

    bool ok = true;
    ok &= check(t, "Hello, world!", {9707, 11, 1879, 0});
    ok &= check(t, "The quick brown fox.", {785, 3974, 13876, 38835, 13});
    ok &= check(t, "Describe this image in detail.", {74785, 419, 2168, 304, 7716, 13});
    std::printf(ok ? "TOKENIZER SELFTEST PASS\n" : "TOKENIZER SELFTEST FAIL\n");
    return ok ? 0 : 1;
}
