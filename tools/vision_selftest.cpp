// SPDX-License-Identifier: Apache-2.0
//
// Integration self-test: load the pnnx-exported vision sub-graphs (real
// Penguin-VL-2B dims, random weights from tools/selftest_export.py) through the
// C++ PenguinVision driver and run the full patch_embed -> encoder(2D-RoPE) ->
// projector pipeline. Verifies the export<->runtime blob contract and shapes.
#include <cstdio>
#include <exception>
#include <limits>
#include <string>

#include <mat.h>

#include "penguin_config.h"
#include "penguin_vision.h"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/tmp/pvl_selftest";
    const int grid_h = 10, grid_w = 10;
    const int N = grid_h * grid_w;
    const int ps = 14;

    pvl::VisionConfig vc;
    vc.enabled = true;
    vc.patch_embed_param = "vision_patch_embed.ncnn.param";
    vc.patch_embed_bin = "vision_patch_embed.ncnn.bin";
    vc.encoder_param = "vision_encoder.ncnn.param";
    vc.encoder_bin = "vision_encoder.ncnn.bin";
    vc.projector_param = "vision_projector.ncnn.param";
    vc.projector_bin = "vision_projector.ncnn.bin";
    vc.patch_size = ps;
    vc.merge_size = 1;
    vc.vision_hidden = 1024;
    vc.vision_head_dim = 128;
    vc.vision_rope_theta = 1000000.0f;

    try {
        pvl::PenguinVision vision(dir, vc, 4, false);

        ncnn::Mat pixel_values(3 * ps * ps, N);
        for (int i = 0; i < N; ++i) {
            float* r = pixel_values.row(i);
            for (int j = 0; j < 3 * ps * ps; ++j) r[j] = ((i * 131 + j * 7) % 255) / 255.0f - 0.5f;
        }

        ncnn::Mat out = vision.encode(pixel_values, grid_h, grid_w);
        std::printf("vision output: w=%d h=%d (expect w=2048 h=%d)\n", out.w, out.h, N);

        bool ok = (out.w == 2048 && out.h == N);
        // sanity: output should be finite and not all-zero
        const float* p = out.row(0);
        float s = 0.f;
        for (int j = 0; j < out.w; ++j) s += p[j] * p[j];
        std::printf("row0 L2^2=%.4f\n", s);
        // Toolchain-independent finite check (avoids std::isfinite, which is
        // unreliable across some MSVC <cmath> configurations): s == s rejects
        // NaN, and the bound comparison rejects +/-inf.
        const bool finite = (s == s) && (s < std::numeric_limits<float>::infinity());
        ok = ok && (s > 0.f) && finite;

        std::printf(ok ? "SELFTEST PASS\n" : "SELFTEST FAIL\n");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("SELFTEST ERROR: %s\n", e.what());
        return 2;
    }
}
