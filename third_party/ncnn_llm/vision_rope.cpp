#include "vision_rope.h"

#include <cmath>
#include <stdexcept>

static std::vector<float> compute_inv_freq_for_axis(int dim, int rope_dim, float theta) {
    std::vector<float> inv_freq(dim);
    for (int i = 0; i < dim; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(i * 2) / (float)rope_dim);
    }
    return inv_freq;
}

void generate_vision_rope_cache_2d(int num_patches_h,
                                   int num_patches_w,
                                   int spatial_merge_size,
                                   float rope_theta,
                                   const std::vector<int>& rope_section,
                                   bool duplicate_sections,
                                   ncnn::Mat& cos_cache,
                                   ncnn::Mat& sin_cache) {
    if (rope_section.size() != 2 || rope_section[0] <= 0 || rope_section[1] <= 0) {
        throw std::runtime_error("vision rope section must be [h_dim, w_dim]");
    }
    if (rope_theta <= 0.0f) {
        throw std::runtime_error("vision rope theta must be positive");
    }

    const int h_dim = rope_section[0];
    const int w_dim = rope_section[1];
    const int rope_dim = h_dim + w_dim;
    const int output_dim = duplicate_sections ? rope_dim * 2 : rope_dim;
    const int seq_len = num_patches_h * num_patches_w;

    std::vector<float> inv_freq_h = compute_inv_freq_for_axis(h_dim, rope_dim, rope_theta);
    std::vector<float> inv_freq_w = compute_inv_freq_for_axis(w_dim, rope_dim, rope_theta);

    cos_cache.create(output_dim, seq_len);
    sin_cache.create(output_dim, seq_len);

    int idx = 0;
    const int grid_h = num_patches_h / spatial_merge_size;
    const int grid_w = num_patches_w / spatial_merge_size;

    for (int gh = 0; gh < grid_h; gh++) {
        for (int gw = 0; gw < grid_w; gw++) {
            for (int mh = 0; mh < spatial_merge_size; mh++) {
                for (int mw = 0; mw < spatial_merge_size; mw++) {
                    const int current_h = gh * spatial_merge_size + mh;
                    const int current_w = gw * spatial_merge_size + mw;

                    float* cos_ptr = cos_cache.row(idx);
                    float* sin_ptr = sin_cache.row(idx);

                    for (int i = 0; i < h_dim; i++) {
                        const float angle_h = (float)current_h * inv_freq_h[i];
                        const float ch = std::cos(angle_h);
                        const float sh = std::sin(angle_h);
                        cos_ptr[i] = ch;
                        sin_ptr[i] = sh;
                        if (duplicate_sections) {
                            cos_ptr[rope_dim + i] = ch;
                            sin_ptr[rope_dim + i] = sh;
                        }
                    }

                    for (int i = 0; i < w_dim; i++) {
                        const float angle_w = (float)current_w * inv_freq_w[i];
                        const float cw = std::cos(angle_w);
                        const float sw = std::sin(angle_w);
                        cos_ptr[h_dim + i] = cw;
                        sin_ptr[h_dim + i] = sw;
                        if (duplicate_sections) {
                            cos_ptr[rope_dim + h_dim + i] = cw;
                            sin_ptr[rope_dim + h_dim + i] = sw;
                        }
                    }

                    idx++;
                }
            }
        }
    }
}
