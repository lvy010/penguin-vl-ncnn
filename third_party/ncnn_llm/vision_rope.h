#pragma once

#include <vector>

#include <mat.h>

void generate_vision_rope_cache_2d(int num_patches_h,
                                   int num_patches_w,
                                   int spatial_merge_size,
                                   float rope_theta,
                                   const std::vector<int>& rope_section,
                                   bool duplicate_sections,
                                   ncnn::Mat& cos_cache,
                                   ncnn::Mat& sin_cache);
