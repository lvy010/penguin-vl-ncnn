# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.15)
project(penguin_vl_ncnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(PENGUINVL_BUILD_TESTS "Build unit tests" ON)

# --- ncnn -------------------------------------------------------------------
# Provide the location of ncnnConfig.cmake via -Dncnn_DIR=<prefix>/lib/cmake/ncnn
find_package(ncnn REQUIRED)

# --- MSVC / Windows niceties ------------------------------------------------
if(MSVC)
    add_compile_options(/utf-8)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# --- runtime library --------------------------------------------------------
set(PVL_SRC
    src/json_min.cpp
    src/penguin_config.cpp
    src/image_preprocess.cpp
    src/penguin_vision.cpp
    src/penguin_vl.cpp
    third_party/ncnn_llm/bpe_tokenizer.cpp
    third_party/ncnn_llm/rope_embed.cpp
    third_party/ncnn_llm/vision_rope.cpp
    third_party/ncnn_llm/image_utils.cpp
    third_party/ncnn_llm/sampling.cpp
)

add_library(penguinvl_runtime STATIC ${PVL_SRC})
target_include_directories(penguinvl_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ncnn_llm
)
target_link_libraries(penguinvl_runtime PUBLIC ncnn)

# --- CLI executable ---------------------------------------------------------
add_executable(penguinvl src/main.cpp)
target_link_libraries(penguinvl PRIVATE penguinvl_runtime)

# --- vision integration self-test -------------------------------------------
# Runs the pnnx-exported vision sub-graphs through the C++ driver. Generate the
# graphs first with `python tools/selftest_export.py` (needs torch + pnnx), then
# run `./vision_selftest /tmp/pvl_selftest`. See docs/SELFTEST.md.
add_executable(vision_selftest tools/vision_selftest.cpp)
target_link_libraries(vision_selftest PRIVATE penguinvl_runtime)

# Validates the C++ BBPE tokenizer against HuggingFace IDs using the real
# tokenizer extracted by tools/extract_tokenizer.py. See docs/SELFTEST.md.
add_executable(tok_selftest tools/tok_selftest.cpp)
target_link_libraries(tok_selftest PRIVATE penguinvl_runtime)

# --- tests ------------------------------------------------------------------
if(PENGUINVL_BUILD_TESTS)
    enable_testing()
    add_executable(penguinvl_tests
        tests/test_main.cpp
        tests/test_json.cpp
        tests/test_preprocess.cpp
        tests/test_rope2d.cpp
        tests/test_tokenizer.cpp
    )
    target_link_libraries(penguinvl_tests PRIVATE penguinvl_runtime)
    add_test(NAME penguinvl_tests COMMAND penguinvl_tests)
endif()
