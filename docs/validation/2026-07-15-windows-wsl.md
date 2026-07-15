# Windows 11 + WSL2 validation (2026-07-15)

This transcript records an independent clean build and export-contract run for
Tencent/ncnn#6788. It was produced from commit `b4d0cd6` plus the portable pnnx
self-test changes in this branch.

## Environment

```text
Host: Windows 11 x86_64, Intel Core i9-12950HX, 31.6 GiB RAM
Linux: WSL2 Ubuntu 24.04, GCC 13.3.0, CMake 3.28, Ninja 1.11
Windows native: MSYS2 UCRT64, GCC 16.1.0, CMake 4.4, Ninja 1.13
ncnn: 1.0.20260715, CPU build, Vulkan disabled
pnnx: 20260526
Python: 3.12 isolated venv
```

## Linux build and CTest

```text
$ cmake -S . -B /home/zhangzherui/build/penguin-vl-ncnn -G Ninja \
    -Dncnn_DIR=/home/zhangzherui/install/ncnn/lib/cmake/ncnn
$ cmake --build /home/zhangzherui/build/penguin-vl-ncnn -j 8
[23/23] Linking CXX executable penguinvl_tests
$ ctest --test-dir /home/zhangzherui/build/penguin-vl-ncnn --output-on-failure
1/1 Test #1: penguinvl_tests .................. Passed
100% tests passed, 0 tests failed out of 1
```

## Native Windows build and CTest

```text
-- The CXX compiler identification is GNU 16.1.0
-- Found ncnn: 20260715
[23/23] Linking CXX executable penguinvl.exe
Test project C:/Users/zhangzherui/Desktop/Tencent/penguin-vl-ncnn-build
1/1 Test #1: penguinvl_tests .................. Passed
100% tests passed out of 1
```

## Explicit-pnnx export contract

```text
$ python tools/selftest_export.py \
    --out /home/zhangzherui/outputs/penguin-selftest100 \
    --layers 2 --n 100 \
    --pnnx /home/zhangzherui/.venvs/hunyuan-ocr/bin/pnnx
[pnnx] .../pnnx .../vision_patch_embed.pt inputshape=[100,588]f32 ...
[pnnx] .../pnnx .../vision_encoder.pt inputshape=[100,1024]f32,[100,128]f32,[100,128]f32 ...
[pnnx] .../pnnx .../vision_projector.pt inputshape=[100,1024]f32 ...
[pnnx] .../pnnx .../decoder.pt inputshape=[3,64]f32,[3,7]f32,...
[done] wrote ncnn sub-graphs to /home/zhangzherui/outputs/penguin-selftest100

$ /home/zhangzherui/build/penguin-vl-ncnn/vision_selftest \
    /home/zhangzherui/outputs/penguin-selftest100
vision output: w=2048 h=100 (expect w=2048 h=100)
row0 L2^2=80.6278
SELFTEST PASS
```

The explicit `--pnnx` path proves the exporter uses the selected isolated
environment instead of whichever executable happens to appear first on PATH.
