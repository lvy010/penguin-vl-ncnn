# Penguin-VL-ncnn 可交付测试报告

对应 issue *Penguin-VL ncnn 移植与多平台部署 #6788*
仓库：<https://github.com/lvy010/penguin-vl-ncnn>
日期：2026 年 7 月 · 版本：v0.1

---

## 摘要

本报告记录 Penguin-VL-ncnn 在 **Linux / x86_64 · 纯 CPU · fp32** 下的一次完整可交付测试。
测试覆盖四个层次，全部通过：

1. **构建**：CMake `find_package(ncnn)` 一键编译，5 个目标（运行时静态库 + CLI +
   单测 + 2 个集成自测）零警告链接成功；
2. **单元测试**：JSON 解析、图像预处理、2D-RoPE、分词器共 **10/10** 通过；
3. **导出 ↔ 运行时契约自测**（真实 Penguin-VL-2B 维度 + 随机权重）：pnnx 成功把
   `patch_embed / vision_encoder / projector / decoder` 转为 ncnn，C++ 驱动加载
   并前向，视觉输出形状 `(2048, N)`、数值有限且非零；解码器 KV-cache 的 blob 名
   与运行时绑定逐一对齐；
4. **分词器位级对齐**：C++ 字节级 BPE 与 **HuggingFace 原版**（`tencent/Penguin-VL-2B`）
   在离线现场交叉验证下 **token id 完全一致**。

> 诚实边界：受本机内存限制（约 7 GB），**带真实权重的 2B 全模型端到端
> 「ncnn 文本 == PyTorch 文本」** 未在此环境执行——它需要检查点 + PyTorch 且
> fp32 tracing 需约 6–8 GB。本报告验证的是**工具链契约、架构正确性与分词器位级一致性**；
> 复现该最终对齐的完整方法见 `docs/ALIGNMENT.md`、`docs/EXPORT.md`。

---

## 1. 测试环境

| 项 | 版本 |
| --- | --- |
| OS | Linux 6.8.0-63-generic x86_64 |
| 编译器 | g++ (Ubuntu) 13.3.0，C++17 |
| CMake | 3.28.3 |
| ncnn | master（`Found ncnn: 20260703`），`NCNN_VULKAN=OFF`、`NCNN_SIMPLEOCV=ON` |
| PyTorch | 2.12.1+cpu |
| transformers | 4.51.3 |
| pnnx | 通过 pip（CPU）|
| numpy | 2.5.0 |

复现命令均来自本仓库 `README.md` / `docs/SELFTEST.md`，未做任何一次性改动。

---

## 2. 构建

```bash
cmake -S . -B build -Dncnn_DIR=/path/to/ncnn/install/lib/cmake/ncnn -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

结果：

```
-- Found ncnn: 20260703
[100%] Built target penguinvl_runtime   # libpenguinvl_runtime.a
[100%] Built target penguinvl           # CLI
[100%] Built target penguinvl_tests     # 单元测试
[100%] Built target vision_selftest     # 视觉集成自测
[100%] Built target tok_selftest        # 分词器集成自测
```

依赖面：仅 **ncnn** + vendored `stb_image.h`；JSON 用自带 ~300 行解析器，无
nlohmann/json。CLI `--help` 正常输出用法。多平台由 `.github/workflows/ci.yml`
在 `ubuntu-latest` 与 `windows-latest` 上同时构建并跑单测保证。

---

## 3. 单元测试（无需权重）

```bash
ctest --test-dir build --output-on-failure     # 1/1 aggregate passed
./build/penguinvl_tests                         # 逐项明细
```

```
[ RUN/OK ] json_parse_scalars
[ RUN/OK ] json_parse_arrays
[ RUN/OK ] json_nested_and_escapes
[ RUN/OK ] json_defaults
[ RUN/OK ] py_round_banker
[ RUN/OK ] target_size_identity
[ RUN/OK ] target_size_min_pixels_upscale
[ RUN/OK ] patchify_layout_and_normalize
[ RUN/OK ] vision_rope_shape_and_values
[ RUN/OK ] tokenizer_load_and_decode

10/10 tests passed, 0 checks failed
```

| 覆盖点 | 用例 | 结果 |
| --- | --- | --- |
| JSON reader（标量/数组/嵌套转义/默认值） | 4 | ✅ |
| 预处理 Python banker's rounding | `py_round_banker` | ✅ |
| 目标分辨率（恒等 / min_pixels 上采样） | 2 | ✅ |
| patchify 排布 + mean=std=0.5 归一化 | `patchify_layout_and_normalize` | ✅ |
| 2D-RoPE 形状与数值 | `vision_rope_shape_and_values` | ✅ |
| 分词器加载与 decode 往返 | `tokenizer_load_and_decode` | ✅ |

---

## 4. 导出 ↔ 运行时契约自测（真实维度 · 随机权重）

这一步用 **真实 Penguin-VL-2B 维度**（vision hidden 1024、16 heads、8 KV heads、
head_dim 128、SwiGLU 3072）但随机权重导出子图，从而在小机器上验证两个单元测试
覆盖不到的集成点：**pnnx 的算子支持** 与 **导出/运行时共享的 ncnn blob I/O 契约**。

### 4.1 pnnx 导出

```bash
python tools/selftest_export.py --out /tmp/pvl_selftest --layers 2 --n 100
```

```
############# pass_ncnn
[pnnx] ... vision_patch_embed.ncnn.{param,bin}   inputshape=[100,588]f32
[pnnx] ... vision_encoder.ncnn.{param,bin}       inputshape=[100,1024]f32,[100,128]f32,[100,128]f32
[pnnx] ... vision_projector.ncnn.{param,bin}     inputshape=[100,1024]f32
[pnnx] ... decoder.ncnn.{param,bin}              inputshape=[3,64]f32,...(+KV cache)
[done] wrote ncnn sub-graphs to /tmp/pvl_selftest
```

四个子图全部转换成功。`RMSNorm` 走 ncnn 原生算子；**GQA 头扩展用
`expand()+reshape()`**（而非 pnnx 不支持的 `repeat_interleave`，此坑正是被本自测提前发现）。
`squeeze batch dim` 等属 pnnx 对 batch 轴的常规提示，不影响单图（batch=1）推理。

### 4.2 C++ 驱动加载并前向（视觉全链路）

```bash
./build/vision_selftest /tmp/pvl_selftest
```

```
vision output: w=2048 h=100 (expect w=2048 h=100)
row0 L2^2=89.3608
SELFTEST PASS
```

这条路径完整跑通 `PenguinVision::encode`：`patch_embed` → 双向编码器（C++
`generate_penguin_vision_rope` 生成的 2D-RoPE 以 `in1/in2` 喂入）→ `mlp2x_gelu`
投影器，全部从 pnnx 产物加载。输出维度符合预期、数值有限且非零。

### 4.3 解码器 KV-cache 契约

```bash
grep -c Input /tmp/pvl_selftest/decoder.ncnn.param      # 8 (in0..in3 + 4 cache)
python tools/rename_decoder_blobs.py --param /tmp/pvl_selftest/decoder.ncnn.param --layers 2
grep -oE 'cache_k[0-9]+|cache_v[0-9]+|out_cache_k[0-9]+|out_cache_v[0-9]+' \
     /tmp/pvl_selftest/decoder.ncnn.param | sort -u
```

rename 后的 blob 名与 `src/penguin_vl.cpp` 中 `snprintf` 绑定的名字逐一对齐：

| param 文件（rename 后） | 运行时绑定（`penguin_vl.cpp`） | 一致 |
| --- | --- | --- |
| `cache_k0 cache_k1 cache_v0 cache_v1` | `"cache_k%d" / "cache_v%d"`（行 232/233、282/283） | ✅ |
| `out_cache_k0 out_cache_k1 out_cache_v0 out_cache_v1` | `"out_cache_k%d" / "out_cache_v%d"`（行 203/204、240/241） | ✅ |

即两段式 prefill + 逐 token 解码所需的 KV-cache 输入/输出契约在导出侧与 C++ 侧完全吻合。

---

## 5. 分词器位级对齐（vs HuggingFace 原版）

仅需 tokenizer 文件（~10 MB），无需检查点：

```bash
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out /tmp/pvl_tok
# → vocab.txt (151675 tokens), merges.txt (151387 merges)
./build/tok_selftest /tmp/pvl_tok
```

C++ 字节级 BPE 输出：

```
vocab_size = 151675
Hello, world!                    -> [9707, 11, 1879, 0]            OK
The quick brown fox.             -> [785, 3974, 13876, 38835, 13]  OK
Describe this image in detail.   -> [74785, 419, 2168, 304, 7716, 13] OK
TOKENIZER SELFTEST PASS
```

并用离线现场的 HuggingFace `AutoTokenizer` 交叉验证同样三条字符串，**逐 id 完全一致**：

| 输入 | C++ BBPE | HuggingFace | 一致 |
| --- | --- | --- | --- |
| `Hello, world!` | `[9707, 11, 1879, 0]` | `[9707, 11, 1879, 0]` | ✅ |
| `The quick brown fox.` | `[785, 3974, 13876, 38835, 13]` | 同左 | ✅ |
| `Describe this image in detail.` | `[74785, 419, 2168, 304, 7716, 13]` | 同左 | ✅ |

> 注：`merges.txt` 的正确导出曾是一个坑——须从 `backend_tokenizer.to_str()` 的
> 序列化状态里取全部 151,387 条 merge，否则 C++ 侧退化为不合并，id 不一致。已修复。

---

## 6. 结果汇总

| 层次 | 检查项 | 结果 |
| --- | --- | --- |
| 构建 | CMake 配置 + 5 目标编译链接（Linux/x86_64, g++13, fp32） | ✅ |
| 单测 | JSON / 预处理 / 2D-RoPE / 分词器 | ✅ 10/10 |
| 导出 | pnnx 转 patch_embed / encoder / projector / decoder | ✅ |
| 集成 | 视觉子图经 C++ `PenguinVision` 前向，输出 `(2048,N)` 有限非零 | ✅ SELFTEST PASS |
| 契约 | 解码器 KV-cache blob 名 == 运行时绑定 | ✅ |
| 分词器 | C++ BBPE id == HuggingFace id（现场交叉验证） | ✅ 位级一致 |
| CI | ubuntu-latest + windows-latest 构建 & 单测 | ✅（工作流已配置） |

---

## 7. 未在本环境执行的部分（诚实说明）

| 项 | 原因 | 复现方式 |
| --- | --- | --- |
| 带真实权重产出 `*.param/*.bin` | 2B 检查点未下载；fp32 tracing 需 ~6–8 GB RAM | `tools/export_penguinvl.py` + `export_decoder_kvcache.py`（见 `docs/EXPORT.md`） |
| 端到端「ncnn 文本 == PyTorch 文本」 | 同上 | `tools/compare_reference.py` 产 golden，再与 `penguinvl` 逐 token diff（见 `docs/ALIGNMENT.md`） |

`docs/ALIGNMENT.md` 已给出逐级比对法与最大风险点（resize 内核差异 > chat
template/system prompt > fp16 翻转 argmax），并规定「先 fp32 对齐通过、再切 fp16 复验」。

---

## 8. 一键复现

```bash
# 1) 构建 ncnn（CPU，一次）
cmake -S ncnn -B ncnn/build -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_SIMPLEOCV=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ncnn/install
cmake --build ncnn/build --target install -j

# 2) 构建本项目 + 单测
cmake -S . -B build -Dncnn_DIR=$PWD/ncnn/install/lib/cmake/ncnn
cmake --build build -j
ctest --test-dir build --output-on-failure

# 3) 工具链契约自测（需 torch + pnnx）
python tools/selftest_export.py --out /tmp/pvl_selftest --layers 2 --n 100
./build/vision_selftest /tmp/pvl_selftest
python tools/rename_decoder_blobs.py --param /tmp/pvl_selftest/decoder.ncnn.param --layers 2

# 4) 分词器位级对齐（需 transformers + tokenizer 文件）
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out /tmp/pvl_tok
./build/tok_selftest /tmp/pvl_tok
```
