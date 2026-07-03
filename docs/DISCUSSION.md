# 将 Penguin-VL 移植到 ncnn：一次多模态大模型的端侧部署实践

> 技术总结（GitHub Discussion 草稿），对应 issue *Penguin-VL ncnn 移植与多平台部署 #6788*。
> 仓库：`Penguin-VL-ncnn`（本目录）。

## 1. 背景与目标

Penguin-VL 是腾讯 AI Lab 开源的视觉语言模型（VLM）。本工作的目标：

1. 用 **pnnx** 把 Penguin-VL 转换到 ncnn；
2. 参考 / fork **ncnn_llm**，用 C++ 实现 LLM 解码，尽量减少第三方依赖；
3. 相同输入下，ncnn 输出的最终文本与 PyTorch 原版一致（贪心解码）；
4. 用 **CMake** 构建，至少在 Linux / Windows 两个平台编译运行。

## 2. 模型结构拆解

Penguin-VL 由三部分组成：

- **Penguin-Encoder**：视觉编码器，用 Qwen3 LLM 权重初始化，改造为
  *双向注意力* + *2D-RoPE*，前置一个 `Conv2d(3, H, k=14, s=14)` 做 patch 嵌入；
- **Projector**：`mlp2x_gelu`（Linear → GELU → Linear），把视觉特征映射到 LLM 维度；
- **LLM**：标准 `Qwen3ForCausalLM`。

一个关键观察：**LLM 侧使用标准 1D RoPE，而非 mRoPE**。视觉 token 只是在
`<image>` 占位符处替换进 `inputs_embeds`，位置编码仍是顺序 1D。这样一来，
Penguin 特有的复杂度全部集中在视觉分支，文本解码可以直接复用 ncnn_llm 的
Qwen3 KV-cache 解码器。

据此把模型切成 6 个 ncnn 子图：`patch_embed / vision_encoder / projector /
embed / decoder / lm_head`，由 `model.json` 统一描述。

## 3. 导出策略（pnnx）

每个子图单独 trace，并把 **RoPE 的 cos/sin 作为外部输入**，避免把位置信息 bake
进图里。这样同一张图能处理变长输入，C++ 侧负责按实际 patch 数 / 序列长度生成
cos/sin。

- `patch_embed`：Conv2d(kernel=stride=patch) 等价于对展平 patch 的一次 Linear，
  导出为 Linear 更利于批量执行；
- `vision_encoder`：单图场景 `cu_seqlens=[0,N]`，即全序列双向注意力，无需 mask；
- `decoder`：沿用 nihui / ncnn_llm 的 KV-cache 图约定
  （`in0..in3` + `cache_k{i}/cache_v{i}` → `out_cache_k{i}/...` + `out0`）。

## 4. C++ 运行时（最小依赖）

依赖仅有 **ncnn** 和一个 vendored 的 `stb_image.h`；配置文件用自带的
~300 行 JSON 解析器读取，不引入 nlohmann/json。运行时包含：

- **图像预处理**：忠实复刻 `image_processing_penguinvl.py` 的
  目标分辨率计算（含 Python `round` 的 banker's rounding）、bicubic resize、
  归一化（mean=std=0.5，即映射到 [-1,1]）、以及 patchify 的通道优先 + 行主序排布；
- **2D-RoPE**：`generate_penguin_vision_rope()`，与 transformers 的
  `apply_multimodal_rotary_pos_emb` 对齐；
- **多模态融合**：在 `<image>` 处注入投影后的视觉 token；
- **KV-cache 解码**：两段式 prefill + 逐 token 解码，默认 **贪心（argmax）**
  保证可复现、与 PyTorch 对齐。

## 5. 数值对齐

采用从输入向输出逐级比对的方法（见 `docs/ALIGNMENT.md`）：预处理 → patch 嵌入
→ 编码器 → 投影 → 融合 embedding → 首个 logits 的 argmax → 完整文本。
经验上最大的对齐风险是 **resize 内核差异**（PIL BICUBIC vs ncnn），其次是
**chat template / system prompt** 与 **fp16 翻转 argmax**。因此流程规定：
先 fp32 对齐通过，再切 fp16 复验。

## 6. 构建与多平台

CMake（≥3.15）`find_package(ncnn)` 即可；GitHub Actions 同时在
`ubuntu-latest` 与 `windows-latest` 上构建 ncnn + 本项目并跑单测。
不依赖 GPU / Vulkan，纯 CPU 可跑。

## 7. 现状与后续

- 已完成：C++ 运行时全链路、预处理 / 2D-RoPE / 分词器接入、KV-cache 解码、
  CMake 多平台构建、无权重可跑的单元测试（JSON / 预处理 / 2D-RoPE / 分词器）；
- 导出脚本已按上述结构与契约给出，需在带 PyTorch + 权重的机器上运行以产出
  `*.param/*.bin` 并完成端到端文本对齐；
- 后续：视频路径（merge_size=2 + TRA 压缩）、fp16/int8 量化、以及向
  ncnn_llm 提交 Qwen3-VL 类结构的 PR。

## 8. 复现

```bash
# 构建
cmake -S . -B build -Dncnn_DIR=<ncnn>/lib/cmake/ncnn && cmake --build build -j
ctest --test-dir build

# 导出（需 torch + 权重）
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out m
python tools/export_penguinvl.py   --model tencent/Penguin-VL-2B --out m

# 运行
./build/penguinvl --model m --image demo.jpg --prompt "Describe this image."
```
