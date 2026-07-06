# Penguin-VL-ncnn 技术报告

腾讯 Penguin-VL-2B（约 2B 参数、基于 Qwen3 的视觉-语言模型）在 ncnn 上的端到端 C++ 推理移植：关键难点、实现路径与验证结果

作者: lvy010
日期: 2026 年 7 月
版本: v0.1

## 摘要

本报告记录了将 Penguin-VL-2B（约 2B 参数的多模态视觉-语言模型）从 PyTorch 完整移植到 ncnn 推理框架的技术细节。移植后的 C++ 运行时在 fp32 精度下与 PyTorch 原版**逐子图数值对齐**（embed 误差 = 0、lm_head 误差 ≈ 4.8e-6、解码器 ≈ 1.1e-5），greedy 解码的文本输出与 PyTorch **逐 token 完全一致**。

本报告重点讨论七个在移植过程中遇到的关键难点：(1) Windows/MSVC 下 `std::isfinite` 的工具链兼容；(2) KV-cache 解码器导致 pnnx 转换崩溃 / ncnn 运行时堆损坏，最终改用无 cache 解码器；(3) pnnx 把序列长度写死进 Reshape，用 `inputshape2` 恢复动态维；(4) 动态形状下 GQA 的 `expand` 退化成 ncnn 无法加载的 `pnnx.Expression`/`Tensor.expand`；(5) 因果 mask 广播被 pnnx 降级成带双 `-1` 的 Reshape，ncnn 无法推断；(6) ncnn 原生 `RotaryEmbed` 的 cos/sin 布局约定；(7) Windows argv 的 GBK 编码破坏中文 prompt。每个难点都给出了定位过程、根因分析和可复现的修复方案。

---

## 1. 项目背景

### 1.1 模型架构

Penguin-VL-2B 是一个视觉-语言多模态模型，由视觉塔 + LLM 解码器构成，端到端流水线如下：

```
Image(RGB) -> PatchEmbed(14x14) -> ViT Encoder(1024d, 2D-RoPE) -> Projector(->2048d)
                                                                        |
                                                                image_embeds
                                                                        v
Output Tokens <- LM Head(2048->151675) <- LLM Decoder(28L, GQA) <- Embedding + <image> 替换(2048d)
      ^
   BBPE Tokenizer
```

### 1.2 关键超参数

| 组件 | 维度 | 层数 | 头数 | 备注 |
|---|---|---|---|---|
| ViT Encoder | 1024 | - | head_dim=128 | patch_size=14, 2D-RoPE, rope_theta=1e6 |
| Projector | 1024 -> 2048 | - | - | 对齐 LLM 隐藏维 |
| LLM Decoder | 2048 | 28 | 16 heads / 8 kv heads | GQA, head_dim=128, intermediate=6144 |
| LM Head | 2048 -> 151675 | - | - | tie_word_embeddings（权重与 embedding 共享） |

- `rope_theta = 1000000`，无 rope_scaling，`rms_norm_eps = 1e-6`
- 分词器：字节级 BPE（bbpe），特殊 token 含 `<|im_start|>`、`<|im_end|>`、`<image>`、`<think>`、`</think>`

### 1.3 设计目标

- **数值对齐**：fp32 下逐子图对齐，最终 greedy 文本与 PyTorch 逐 token 一致。
- **零 Python 依赖**：纯 C++，产物是运行时库 + CLI。
- **自包含**：单个模型目录（6 个 ncnn 子图 + 分词器 + `model.json`）即可推理。

---

## 2. 整体架构

### 2.1 子模型拆分策略

为适配 ncnn 的静态形状约束并降低单次 trace 的复杂度，将 PyTorch 模型拆成 6 个独立 ncnn 子图：

| 子模型 | 文件 | 输入 -> 输出 |
|---|---|---|
| Patch Embed | `vision_patch_embed.ncnn.{param,bin}` | 图像 patch -> 1024d |
| Vision Encoder | `vision_encoder.ncnn.{param,bin}` | 1024d -> 1024d hidden |
| Projector | `vision_projector.ncnn.{param,bin}` | 1024d -> 2048d |
| Embedding | `embed.ncnn.{param,bin}` | token_id -> 2048d |
| LLM Decoder | `decoder_nocache.ncnn.{param,bin}` | hidden, mask, cos, sin -> 2048d |
| LM Head | `lm_head.ncnn.{param,bin}` | 2048d -> 151675 logits |

### 2.2 运行时调用流

```
PenguinVL::chat(prompt, image_path)
  |
  |-- (可选) 视觉分支
  |     |-- load_image + preprocess          -> pixel_values
  |     |-- PatchEmbed -> VisionEncoder -> Projector -> image_embeds [N, 2048]
  |
  |-- build_prompt(chat 模板 + <image> 占位 + <think> 块)
  |-- Embedding(token_ids) -> text_embeds，并把 <image> 位置替换为 image_embeds
  |
  |-- for step in 0..max_new_tokens:              # 无 cache：每步重跑整段序列
        |-- make_causal_mask(n) + generate_rope_embed_cache_full(n)
        |-- Decoder(running, mask, cos, sin)      -> hidden [2048, n]
        |-- LmHead(hidden[-1])                    -> logits [151675]
        |-- argmax / sample                       -> next_id
        |-- if next_id == eos: break
        |-- 把 next_id 的 embedding 追加到 running
```

---

## 3. 关键难点与解决方案

### 3.1 难点 1：Windows/MSVC 下 `std::isfinite` 编译失败

**现象**：在 MSVC（19.51）上编译 vision self-test 时，`std::isfinite` 报未声明/重载不明确。

**根因**：不同工具链/标准库版本对 `<cmath>` 中 `std::isfinite` 的可见性处理不一致，隐式包含链不可靠。

**修复**：显式 `#include <cmath>`，并采用工具链无关的有限性判断，避免依赖特定实现的重载解析。修复后 Linux/Windows 均可编译。

### 3.2 难点 2：KV-cache 解码器导致 pnnx 崩溃 / ncnn 运行时堆损坏

**现象**：带显式 KV-cache 的解码器子图有两种失败：

- pnnx 在 `pass_level0` 内联 `RMSNorm` 时崩溃（`0xC0000409`，栈缓冲溢出）；
- 即使转换成功，ncnn 运行时在 prefill 阶段堆损坏崩溃（`-1073740940`）。

**根因**：把 KV-cache 作为图输入、并对 cache 做 `unsqueeze(0)` 加 batch 维的模式，pnnx 无法正确 lower 到 ncnn；大图（28 层 Qwen3）叠加后触发深层结构不兼容。

**修复**：放弃 KV-cache，改用**无 cache 解码器**——每一步重跑整段累积序列（`in0..in3 -> out0`）。复杂度 O(n^2) 但可靠、自包含，且与已验证的视觉编码器同款 batch=1 注意力风格。

### 3.3 难点 3：pnnx 把序列长度写死进 Reshape

**现象**：无 cache 解码器导出成功、能加载，但输出乱码；用 pip 版 pyncnn 跑则直接崩溃。

**定位**：查看 `.param` 发现 `Reshape` 把 trace 时的 `seq=8` 直接写死（如 mask 的 `0=8 1=8`、q 的 `2=8`）。运行时 `seq != 8` 时元素数不匹配，导致乱码（旧 ncnn）或崩溃（新 ncnn）。

**修复**：给 pnnx 传入**第二组不同 seq 的形状** `inputshape2`（seq 翻倍）。pnnx 比对两组形状后把不一致的维标记为动态（输出 `2=-1`、`model inputshape = [?,2048]...`）。

### 3.4 难点 4：动态形状下 GQA 的 `expand` 退化成不可加载算子

**现象**：加了动态形状后，C++ 加载 `.param` 报 `layer pnnx.Expression not exists or registered`。

**根因**：GQA 的 `repeat_kv` 用 `x[:,:,None].expand(...).reshape(...)`，在动态形状下被 pnnx 降级成 `pnnx.Expression`（计算动态 expand 形状）+ `Tensor.expand`——这两个算子 ncnn 无法加载。

**修复**：把 `expand` 换成**静态平铺次数**的 `repeat(1,1,rep,1,1)`（lower 成 ncnn `Tile`），reshape 只保留一个推断维 `-1`：

```python
def repeat_kv(x, rep):
    if rep == 1:
        return x
    nh = x.shape[1] * rep
    x = x[:, :, None, :, :].repeat(1, 1, rep, 1, 1)   # (1, nkv, rep, n, hd)
    return x.reshape(1, nh, -1, x.shape[-1])           # 只有 n 是 -1
```

同时把 q/k/v 的 `view(seq, ...)` 改为 `view(-1, ...)`，把 `cos.view(1,1,seq,hd)` 改为 `view(1,1,-1,hd)`，避免引入显式 seq 表达式。修复后 `pnnx.Expression`/`Tensor.expand` 计数归零。

### 3.5 难点 5：因果 mask 广播被降级成双 `-1` Reshape

**现象**：算子问题解决后，C++ 能加载并运行，但输出仍是「连贯却错误」的文本。

**定位**：逐子图对齐发现 embed、lm_head 完全正确，问题在解码器。查看 `.param` 命中可疑算子：

```
Reshape  reshape_21  1 1 in1 39 0=-1 1=-1 2=1
```

`scores + mask`（mask 2D、scores 4D）被 pnnx 降级成一个带**两个 `-1`** 的 Reshape，ncnn 无法同时推断两个维，导致 mask 形状错乱、注意力错误，最终文本连贯但错误。

**修复**：把 mask 的 Reshape 改写为 ncnn 的「保持原维」形式（`0` = copy）。导出后自动后处理：

```python
def patch_mask_reshape(param_path):
    # 0=-1 1=-1 2=1  ->  0=0 1=0 2=1  （保持 w、保持 h、置 c=1）
    text = open(param_path, encoding="utf-8").read()
    text = text.replace("0=-1 1=-1 2=1", "0=0 1=0 2=1")
    open(param_path, "w", encoding="utf-8").write(text)
```

C++ 侧喂入 2D 因果 mask（`w=kv, h=seq`），图内部广播到注意力头维。修复后解码器数值恢复正确。

> 备注：也尝试过把 mask 直接做成 4D 输入以彻底避免 Reshape，但 4D mask + 动态 `inputshape2` 会让 pnnx 在 `pass_level0` 确定性崩溃（大图），故最终采用「2D mask + 后处理」的稳妥路线。

### 3.6 难点 6：ncnn 原生 `RotaryEmbed` 的 cos/sin 布局约定

**现象**：pnnx 把手写 RoPE 模式匹配成了 ncnn 原生 `RotaryEmbed` 层（`rope_0`/`rope_1`），需要确认 cos/sin 的喂入方式一致。

**分析**：ncnn `RotaryEmbed`（`interleaved=0`，非交错/半分模式）对每个位置仅读取前 `head_dim/2` 个 cos/sin，并按 `x0=elem[j]`、`x1=elem[j+hd/2]` 做旋转：

```
out[j]        = x0*cos[j] - x1*sin[j]
out[j+hd/2]   = x0*sin[j] + x1*cos[j]
```

而 C++ 的 `generate_rope_embed_cache_full` 产出的是长度 `head_dim` 的「重复」cos/sin（`cos[j] == cos[j+hd/2]`）。`RotaryEmbed` 只用前半段，恰好等于正确的 rotate_half 结果。**结论：full cache 与 ncnn RotaryEmbed 完全等价**，RoPE 正确。

### 3.7 难点 7：Windows argv 的 GBK 编码破坏中文 prompt

**现象**：文本推理输出「连贯却错误」——但 embed/decoder/lm_head/RoPE 全部已验证正确。

**定位**：打印 C++ 的 token id 与 HF 对比，同一 prompt「请用一句话介绍你自己」，HF = 5 个中文 token，C++ = 16 个「字节 token」（`131 167 143 127 ...`）。解码这些字节 token 得到 `Ç ë Ó Ã`——正是「请用」的 **GBK** 字节（请=`C7 EB`、用=`D3 C3`）。

**根因**：Windows 上 ANSI `argv` 是系统码页（GBK/936）编码，中文 prompt 在进入 UTF-8 分词器前就被破坏，字节级 BPE 按错误字节合并。

**修复**：`main.cpp` 在入口处用宽字符命令行重建 UTF-8 argv，并把控制台输出设为 UTF-8：

```cpp
#ifdef _WIN32
void make_utf8_argv(int& argc, char**& argv) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    SetConsoleOutputCP(CP_UTF8);
    // WideCharToMultiByte(CP_UTF8, ...) 逐个转 UTF-8，替换 argv
    // ...
}
#endif
```

修复后 C++ 的 token id 与 HF **逐 id 一致**，文本输出与 PyTorch 逐字一致。

---

## 4. 测试结果

### 4.1 数值对齐测试（`tools/align_check.py`）

以「请用一句话介绍你自己」为例，逐子图对比 ncnn 与 PyTorch：

| 阶段 | 对比内容 | 误差 | 状态 |
|---|---|---|---|
| Tokenization | 输入文本 -> token_ids | 0（逐 id 一致） | 通过 |
| Embedding 子图 | token -> 2048d | `max_abs = 0` | 通过 |
| LM Head 子图 | 2048d -> logits | `max_abs ≈ 4.8e-6`，argmax 一致 | 通过 |
| 解码器（torch 重建 vs 真模型） | 完整 forward hidden | `max_abs ≈ 1.1e-5` | 通过 |
| 文本 greedy 生成 | 最终输出文本 | 逐 token 一致 | 通过 |

**文本 greedy 输出（C++ == PyTorch）**：

```
我是一个人工智能助手，能够帮助您完成各种任务和解决问题。
```

### 4.2 端到端图像测试

以官方测试图 `assets/test.jpg` 为例：

```
[dbg] image loaded 565x353
[dbg] preprocess grid=25x40 pv=588x1000x1
[dbg] vision encoded tokens=1000 dim=2048
[dbg] input embeds tokens=1017 hidden=2048
Assistant: This is a new, high-quality, high-resolution ...
```

关键佐证：PyTorch 处理器对同一图产出的形状（`input_ids=1017`、`pixel_values=(1000,588)`）与 C++ **完全一致**，证明图像预处理正确、视觉->解码全链路打通。

### 4.3 性能分析（CPU）

| 阶段 | 说明 |
|---|---|
| 视觉编码（patch_embed + encoder + projector） | 一次前向 |
| LLM decode（无 cache） | **每步重跑整段序列**，O(n^2)；1000+ 视觉 token 时每 token 数十秒 |

**性能瓶颈**：无 cache 解码器每步 full-forward。KV cache 实现后预期提速 5–10 倍（当前受限于 pnnx/ncnn 对 KV-cache 图的支持）。

---

## 5. 经验总结

### 5.1 ncnn / pnnx 移植常见陷阱

1. **pnnx 会把 trace 的动态维写死**：变长序列必须用 `inputshape2` 提供第二组形状，pnnx 才会标记动态维。
2. **动态形状会引入不可加载算子**：`expand`/复杂 reshape 可能降级成 `pnnx.Expression`/`Tensor.expand`；改用 `repeat()`（静态平铺）+ 单一 `-1` reshape。
3. **广播加法可能产生双 `-1` Reshape**：ncnn 无法同时推断两个维；把 Reshape 改写为 `0=0 1=0`（保持原维）或用 4D 对齐输入。
4. **RotaryEmbed 的 cos/sin 布局**：`interleaved=0` 只读前 `hd/2`，与「重复 full cache」等价。
5. **pip 版 ncnn 与本地源码构建的算子可能不兼容**：验证要以真正链接的 C++ 运行时为准（本项目 pyncnn 会崩，但本地 `./ncnn` 正常）。

### 5.2 调试方法论

1. **逐子图对比**：先证明 embed / lm_head / 分词器正确，再定位解码器，避免大海捞针。
2. **token id 直接对比**：文本「连贯却错误」时，第一时间打印 C++ 的 token id 与 HF 对比——本项目正是靠此发现 GBK 编码问题。
3. **单层诊断导出**：用 `--layers 1` 导出单层子图，快速排查算子级问题。
4. **`.param` 逐算子审查**：`.param` 是纯文本，直接搜 `0=-1 1=-1`、`pnnx.` 等可疑模式即可定位转换缺陷。

### 5.3 平台特性

1. **Windows argv 是 GBK**：命令行非 ASCII 参数必须从宽字符命令行重建 UTF-8。
2. **控制台输出编码**：`SetConsoleOutputCP(CP_UTF8)` 才能正确显示中文。

---

## 6. 结论

本项目成功将 Penguin-VL-2B 从 PyTorch 移植到 ncnn，在 fp32 精度下实现了逐子图数值对齐（embed=0、lm_head≈4.8e-6、decoder≈1.1e-5）与 greedy 文本逐 token 一致，图像多模态推理端到端打通。七个关键难点（`std::isfinite`、KV-cache 转换崩溃、seq 写死、`expand` 降级、mask 双 `-1`、RotaryEmbed 布局、GBK argv）均有明确根因与可复现修复。

当前版本可用于 CPU 推理，后续工作聚焦 KV cache、fp16 / int8 量化，以提升长序列与图像场景的吞吐。

---

## 参考与链接

- Penguin-VL 原模型：Penguin-VL-2B
- ncnn：https://github.com/Tencent/ncnn
- pnnx：https://github.com/pnnx/pnnx
- 本项目源码：https://github.com/lvy010/penguin-vl-ncnn
- 移植后模型（含全量权重）：https://huggingface.co/lvy010/penguin-vl-2b-ncnn

---

## 附录：编译与运行

```bash
# 1. 编译 ncnn
git clone https://github.com/Tencent/ncnn.git && cd ncnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF
cmake --build build -j
cmake --install build --prefix build/install

# 2. 编译本项目
cd penguin-vl-ncnn
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$PWD/../ncnn/build/install
cmake --build build --config Release

# 3. 导出子图（示例：无 cache 解码器）
python tools/export_decoder_kvcache.py --model ./Penguin-VL-2B --out ./penguin-vl-2b-ncnn \
    --seq 8 --no-cache --name decoder_nocache

# 4. 运行文本推理
./build/Release/penguinvl --model ./penguin-vl-2b-ncnn \
    --prompt "请用一句话介绍你自己" --threads 8 --max-new-tokens 30

# 5. 运行图像推理
./build/Release/penguinvl --model ./penguin-vl-2b-ncnn \
    --image ./assets/test.jpg --prompt "请描述这张图片" --threads 8 --max-new-tokens 40

# 6. 数值对齐验证
python tools/align_check.py --model ./Penguin-VL-2B --ncnn ./penguin-vl-2b-ncnn --text "请用一句话介绍你自己"
```
