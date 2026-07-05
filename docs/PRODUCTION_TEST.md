# 生产级完整测试规范（Windows）

本规范定义 Penguin-VL-ncnn 在 Windows / MSVC / 纯 CPU 下的一次**可交付、可复现、
带验收标准**的完整测试，覆盖从工具链到「真实图片 → 场景描述」再到「ncnn 文本 ==
PyTorch 文本」的全链路。配套驱动脚本：[`tools/run_full_test.ps1`](../tools/run_full_test.ps1)。

> 关键前提：能产出真实描述的**带权重 ncnn 模型只有一个来源**——从
> `tencent/Penguin-VL-2B` 源检查点导出（阶段 D）。不导出就没有生产级图片测试，无替代方案。
> 拉取源检查点与「用自己的 HF 仓库存模型」是两回事，后者非必需。

## 测试分层与验收标准

| 层 | 阶段 | 依赖 | 验收标准 |
| --- | --- | --- | --- |
| L1 构建 | A. CMake + MSVC Release | ncnn 已装 | 5 个目标零错误编译链接 |
| L2 单测 | B. `ctest` | 无 | **10/10** 通过 |
| L3 契约 | C. pnnx 导出 + vision/decoder 自测 | torch(cpu)+pnnx | `SELFTEST PASS`；KV-cache blob 名对齐 |
| L3 契约 | C. 分词器位级对齐 | transformers | `TOKENIZER SELFTEST PASS`（id 与 HF 完全一致）|
| L4 端到端 | D. 真实图片 → 描述 | 带权重模型 | 每张 `assets/` 图片产出非空、语义合理的描述 |
| L5 对齐 | E. ncnn 文本 == PyTorch | 源检查点 + PyTorch | 贪心解码下最终文本逐 token 一致 |

一次「生产级完整测试」= **L1–L5 全绿**。L1–L3 无需大模型即可先行验收（工具链正确性），
L4–L5 需真实权重（架构 + 数值正确性）。

## 环境（一次性）

Visual Studio 2022（含 C++ 桌面开发）、Git、CMake ≥ 3.15、Python 3.10+。验证：

```powershell
git --version; cmake --version; python --version
```

## 阶段 A/B/C —— 工具链层（不下大模型）

```powershell
# 编译 ncnn（与 CI 一致，一次）
git clone --depth 1 https://github.com/Tencent/ncnn.git
cmake -S ncnn -B ncnn/build -DCMAKE_BUILD_TYPE=Release `
  -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TOOLS=OFF `
  -DNCNN_BUILD_TESTS=OFF -DNCNN_SIMPLEOCV=ON `
  -DCMAKE_INSTALL_PREFIX="$PWD/ncnn-install"
cmake --build ncnn/build --target install --config Release -j 4

# 契约自测所需的 Python 依赖（仅几十 MB）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pnnx numpy transformers

# 一键跑 L1–L3（构建 + 单测 + 契约自测 + 分词器对齐）
pwsh tools/run_full_test.ps1 -NcnnDir "$PWD/ncnn-install/lib/cmake/ncnn"
```

脚本会打印每个阶段的 PASS/FAIL 与末尾 SUMMARY，任一失败退出码非零。

## 阶段 D —— 导出带权重模型（≥16 GB 内存，联网一次）

```powershell
pip install "torch>=2.1" transformers huggingface_hub pnnx numpy pillow
hf auth login                       # 交互式粘贴 token，勿写入文件
hf download tencent/Penguin-VL-2B --local-dir .\Penguin-VL-2B

python tools\extract_tokenizer.py --model .\Penguin-VL-2B --out .\penguin-vl-2b-ncnn
python tools\export_penguinvl.py   --model .\Penguin-VL-2B --out .\penguin-vl-2b-ncnn
python tools\rename_decoder_blobs.py --param .\penguin-vl-2b-ncnn\decoder.ncnn.param --layers 28
```

> 层数 = `Penguin-VL-2B\config.json` 的 `num_hidden_layers`（Qwen3-2B 通常 28）。

## 阶段 D/E —— 端到端 + PyTorch 对齐（一键）

```powershell
pwsh tools/run_full_test.ps1 `
  -NcnnDir "$PWD/ncnn-install/lib/cmake/ncnn" `
  -Model .\penguin-vl-2b-ncnn `
  -ReferenceModel .\Penguin-VL-2B `
  -Prompt "请描述这张图片" `
  -SkipBuild
```

此时脚本会对 `assets/` 下每张图片跑 `penguinvl.exe`（阶段 D），并用
`compare_reference.py` 产 PyTorch golden 后逐 token 比对（阶段 E）。

## 对齐失败的排查顺序

若 L5 文本对不齐，按 [`docs/ALIGNMENT.md`](ALIGNMENT.md) 的风险优先级排查：
1. **resize 内核**（PIL BICUBIC vs ncnn，最大嫌疑）；
2. chat template / system prompt；
3. fp16 边界 argmax 翻转（先 fp32 对齐再切 fp16）；
4. RMSNorm eps / q_norm / k_norm / GQA；
5. RoPE theta（LLM ≈1e6，vision 1e4）。

对齐通过后，把该 `(prompt, image)` 的 `prompt.txt` + `output.txt` 提交到
`assets/golden/`，加脚本化 diff 做回归护栏。

## 与既有报告的关系

[`docs/TESTREPORT.md`](TESTREPORT.md) 是 L1–L3 已完成的可交付记录；本规范把它扩展为
含 L4–L5 的完整生产口径，并用 `run_full_test.ps1` 统一为一条可复现命令。
