# Penguin-VL-ncnn

A lightweight C++ runtime that runs [tencent-ailab/Penguin-VL](https://github.com/tencent-ailab/Penguin-VL)
on [ncnn](https://github.com/Tencent/ncnn), plus a `pnnx`-based export pipeline.

It targets the issue **"Penguin-VL ncnn 移植与多平台部署 (#6788)"**:

- Convert Penguin-VL to ncnn with **pnnx**.
- A small C++ decoder/vision runtime with **minimal third-party dependencies**
  (only ncnn + a vendored `stb_image.h`; JSON is parsed by a ~300-line built-in reader).
- **Greedy** decoding so the final text matches the PyTorch original bit-for-bit.
- **CMake** build validated on **Linux and Windows** (see `.github/workflows/ci.yml`).

> Status: the C++ runtime, preprocessing, 2D-RoPE, tokenizer glue, KV-cache decode
> loop, CMake build and unit tests are complete and CI-tested. The pnnx export
> path and the export↔runtime blob contract are **verified end-to-end on the
> vision pipeline using real Penguin-VL-2B dimensions** (random weights) — see
> [docs/SELFTEST.md](docs/SELFTEST.md). Producing the *weighted* `.param/.bin` and
> the final PyTorch↔ncnn text-equality check need the checkpoint + PyTorch on a
> box with enough RAM (the 2B decoder needs ~6–8 GB for fp32 tracing); see
> [docs/EXPORT.md](docs/EXPORT.md) and [docs/ALIGNMENT.md](docs/ALIGNMENT.md).
>
> All architecture dimensions and preprocessing constants in this repo were taken
> from `tencent/Penguin-VL-2B` `config.json` / `preprocessor_config.json`
> (image mean=std=0.5, vision hidden 1024, 28 layers, head_dim 128, rope θ=1e6).

## Architecture

Penguin-VL = **Penguin-Encoder** (a vision encoder *initialized from a Qwen3 LLM*,
converted to bidirectional attention + 2D-RoPE) → **mlp2x_gelu projector** →
**Qwen3 causal LLM**. Crucially, the LLM side uses *standard 1D RoPE* (no mRoPE),
so only the vision path needs Penguin-specific code.

```
image ─preprocess─▶ patch_embed(Conv2d) ─▶ vision_encoder(12× bidirectional Qwen3 + 2D-RoPE)
                                                     │
                                                     ▼
                                              projector(mlp2x_gelu) ─▶ visual tokens
                                                     │  (injected at <image> positions)
text  ─tokenizer─▶ embed_tokens ─▶ token embeds ─────┤
                                                     ▼
                                        Qwen3 decoder (1D RoPE + KV-cache) ─▶ lm_head ─▶ argmax
```

The ncnn deployment is split into six sub-graphs, described by `model.json`
(see [`model.example.json`](model.example.json)).

## Build

Requires a C++17 compiler, CMake ≥ 3.15, and an installed ncnn (`master`).

```bash
# 1) build ncnn (CPU, once)
git clone https://github.com/Tencent/ncnn
cmake -S ncnn -B ncnn/build -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_SIMPLEOCV=ON \
      -DCMAKE_INSTALL_PREFIX=$PWD/ncnn/install
cmake --build ncnn/build --target install -j

# 2) build this project
cmake -S . -B build -Dncnn_DIR=$PWD/ncnn/install/lib/cmake/ncnn
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows (MSVC) is identical via the Visual Studio / Ninja generators; CI builds
both Linux and Windows.

## Convert a model

```bash
pip install torch transformers            # on a machine with the checkpoint
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out penguin-vl-2b-ncnn
python tools/export_penguinvl.py   --model tencent/Penguin-VL-2B --out penguin-vl-2b-ncnn
```

This writes `*.ncnn.param/bin`, `vocab.txt`, `merges.txt` and `model.json` into
the output directory. See [docs/EXPORT.md](docs/EXPORT.md) for the decoder
KV-cache export contract.

## Run

```bash
./build/penguinvl --model ./penguin-vl-2b-ncnn \
  --image ./assets/demo.jpg --prompt "Describe this image in detail." --threads 8
```

Text-only:

```bash
./build/penguinvl --model ./penguin-vl-2b-ncnn --prompt "Hello, who are you?"
```

| Option | Meaning |
| --- | --- |
| `--model` | model directory containing `model.json` |
| `--prompt` | user prompt (required) |
| `--image` | image path for VL input |
| `--threads` | CPU threads (default 4) |
| `--max-new-tokens` | generation cap (default 512) |
| `--sample` / `--temperature` / `--top-k` / `--top-p` | sampling (off by default; greedy) |
| `--think` | keep the assistant thinking turn open |

## Verifying text equality

`tools/compare_reference.py` produces the PyTorch greedy "golden" output for a
fixed (prompt, image). Run the same inputs through `penguinvl` and diff. See
[docs/ALIGNMENT.md](docs/ALIGNMENT.md) for the layer-by-layer methodology and
known alignment caveats (the resize kernel is the main risk).

## Layout

```
penguin-vl-ncnn/
├── src/                    Penguin-specific runtime (config, preprocess, vision, decode, CLI)
├── third_party/ncnn_llm/   vendored engine (tokenizer, RoPE, sampling, image IO) — Apache-2.0
├── tools/                  pnnx export + tokenizer extract + reference dumper
├── tests/                  weight-free unit tests (JSON, preprocess, 2D-RoPE, tokenizer)
├── docs/                   EXPORT / ALIGNMENT / DISCUSSION
└── CMakeLists.txt
```

## License

Apache-2.0. Vendored files under `third_party/` keep their original licenses;
see [NOTICE](NOTICE).
