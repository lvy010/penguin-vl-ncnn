# Toolchain self-test (no checkpoint required)

These self-tests prove the **export → ncnn → C++ runtime contract** end-to-end
using the *real* Penguin-VL-2B dimensions but random weights, so they fit in a
small machine and need no multi-GB download. They de-risk the two integration
points that a plain unit test cannot cover: pnnx op support and the ncnn blob
I/O contract shared by the export scripts and the C++ driver.

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pnnx numpy
```

## 1. Export the vision + decoder sub-graphs with pnnx

```bash
python tools/selftest_export.py --out /tmp/pvl_selftest --layers 2 --n 100
```

Produces `vision_patch_embed`, `vision_encoder`, `vision_projector` and a small
`decoder` as `*.ncnn.param/bin`. Dimensions match `tencent/Penguin-VL-2B`
(vision hidden 1024, 16 heads, 8 KV heads, head_dim 128, SwiGLU 3072).

## 2. Run the vision pipeline through the C++ driver

```bash
cmake --build build --target vision_selftest
./build/vision_selftest /tmp/pvl_selftest
```

Expected:

```
vision output: w=2048 h=100 (expect w=2048 h=100)
row0 L2^2=...   (finite, non-zero)
SELFTEST PASS
```

This exercises `PenguinVision::encode`: `patch_embed` → bidirectional encoder
with the C++ 2D-RoPE (`generate_penguin_vision_rope`) fed as `in1/in2` → mlp2x_gelu
projector, all loaded from the pnnx output.

## 3. Verify the decoder KV-cache contract

```bash
grep '^Input' /tmp/pvl_selftest/decoder.ncnn.param        # in0..in3 + cache inputs
python tools/rename_decoder_blobs.py --param /tmp/pvl_selftest/decoder.ncnn.param --layers 2
grep -oE 'cache_k[0-9]+|out_cache_k[0-9]+' /tmp/pvl_selftest/decoder.ncnn.param | sort -u
```

The rename step maps pnnx's `in4..`/`out1..` to `cache_k{i}/cache_v{i}` and
`out_cache_k{i}/out_cache_v{i}` — exactly the names `src/penguin_vl.cpp` binds.

## 4. Verify the real tokenizer (bit-exact vs HuggingFace)

Needs only the tokenizer files (~10 MB), not the checkpoint:

```bash
pip install transformers
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out /tmp/pvl_tok
cmake --build build --target tok_selftest
./build/tok_selftest /tmp/pvl_tok
```

Expected — the C++ byte-level BPE reproduces HuggingFace's IDs exactly:

```
Hello, world!                    -> [9707, 11, 1879, 0]  OK
The quick brown fox.             -> [785, 3974, 13876, 38835, 13]  OK
Describe this image in detail.   -> [74785, 419, 2168, 304, 7716, 13]  OK
TOKENIZER SELFTEST PASS
```

## Verified results (this repo)

| Check | Result |
| --- | --- |
| pnnx converts patch_embed / encoder / projector | ✅ (native `RMSNorm`, GQA via expand+reshape) |
| Vision graphs run through C++ `PenguinVision` | ✅ output `(2048, N)`, finite, non-zero |
| Graph inputs named `in0/in1/in2`, output `out0` | ✅ matches runtime |
| Decoder KV-cache converts (no unsupported ops) | ✅ |
| `rename_decoder_blobs.py` → runtime blob names | ✅ |
| C++ tokenizer IDs == HuggingFace (real Penguin-VL tokenizer) | ✅ bit-exact |

> Note: `repeat_interleave` is **not** supported by pnnx→ncnn; GQA head expansion
> must use `expand()+reshape()` (already applied in all export scripts). This was
> caught by the self-test.
