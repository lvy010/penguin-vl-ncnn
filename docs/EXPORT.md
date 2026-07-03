# Export pipeline (PyTorch → pnnx → ncnn)

Penguin-VL is split into six ncnn sub-graphs. Each is traced in isolation with
RoPE applied from **external cos/sin inputs**, so no positional information is
baked into the graphs (the C++ runtime supplies them).

| Sub-graph | Input(s) | Output | Notes |
| --- | --- | --- | --- |
| `vision_patch_embed` | `in0` = pixel_values `(N, 3·ps·ps)` | `out0` `(N, Hvis)` | Conv2d(3,Hvis,ps,ps) re-expressed as a Linear over the flattened patch |
| `vision_encoder` | `in0` hidden `(N, Hvis)`, `in1` cos `(N, hd)`, `in2` sin `(N, hd)` | `out0` `(N, Hvis)` | 12× bidirectional Qwen3 blocks + final RMSNorm; **full attention** (no mask) for one image |
| `vision_projector` | `in0` `(N, Hvis)` | `out0` `(N, Hllm)` | `mlp2x_gelu`: Linear → GELU → Linear |
| `embed` | `in0` ids `(seq)` | `out0` `(seq, Hllm)` | token embedding |
| `decoder` | see below | see below | Qwen3 causal decoder with KV-cache |
| `lm_head` | `in0` `(seq, Hllm)` | `out0` `(seq, vocab)` | final logits |

Run:

```bash
python tools/extract_tokenizer.py --model tencent/Penguin-VL-2B --out out_dir
python tools/export_penguinvl.py   --model tencent/Penguin-VL-2B --out out_dir
```

`PNNX=/path/to/pnnx` can override the pnnx binary.

## Why the vision encoder needs no attention mask

For a single image the reference builds `cu_seqlens = [0, N]`, i.e. one packed
sequence, so attention is full/bidirectional over all `N` patches. We therefore
export the encoder without a mask input and run one image at a time.

## 2D-RoPE layout

`transformers`' `apply_multimodal_rotary_pos_emb` with `rope_section =
[hd/2, hd/2]` yields, per patch:

```
cos[0 : hd/2]  = cos(h_coord · inv_freq[0 : hd/2])
cos[hd/2 : hd] = cos(w_coord · inv_freq[0 : hd/2])   ,  inv_freq[i] = theta^(-2i/hd)
```

with patches in row-major `(grid_h, grid_w)` order (merge_size = 1). This is
implemented in C++ by `generate_penguin_vision_rope()` and unit-tested in
`tests/test_rope2d.cpp`. `rotate_half` pairs dim `i` with `i + hd/2`.

## Decoder KV-cache contract

The C++ decode loop (`src/penguin_vl.cpp`) drives the decoder graph with the
same blob layout used by `ncnn_llm` (nihui's KV-cache decoder):

```
inputs : in0 = hidden (Hllm, seq)
         in1 = attention mask (past+seq, seq)   (0 = attend, -1e38 = masked)
         in2 = rope cos, in3 = rope sin          (rope_head_dim/2, seq)
         cache_k{i} / cache_v{i}                 (omitted on the first prefill pass)
outputs: out_cache_k{i} / out_cache_v{i}
         out0 = hidden of the last position (Hllm, seq)
```

`i` runs over the `attn_cnt` layers from `model.json`. Because Penguin-VL's LLM
uses standard 1D Qwen3 RoPE, the existing `ncnn_llm` Qwen3 decoder export applies
directly — fork it, or use `tools/export_decoder_kvcache.py` (a self-contained
traceable Qwen3 decoder with explicit cache in/out). After pnnx, rename the
cache blobs to `cache_k{i}/out_cache_k{i}` as above (pnnx numbers them `in*/out*`
by graph order); a `rename_decoder_blobs.py` helper is provided for this.

## Dtype

Export in fp32 first and confirm exact text equality (see
[ALIGNMENT.md](ALIGNMENT.md)). Only then switch to fp16 (`fp16=1` in the pnnx
call) and re-verify — fp16 can flip an argmax and diverge the token stream.
