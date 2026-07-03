# Numerical alignment methodology

Goal: **identical final text** between PyTorch and ncnn for the same
`(prompt, image)` under greedy decoding. Because greedy decoding is an
`argmax`, only the *rank order* of the top logit must be preserved at every
step — but a single flipped token cascades, so alignment is verified stage by
stage from the input side outwards.

## 1. Fix the inputs

Use `tools/compare_reference.py` to dump, for one fixed pair:

- `prompt.txt` — the exact templated prompt (tokenizer + chat template must match),
- `pixel_values.npy`, `grid_sizes.npy` — processor output,
- `output.txt` — the greedy golden text.

## 2. Compare stage by stage

| Stage | PyTorch tensor | ncnn tensor | Tolerance |
| --- | --- | --- | --- |
| Preprocess | `pixel_values` | `PreprocessResult.pixel_values` | see resize caveat |
| Patch embed | encoder input | `patch_embed` out0 | 1e-3 |
| Vision encoder | `image_features` (pre-projector) | `encoder` out0 | 1e-2 |
| Projector | `mm_features` | `projector` out0 | 1e-2 |
| Token embeds (fused) | `inputs_embeds` | `build_input_embeds()` | 1e-3 |
| First logits | `logits[:, -1]` | `lm_head` out0 | argmax must match |
| Full decode | `output.txt` | `penguinvl` stdout | exact string |

Dump ncnn intermediates by calling each `ncnn::Extractor` in isolation and
`np.save`-ing the `ncnn::Mat` (a small `--dump` hook can be added to `main.cpp`).

## 3. Known alignment risks (ordered by likelihood)

1. **Resize kernel.** The processor uses PIL `BICUBIC`; ncnn's `resize_bicubic`
   is a different implementation. This is the most likely source of divergence.
   Options: (a) accept it and check the argmax margin is comfortable; (b) do the
   resize with a PIL-matching bicubic in C++; (c) export a fixed-resolution model
   and pre-resize with the reference. Track this per test image.
2. **Chat template / system prompt.** `add_system_prompt=True` in the reference
   must be mirrored by `setting.system_prompt` in `model.json`. A one-token
   template difference changes everything downstream.
3. **fp16.** Export/verify in fp32 first; fp16 can flip a borderline argmax.
4. **RMSNorm eps, q_norm/k_norm, GQA repeat.** Covered by the export wrapper;
   confirm `rms_norm_eps` and head counts from the checkpoint config.
5. **RoPE theta.** LLM `rope_theta` (≈1e6 for Qwen3) vs vision `rope_theta`
   (1e4) are different — both come from `model.json`.

## 4. Regression

Once a `(prompt, image)` pair matches, commit its `prompt.txt` + `output.txt`
under `assets/golden/` and add a scripted diff so future changes can't silently
regress the text.
