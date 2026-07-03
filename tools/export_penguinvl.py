#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Export a HuggingFace Penguin-VL checkpoint to ncnn using pnnx.
#
# The model is split into the same sub-graphs the C++ runtime loads:
#   1. patch_embed   Conv2d(3, Hvis, k=patch, s=patch)  -> per-patch projection
#   2. vision_enc    bidirectional Qwen3 encoder + final RMSNorm (2D-RoPE external)
#   3. projector     mlp2x_gelu  (Hvis -> Hllm)
#   4. embed_tokens  LLM token embedding
#   5. decoder       Qwen3 causal decoder with explicit KV-cache in/out
#   6. lm_head       final projection to vocab logits
#
# It also writes model.json describing the deployment.
#
# Requirements (NOT available in a minimal CI box; run this on a torch machine):
#   pip install torch transformers
#   pnnx binary on PATH (built from ncnn/tools/pnnx) or `pip install pnnx`
#
# Usage:
#   python tools/export_penguinvl.py \
#       --model tencent/Penguin-VL-2B \
#       --out   ./penguin-vl-2b-ncnn \
#       --dtype fp16
#
# See docs/EXPORT.md for the full pipeline and the decoder KV-cache contract.

import argparse
import json
import os
import subprocess
import sys

import torch


def run_pnnx(pt_path, input_shapes, out_prefix):
    """Invoke pnnx on a traced TorchScript file.

    input_shapes: list of shape lists, e.g. [[196, 588]] -> "[196,588]".
    Produces <out_prefix>.ncnn.param / <out_prefix>.ncnn.bin next to pt_path.
    """
    shape_arg = ",".join("[" + ",".join(str(d) for d in s) + "]f32" for s in input_shapes)
    cmd = [
        os.environ.get("PNNX", "pnnx"),
        pt_path,
        f"inputshape={shape_arg}",
        f"ncnnparam={out_prefix}.ncnn.param",
        f"ncnnbin={out_prefix}.ncnn.bin",
        f"ncnnpy={out_prefix}_ncnn.py",
        "fp16=0",
    ]
    print("[pnnx]", " ".join(cmd))
    subprocess.check_call(cmd)


def trace_and_export(module, example_inputs, input_shapes, out_prefix, work_dir):
    module.eval()
    pt_path = os.path.join(work_dir, os.path.basename(out_prefix) + ".pt")
    with torch.no_grad():
        ts = torch.jit.trace(module, example_inputs, check_trace=False)
    ts.save(pt_path)
    run_pnnx(pt_path, input_shapes, out_prefix)


# --------------------------------------------------------------------------- #
# Traceable wrappers. Each isolates one stage and applies RoPE using external
# cos/sin so positions are never baked into the graph (the C++ side supplies
# them). They mirror the reference implementation in penguinvl/model/*.
# --------------------------------------------------------------------------- #

class PatchEmbed(torch.nn.Module):
    """Flattened-patch -> embedding. Equivalent to the reference Conv2d applied
    per patch, expressed as a single Linear over the (3*ps*ps) feature vector."""

    def __init__(self, conv, patch_size, num_channels):
        super().__init__()
        w = conv.weight.reshape(conv.out_channels, -1).contiguous()  # (H, 3*ps*ps)
        self.linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=conv.bias is not None)
        with torch.no_grad():
            self.linear.weight.copy_(w)
            if conv.bias is not None:
                self.linear.bias.copy_(conv.bias)

    def forward(self, pixel_values):  # (N, 3*ps*ps) -> (N, H)
        return self.linear(pixel_values)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, rep):
    # GQA head expansion via expand+reshape (pnnx->ncnn friendly; avoids the
    # unsupported repeat_interleave op).
    if rep == 1:
        return x
    b, nkv, n, hd = x.shape
    return x[:, :, None, :, :].expand(b, nkv, rep, n, hd).reshape(b, nkv * rep, n, hd)


class VisionEncoder(torch.nn.Module):
    """Bidirectional Qwen3 encoder over patch embeddings, 2D-RoPE supplied as
    cos/sin (N, head_dim). Full attention (no mask) for a single image."""

    def __init__(self, encoder, head_dim, num_heads, num_kv_heads, eps):
        super().__init__()
        self.layers = encoder.layers
        self.norm = encoder.norm
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.eps = eps

    def forward(self, hidden, cos, sin):  # hidden (N, H), cos/sin (N, head_dim)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        x = hidden.unsqueeze(0)  # (1, N, H)
        for layer in self.layers:
            residual = x
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            N = h.shape[1]
            q = attn.q_norm(attn.q_proj(h).view(1, N, self.num_heads, self.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(h).view(1, N, self.num_kv_heads, self.head_dim)).transpose(1, 2)
            v = attn.v_proj(h).view(1, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)
            k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
            rep = self.num_heads // self.num_kv_heads
            k = repeat_kv(k, rep)
            v = repeat_kv(v, rep)
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            probs = torch.softmax(scores, dim=-1)
            o = torch.matmul(probs, v).transpose(1, 2).reshape(1, N, -1)
            x = residual + attn.o_proj(o)
            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)
        x = self.norm(x)
        return x.squeeze(0)  # (N, H)


class Projector(torch.nn.Module):
    def __init__(self, projector):
        super().__init__()
        self.readout = projector.readout  # nn.Sequential(Linear, GELU, Linear, ...)

    def forward(self, x):  # (N, Hvis) -> (N, Hllm)
        return self.readout(x)


class EmbedTokens(torch.nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed = embed

    def forward(self, ids):  # (seq,) int -> (seq, H)
        return self.embed(ids)


class LmHead(torch.nn.Module):
    def __init__(self, lm_head):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden):  # (seq, H) -> (seq, vocab)
        return self.lm_head(hidden)


def build_model_json(args, cfg, vcfg, out_dir):
    doc = {
        "model_type": "penguinvl_qwen3",
        "params": {
            "embed_token_param": "embed.ncnn.param",
            "embed_token_bin": "embed.ncnn.bin",
            "decoder_param": "decoder.ncnn.param",
            "decoder_bin": "decoder.ncnn.bin",
            "lm_head_param": "lm_head.ncnn.param",
            "lm_head_bin": "lm_head.ncnn.bin",
        },
        "tokenizer": {
            "type": "bbpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "bos": "",
            "eos": "<|im_end|>",
            "additional_special_tokens": [
                "<|im_start|>", "<|im_end|>", "<image>", "<think>", "</think>",
            ],
        },
        "setting": {
            "hidden_size": cfg.hidden_size,
            "attn_cnt": cfg.num_hidden_layers,
            "system_prompt": "",
            "rope": {
                "type": "RoPE",
                "rope_head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
                "rope_theta": float(getattr(cfg, "rope_theta", 1000000.0)),
            },
            "vision": {
                "patch_embed_param": "vision_patch_embed.ncnn.param",
                "patch_embed_bin": "vision_patch_embed.ncnn.bin",
                "encoder_param": "vision_encoder.ncnn.param",
                "encoder_bin": "vision_encoder.ncnn.bin",
                "projector_param": "vision_projector.ncnn.param",
                "projector_bin": "vision_projector.ncnn.bin",
                "patch_size": vcfg.patch_size,
                "merge_size": 1,
                "min_tokens": 16,
                "max_tokens": 16384,
                "vision_hidden": vcfg.hidden_size,
                "vision_head_dim": getattr(vcfg, "head_dim", vcfg.hidden_size // vcfg.num_attention_heads),
                "vision_rope_theta": float(getattr(vcfg, "rope_theta", 1000000.0)),
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "image_token": "<image>",
            },
        },
    }
    with open(os.path.join(out_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print("[ok] wrote model.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--work", default=None, help="scratch dir for .pt/.onnx (default: <out>/_work)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = args.work or os.path.join(args.out, "_work")
    os.makedirs(work, exist_ok=True)

    # Import here so the file can be inspected without transformers installed.
    from transformers import AutoModelForCausalLM
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(f"[load] {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    llm = model.model                 # PenguinVLQwen3Model (VLMMetaModel + Qwen3Model)
    cfg = model.config
    vision = llm.get_vision_encoder().vision_encoder  # PenguinVLVisionEncoderModel
    vcfg = vision.config
    projector = llm.get_vision_projector()

    Hvis = vcfg.hidden_size
    ps = vcfg.patch_size
    # Qwen3 decouples head_dim from hidden_size/num_heads, so read it explicitly.
    vis_head_dim = getattr(vcfg, "head_dim", Hvis // vcfg.num_attention_heads)

    # 1. patch embed
    trace_and_export(
        PatchEmbed(vision.embeddings.patch_embedding, ps, vcfg.num_channels),
        (torch.randn(196, 3 * ps * ps),), [[196, 3 * ps * ps]],
        os.path.join(args.out, "vision_patch_embed"), work)

    # 2. vision encoder
    N = 196
    trace_and_export(
        VisionEncoder(vision.encoder, vis_head_dim, vcfg.num_attention_heads,
                      vcfg.num_key_value_heads, getattr(vcfg, "rms_norm_eps", 1e-6)),
        (torch.randn(N, Hvis), torch.randn(N, vis_head_dim), torch.randn(N, vis_head_dim)),
        [[N, Hvis], [N, vis_head_dim], [N, vis_head_dim]],
        os.path.join(args.out, "vision_encoder"), work)

    # 3. projector
    trace_and_export(
        Projector(projector), (torch.randn(N, Hvis),), [[N, Hvis]],
        os.path.join(args.out, "vision_projector"), work)

    # 4. embed tokens
    trace_and_export(
        EmbedTokens(llm.embed_tokens), (torch.randint(0, cfg.vocab_size, (8,)),), [[8]],
        os.path.join(args.out, "embed"), work)

    # 6. lm_head
    trace_and_export(
        LmHead(model.lm_head), (torch.randn(1, cfg.hidden_size),), [[1, cfg.hidden_size]],
        os.path.join(args.out, "lm_head"), work)

    # 5. decoder (KV-cache). This uses the ncnn_llm / nihui KV-cache decoder
    #    export convention; see docs/EXPORT.md. The traceable wrapper lives in
    #    tools/export_decoder_kvcache.py and must be exported on a torch machine.
    print("[note] decoder KV-cache export: see docs/EXPORT.md and "
          "tools/export_decoder_kvcache.py (blob names must be in0..in3, "
          "cache_k{i}/cache_v{i}, out_cache_k{i}/out_cache_v{i}, out0).")

    build_model_json(args, cfg, vcfg, args.out)
    print(f"[done] exported sub-graphs to {args.out}")


if __name__ == "__main__":
    main()
