#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Export the Qwen3 causal decoder of Penguin-VL to ncnn with an explicit
# KV-cache interface matching the C++ runtime contract (docs/EXPORT.md).
#
# forward(hidden, mask, cos, sin, k0, v0, k1, v1, ...) ->
#         (hidden_out, k0_out, v0_out, k1_out, v1_out, ...)
#
# After pnnx, run tools/rename_decoder_blobs.py to name the blobs
# in0..in3 / cache_k{i} / cache_v{i} / out_cache_k{i} / out_cache_v{i} / out0.
#
# Usage:
#   python tools/export_decoder_kvcache.py --model tencent/Penguin-VL-2B --out out_dir

import argparse
import os
import subprocess

import torch


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, rep):
    # GQA head expansion via expand+reshape (pnnx->ncnn friendly).
    if rep == 1:
        return x
    b, nkv, n, hd = x.shape
    return x[:, :, None, :, :].expand(b, nkv, rep, n, hd).reshape(b, nkv * rep, n, hd)


class Qwen3DecoderKV(torch.nn.Module):
    """Traceable Qwen3 decoder with per-layer external KV-cache."""

    def __init__(self, model, num_heads, num_kv_heads, head_dim, eps):
        super().__init__()
        self.layers = model.layers
        self.norm = model.norm
        self.nh = num_heads
        self.nkv = num_kv_heads
        self.hd = head_dim
        self.eps = eps

    def forward(self, hidden, mask, cos, sin, *caches):
        # hidden (seq, H); cos/sin (seq, hd); caches = [k0,v0,k1,v1,...] (past, nkv, hd)
        x = hidden.unsqueeze(0)
        seq = x.shape[1]
        cos_ = cos.view(1, 1, seq, self.hd)
        sin_ = sin.view(1, 1, seq, self.hd)
        outs = []
        rep = self.nh // self.nkv
        for i, layer in enumerate(self.layers):
            residual = x
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            q = attn.q_norm(attn.q_proj(h).view(1, seq, self.nh, self.hd)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(h).view(1, seq, self.nkv, self.hd)).transpose(1, 2)
            v = attn.v_proj(h).view(1, seq, self.nkv, self.hd).transpose(1, 2)
            q = q * cos_ + rotate_half(q) * sin_
            k = k * cos_ + rotate_half(k) * sin_

            past_k = caches[2 * i].unsqueeze(0).transpose(1, 2) if caches else None
            past_v = caches[2 * i + 1].unsqueeze(0).transpose(1, 2) if caches else None
            if past_k is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            # export the updated cache (past+seq, nkv, hd)
            outs.append(k.transpose(1, 2).squeeze(0))
            outs.append(v.transpose(1, 2).squeeze(0))

            kk = repeat_kv(k, rep)
            vv = repeat_kv(v, rep)
            scores = torch.matmul(q, kk.transpose(-1, -2)) / (self.hd ** 0.5)
            scores = scores + mask.view(1, 1, seq, -1)
            probs = torch.softmax(scores, dim=-1)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(1, seq, -1)
            x = residual + attn.o_proj(o)
            residual = x
            h = layer.post_attention_layernorm(x)
            x = residual + layer.mlp(h)
        x = self.norm(x)
        return (x.squeeze(0), *outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=8, help="trace prefill length")
    ap.add_argument("--past", type=int, default=0, help="trace past length")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 trust_remote_code=True).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dec = Qwen3DecoderKV(model.model, cfg.num_attention_heads, cfg.num_key_value_heads,
                         hd, getattr(cfg, "rms_norm_eps", 1e-6)).eval()

    seq, past = args.seq, args.past
    hidden = torch.randn(seq, cfg.hidden_size)
    mask = torch.zeros(seq, past + seq)
    cos = torch.randn(seq, hd)
    sin = torch.randn(seq, hd)
    caches = []
    if past > 0:
        for _ in range(cfg.num_hidden_layers):
            caches.append(torch.randn(past, cfg.num_key_value_heads, hd))
            caches.append(torch.randn(past, cfg.num_key_value_heads, hd))
    example = (hidden, mask, cos, sin, *caches)

    pt = os.path.join(args.out, "_work_decoder.pt")
    os.makedirs(os.path.dirname(pt), exist_ok=True)
    with torch.no_grad():
        ts = torch.jit.trace(dec, example, check_trace=False)
    ts.save(pt)

    shapes = [[seq, cfg.hidden_size], [seq, past + seq], [seq, hd], [seq, hd]]
    for _ in range(len(caches)):
        shapes.append([max(1, past), cfg.num_key_value_heads, hd])
    shape_arg = ",".join("[" + ",".join(str(d) for d in s) + "]f32" for s in shapes)
    cmd = [os.environ.get("PNNX", "pnnx"), pt, f"inputshape={shape_arg}",
           f"ncnnparam={args.out}/decoder.ncnn.param",
           f"ncnnbin={args.out}/decoder.ncnn.bin", "fp16=0"]
    print("[pnnx]", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[note] now run tools/rename_decoder_blobs.py to set cache blob names")


if __name__ == "__main__":
    main()
