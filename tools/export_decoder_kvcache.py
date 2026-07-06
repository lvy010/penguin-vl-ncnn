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
import gc
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the pnnx-verified plain building blocks (RMSNorm/linear cloning) so the
# traced graph matches the self-test exactly. Tracing the stock Qwen3 nn.Modules
# directly aborts pnnx on this torch build (dtype-cast op patterns).
from export_penguinvl import RMSNorm, _clone_linear


def _share_linear(real):
    """Wrap a real nn.Linear in a plain nn.Linear that SHARES its weight tensor
    (no copy). The decoder is ~8 GB in fp32; cloning it while the source model is
    still resident doubles RAM and crashes. Sharing keeps the peak flat — the
    source model wrapper is dropped afterwards while these tensors stay alive."""
    lin = torch.nn.Linear(real.in_features, real.out_features,
                          bias=real.bias is not None)
    lin.weight = real.weight
    if real.bias is not None:
        lin.bias = real.bias
    return lin


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
    """Traceable Qwen3 decoder with per-layer external KV-cache. Each real layer
    is rebuilt from plain ops with the real weights copied in (see the vision
    encoder note in export_penguinvl.py)."""

    def __init__(self, model, num_heads, num_kv_heads, head_dim, eps, no_cache=False):
        super().__init__()
        self.nh = num_heads
        self.nkv = num_kv_heads
        self.hd = head_dim
        self.eps = eps
        self.no_cache = no_cache
        self.layers = torch.nn.ModuleList()
        for layer in model.layers:
            a = layer.self_attn
            self.layers.append(torch.nn.ModuleDict({
                "input_layernorm": RMSNorm(layer.input_layernorm.weight, eps),
                "post_attention_layernorm": RMSNorm(layer.post_attention_layernorm.weight, eps),
                "q_proj": _share_linear(a.q_proj),
                "k_proj": _share_linear(a.k_proj),
                "v_proj": _share_linear(a.v_proj),
                "o_proj": _share_linear(a.o_proj),
                "q_norm": RMSNorm(a.q_norm.weight, eps),
                "k_norm": RMSNorm(a.k_norm.weight, eps),
                "gate_proj": _share_linear(layer.mlp.gate_proj),
                "up_proj": _share_linear(layer.mlp.up_proj),
                "down_proj": _share_linear(layer.mlp.down_proj),
            }))
        self.norm = RMSNorm(model.norm.weight, eps)

    def forward(self, hidden, mask, cos, sin, *caches):
        # hidden (seq, H); cos/sin (seq, hd); caches = [k0,v0,k1,v1,...] (past, nkv, hd)
        # Token stream stays 2D (seq, H) for clean Gemms; attention lifts to a
        # batch=1 4D form (1, heads, len, hd) exactly like the pnnx/ncnn-verified
        # vision encoder -- this is the ONLY attention layout that both converts
        # through pnnx AND runs correctly in ncnn on this toolchain.
        #
        # no_cache=True drops the external cache entirely (in0..in3 -> out0). The
        # runtime then reprocesses the whole running sequence each decode step,
        # which sidesteps the KV-cache concat path that pnnx cannot lower here.
        x = hidden
        seq = x.shape[0]
        cos_ = cos.view(1, 1, seq, self.hd)
        sin_ = sin.view(1, 1, seq, self.hd)
        outs = []
        rep = self.nh // self.nkv
        for i, m in enumerate(self.layers):
            residual = x
            h = m["input_layernorm"](x)
            q = m["q_norm"](m["q_proj"](h).view(seq, self.nh, self.hd)).unsqueeze(0).transpose(1, 2)
            k = m["k_norm"](m["k_proj"](h).view(seq, self.nkv, self.hd)).unsqueeze(0).transpose(1, 2)
            v = m["v_proj"](h).view(seq, self.nkv, self.hd).unsqueeze(0).transpose(1, 2)
            q = q * cos_ + rotate_half(q) * sin_
            k = k * cos_ + rotate_half(k) * sin_

            if caches:
                past_k = caches[2 * i].unsqueeze(0).transpose(1, 2)
                past_v = caches[2 * i + 1].unsqueeze(0).transpose(1, 2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)
            if not self.no_cache:
                outs.append(k.transpose(1, 2).squeeze(0))
                outs.append(v.transpose(1, 2).squeeze(0))

            kk = repeat_kv(k, rep)
            vv = repeat_kv(v, rep)
            scores = torch.matmul(q, kk.transpose(-1, -2)) / (self.hd ** 0.5)
            scores = scores + mask.view(1, 1, seq, -1)
            probs = torch.softmax(scores, dim=-1)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(seq, -1)
            x = residual + m["o_proj"](o)
            residual = x
            h = m["post_attention_layernorm"](x)
            x = residual + m["down_proj"](torch.nn.functional.silu(m["gate_proj"](h)) * m["up_proj"](h))
        x = self.norm(x)
        if self.no_cache:
            return x
        return (x, *outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=8, help="trace prefill length")
    ap.add_argument("--past", type=int, default=1,
                    help="trace past length (>=1 so the KV-cache inputs exist)")
    ap.add_argument("--no-cache", action="store_true",
                    help="export a cacheless decoder (in0..in3 -> out0 full hidden); "
                         "the runtime reprocesses the whole sequence each step. This "
                         "sidesteps pnnx/ncnn KV-cache concat limitations.")
    ap.add_argument("--name", default="decoder",
                    help="output basename (e.g. decoder / decoder_nocache)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoModelForCausalLM
    # Load native dtype then cast to fp32 (casting bf16->fp32 during the
    # safetensors load can segfault on some torch builds; post-load cast is safe).
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 trust_remote_code=True).float().eval()
    print("[decoder] model loaded", flush=True)
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dec = Qwen3DecoderKV(model.model, cfg.num_attention_heads, cfg.num_key_value_heads,
                         hd, getattr(cfg, "rms_norm_eps", 1e-6),
                         no_cache=args.no_cache).eval()
    # dec now holds independent copies of the decoder weights; drop the ~9 GB
    # source model so pnnx has room to convert the (large) decoder graph.
    n_layers = cfg.num_hidden_layers
    del model
    gc.collect()
    print(f"[decoder] rebuilt {n_layers} layers, freed source model", flush=True)

    seq, past = args.seq, (0 if args.no_cache else args.past)
    hidden = torch.randn(seq, cfg.hidden_size)
    mask = torch.zeros(seq, past + seq)
    cos = torch.randn(seq, hd)
    sin = torch.randn(seq, hd)
    caches = []
    if past > 0:
        for _ in range(n_layers):
            caches.append(torch.randn(past, cfg.num_key_value_heads, hd))
            caches.append(torch.randn(past, cfg.num_key_value_heads, hd))
    example = (hidden, mask, cos, sin, *caches)

    pt = os.path.join(args.out, f"_work_{args.name}.pt")
    os.makedirs(os.path.dirname(pt), exist_ok=True)
    print("[decoder] tracing...", flush=True)
    with torch.no_grad():
        ts = torch.jit.trace(dec, example, check_trace=False)
    print("[decoder] traced, saving .pt...", flush=True)
    ts.save(pt)
    del dec, ts
    gc.collect()
    print("[decoder] saved .pt, running pnnx...", flush=True)

    shapes = [[seq, cfg.hidden_size], [seq, past + seq], [seq, hd], [seq, hd]]
    for _ in range(len(caches)):
        shapes.append([max(1, past), cfg.num_key_value_heads, hd])
    shape_arg = ",".join("[" + ",".join(str(d) for d in s) + "]f32" for s in shapes)
    cmd = [os.environ.get("PNNX", "pnnx"), pt, f"inputshape={shape_arg}",
           f"ncnnparam={args.out}/{args.name}.ncnn.param",
           f"ncnnbin={args.out}/{args.name}.ncnn.bin", "fp16=0"]
    print("[pnnx]", " ".join(cmd))
    # The decoder .pt is multi-GB; right after saving it the OS write-back cache
    # can transiently starve pnnx and it aborts. Let the cache flush, then retry
    # once from the already-saved .pt if the first attempt fails.
    import time
    time.sleep(5)
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("[pnnx] first attempt failed (likely transient memory); retrying...",
              flush=True)
        gc.collect()
        time.sleep(15)
        subprocess.check_call(cmd)
    if args.no_cache:
        print("[note] cacheless decoder: inputs in0..in3, output out0 -- no blob "
              "rename needed. Set \"kv_cache\": false in model.json.")
    else:
        print("[note] now run tools/rename_decoder_blobs.py to set cache blob names")


if __name__ == "__main__":
    main()
