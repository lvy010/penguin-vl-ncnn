#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Toolchain self-test: build the vision sub-graphs with the REAL Penguin-VL-2B
# dimensions but random weights, trace them, and convert with pnnx. This proves
# the export path (ops + blob I/O contract) works and matches the C++ runtime,
# without needing the multi-GB checkpoint or the large-RAM decoder.
#
# Dims (from tencent/Penguin-VL-2B config.json, vision_encoder_config):
#   hidden=1024, heads=16, kv_heads=8, head_dim=128, intermediate=3072, patch=14
#   projector: mlp2x_gelu 1024 -> 2048 (LLM hidden)
#
# Usage: python tools/selftest_export.py --out /tmp/pvl_selftest [--layers 2] [--n 100]

import argparse
import os
import subprocess

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x, rep):
    # GQA head expansion without repeat_interleave (unsupported by pnnx->ncnn).
    # x: (1, nkv, N, hd) -> (1, nkv*rep, N, hd), each kv head repeated rep times.
    if rep == 1:
        return x
    b, nkv, n, hd = x.shape
    x = x[:, :, None, :, :].expand(b, nkv, rep, n, hd)
    return x.reshape(b, nkv * rep, n, hd)


class Attn(nn.Module):
    def __init__(self, h, nh, nkv, hd):
        super().__init__()
        self.nh, self.nkv, self.hd = nh, nkv, hd
        self.q_proj = nn.Linear(h, nh * hd, bias=False)
        self.k_proj = nn.Linear(h, nkv * hd, bias=False)
        self.v_proj = nn.Linear(h, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, h, bias=False)
        self.q_norm = RMSNorm(hd)
        self.k_norm = RMSNorm(hd)

    def forward(self, x, cos, sin):
        # x is (N, H). Keep the Linear/Gemm ops in 2D (ncnn Gemm is exact on 2D
        # but collapses the token dim on a batched 3D input). For attention, add
        # an explicit batch=1 at dim 0 so pnnx strips it and emits clean 3D
        # (heads, N, hd) batched matmuls — dim 0 must stay the batch and never be
        # permuted (pnnx cannot transpose the batch axis).
        N = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(N, self.nh, self.hd)).unsqueeze(0).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(N, self.nkv, self.hd)).unsqueeze(0).transpose(1, 2)
        v = self.v_proj(x).view(N, self.nkv, self.hd).unsqueeze(0).transpose(1, 2)
        c = cos.view(1, 1, N, self.hd)
        s = sin.view(1, 1, N, self.hd)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        rep = self.nh // self.nkv
        k = repeat_kv(k, rep)
        v = repeat_kv(v, rep)
        att = torch.matmul(q, k.transpose(-1, -2)) / (self.hd ** 0.5)
        att = torch.softmax(att, dim=-1)
        o = torch.matmul(att, v).transpose(1, 2).reshape(N, -1)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, h, nh, nkv, hd, inter):
        super().__init__()
        self.input_layernorm = RMSNorm(h)
        self.self_attn = Attn(h, nh, nkv, hd)
        self.post_attention_layernorm = RMSNorm(h)
        self.mlp = MLP(h, inter)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class VisionEncoder(nn.Module):
    def __init__(self, h, nh, nkv, hd, inter, layers):
        super().__init__()
        self.layers = nn.ModuleList([Layer(h, nh, nkv, hd, inter) for _ in range(layers)])
        self.norm = RMSNorm(h)

    def forward(self, hidden, cos, sin):
        x = hidden  # (N, H); batch-free so pnnx emits clean 2D Gemms
        for l in self.layers:
            x = l(x, cos, sin)
        return self.norm(x)


class PatchEmbed(nn.Module):
    def __init__(self, feat, h):
        super().__init__()
        self.linear = nn.Linear(feat, h)

    def forward(self, x):
        return self.linear(x)


class Projector(nn.Module):
    def __init__(self, hvis, hllm):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(hvis, hllm), nn.GELU(), nn.Linear(hllm, hllm))

    def forward(self, x):
        return self.readout(x)


class DecoderKV(nn.Module):
    """Minimal Qwen3-style causal decoder with explicit per-layer KV-cache,
    mirroring tools/export_decoder_kvcache.py (small dims for a fast pnnx check)."""

    def __init__(self, h, nh, nkv, hd, inter, layers):
        super().__init__()
        self.layers = nn.ModuleList([Layer(h, nh, nkv, hd, inter) for _ in range(layers)])
        self.norm = RMSNorm(h)
        self.nh, self.nkv, self.hd = nh, nkv, hd

    def forward(self, hidden, mask, cos, sin, *caches):
        # Batch-free token stream (see Attn.forward): 2D Gemms, batch=1 attention.
        x = hidden
        seq = x.shape[0]
        c = cos.view(1, 1, seq, self.hd)
        s = sin.view(1, 1, seq, self.hd)
        rep = self.nh // self.nkv
        outs = []
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            q = attn.q_norm(attn.q_proj(h).view(seq, self.nh, self.hd)).unsqueeze(0).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(h).view(seq, self.nkv, self.hd)).unsqueeze(0).transpose(1, 2)
            v = attn.v_proj(h).view(seq, self.nkv, self.hd).unsqueeze(0).transpose(1, 2)
            q = q * c + rotate_half(q) * s
            k = k * c + rotate_half(k) * s
            pk = caches[2 * i].unsqueeze(0).transpose(1, 2)
            pv = caches[2 * i + 1].unsqueeze(0).transpose(1, 2)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            outs.append(k.transpose(1, 2).squeeze(0))
            outs.append(v.transpose(1, 2).squeeze(0))
            kk, vv = repeat_kv(k, rep), repeat_kv(v, rep)
            att = torch.matmul(q, kk.transpose(-1, -2)) / (self.hd ** 0.5) + mask.view(1, 1, seq, -1)
            att = torch.softmax(att, dim=-1)
            o = torch.matmul(att, vv).transpose(1, 2).reshape(seq, -1)
            x = x + attn.o_proj(o)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return (self.norm(x), *outs)


def export(module, example, shapes, prefix):
    module.eval()
    pt = prefix + ".pt"
    with torch.no_grad():
        ts = torch.jit.trace(module, example, check_trace=False)
    ts.save(pt)
    shape_arg = ",".join("[" + ",".join(str(d) for d in s) + "]f32" for s in shapes)
    cmd = ["pnnx", pt, f"inputshape={shape_arg}",
           f"ncnnparam={prefix}.ncnn.param", f"ncnnbin={prefix}.ncnn.bin", "fp16=0"]
    print("[pnnx]", " ".join(cmd))
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/pvl_selftest")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(0)
    H, NH, NKV, HD, INTER = 1024, 16, 8, 128, 3072
    HLLM, PS = 2048, 14
    N = args.n

    export(PatchEmbed(3 * PS * PS, H), (torch.randn(N, 3 * PS * PS),),
           [[N, 3 * PS * PS]], os.path.join(args.out, "vision_patch_embed"))
    export(VisionEncoder(H, NH, NKV, HD, INTER, args.layers),
           (torch.randn(N, H), torch.randn(N, HD), torch.randn(N, HD)),
           [[N, H], [N, HD], [N, HD]], os.path.join(args.out, "vision_encoder"))
    export(Projector(H, HLLM), (torch.randn(N, H),), [[N, H]],
           os.path.join(args.out, "vision_projector"))

    # Decoder KV-cache conversion check (tiny dims, past=4, seq=3, 2 layers).
    dh, dnh, dnkv, dhd, dinter = 64, 4, 2, 16, 128
    seq, past = 3, 4
    dec = DecoderKV(dh, dnh, dnkv, dhd, dinter, args.layers)
    caches = []
    for _ in range(args.layers):
        caches.append(torch.randn(past, dnkv, dhd))
        caches.append(torch.randn(past, dnkv, dhd))
    ex = (torch.randn(seq, dh), torch.zeros(seq, past + seq),
          torch.randn(seq, dhd), torch.randn(seq, dhd), *caches)
    shapes = [[seq, dh], [seq, past + seq], [seq, dhd], [seq, dhd]]
    shapes += [[past, dnkv, dhd]] * len(caches)
    export(dec, ex, shapes, os.path.join(args.out, "decoder"))
    print("[done] wrote ncnn sub-graphs to", args.out)


if __name__ == "__main__":
    main()
