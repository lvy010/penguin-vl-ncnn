#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Numerical alignment check: compare the exported ncnn sub-graphs (embed ->
# cacheless decoder -> lm_head) against the PyTorch reference on the same token
# ids. Reports max/mean abs error per stage and whether the argmax next-token
# matches.
#
# Usage:
#   python tools/align_check.py --model .\Penguin-VL-2B --ncnn .\penguin-vl-2b-ncnn --text "你好"

import argparse
import numpy as np
import torch


def rope_full(n, head_dim, theta, pos0=0):
    """Replicates generate_rope_embed_cache_full (ncnn_llm): cos/sin of shape
    (n, head_dim) with the [f, f] duplicated layout used by rotate_half."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(half) * 2.0 / head_dim))
    cos = np.zeros((n, head_dim), dtype=np.float32)
    sin = np.zeros((n, head_dim), dtype=np.float32)
    for i in range(n):
        t = (pos0 + i) * inv
        cos[i, :half] = np.cos(t); cos[i, half:] = np.cos(t)
        sin[i, :half] = np.sin(t); sin[i, half:] = np.sin(t)
    return cos, sin


def causal_mask(n):
    m = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        m[i, i + 1:] = -1e38
    return m


def stats(name, a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    if a.shape != b.shape:
        print(f"[{name}] SHAPE MISMATCH ncnn={a.shape} torch={b.shape}")
        n = min(a.size, b.size)
        a = a.flatten()[:n]; b = b.flatten()[:n]
    diff = np.abs(a - b)
    denom = np.abs(b).mean() + 1e-9
    print(f"[{name}] max_abs={diff.max():.4e} mean_abs={diff.mean():.4e} "
          f"rel_mean={diff.mean()/denom:.4e} torch|mean|={np.abs(b).mean():.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ncnn", required=True)
    ap.add_argument("--text", default="你好")
    args = ap.parse_args()

    import ncnn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok(args.text, return_tensors="pt").input_ids[0].tolist()
    n = len(ids)
    print("tokens:", ids, "n=", n)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 trust_remote_code=True).float().eval()
    cfg = model.config
    H = cfg.hidden_size
    hd = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
    theta = getattr(cfg, "rope_theta", 1e6)

    # ---- torch reference ----
    with torch.no_grad():
        emb_t = model.model.embed_tokens(torch.tensor([ids]))          # (1,n,H)
        out = model.model(inputs_embeds=emb_t)
        hid_t = out.last_hidden_state[0]                               # (n,H)
        logits_t = model.lm_head(hid_t)                               # (n,vocab)
    emb_t = emb_t[0].numpy()
    hid_t = hid_t.numpy()
    logits_t = logits_t.numpy()

    cos, sin = rope_full(n, hd, theta)
    mask = causal_mask(n)

    def run(param, binf, feeds, outname="out0"):
        net = ncnn.Net()
        net.opt.num_threads = 4
        net.opt.use_vulkan_compute = False
        net.load_param(f"{args.ncnn}/{param}")
        net.load_model(f"{args.ncnn}/{binf}")
        ex = net.create_extractor()
        mats = []
        for name, arr in feeds:
            m = ncnn.Mat(np.ascontiguousarray(arr))
            mats.append(m)  # keep alive during extract
            ex.input(name, m)
        ret, out = ex.extract(outname)
        res = np.array(out).copy()
        del out
        del ex
        del mats
        net.clear()
        del net
        return res

    # ---- ncnn embed ----
    ids_i = np.array(ids, dtype=np.int32)  # embed graph input (i64->int in ncnn)
    emb_n = run("embed.ncnn.param", "embed.ncnn.bin", [("in0", ids_i)])
    print("emb_n shape", emb_n.shape, "emb_t shape", emb_t.shape)
    stats("embed", emb_n.reshape(emb_t.shape), emb_t)

    # ---- ncnn lm_head on last position (feed torch hidden) ----
    last_t = hid_t[-1:].astype(np.float32)  # (1,H)
    log_n = run("lm_head.ncnn.param", "lm_head.ncnn.bin", [("in0", last_t)])
    stats("lm_head", log_n.flatten(), logits_t[-1])
    print("torch argmax next:", int(logits_t[-1].argmax()),
          "=", repr(tok.decode([int(logits_t[-1].argmax())])))
    print("ncnn  argmax next:", int(log_n.flatten().argmax()),
          "=", repr(tok.decode([int(log_n.flatten().argmax())])))

    # ---- ncnn decoder (feed torch embed so errors don't compound) ----
    hid_n = run("decoder_nocache.ncnn.param", "decoder_nocache.ncnn.bin",
                [("in0", emb_t.astype(np.float32)),
                 ("in1", mask), ("in2", cos), ("in3", sin)])
    print("hid_n shape", hid_n.shape)
    stats("decoder", hid_n.reshape(hid_t.shape), hid_t)


if __name__ == "__main__":
    main()
