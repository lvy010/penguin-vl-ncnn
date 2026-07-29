"""Dependency-free regression for the explicit full-width vision RoPE."""
import numpy as np


def explicit(q, cos, sin):
    half = q.shape[-1] // 2
    if q.shape[-1] % 2:
        raise ValueError("head_dim must be even")
    q1, q2 = q[..., :half], q[..., half:]
    return np.concatenate((q1 * cos[..., :half] - q2 * sin[..., :half],
                           q2 * cos[..., half:] + q1 * sin[..., half:]), -1)


for dim in (8, 128):
    q = np.arange(2 * 3 * dim, dtype=np.float32).reshape(2, 3, dim) / 17
    cos = (np.arange(3 * dim, dtype=np.float32).reshape(3, dim) + 3) / 19
    sin = (np.arange(3 * dim, dtype=np.float32).reshape(3, dim) + 5) / 23
    half = dim // 2
    rotate = np.concatenate((-q[..., half:], q[..., :half]), -1)
    reference = q * cos[None] + rotate * sin[None]
    got = explicit(q, cos[None], sin[None])
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-6)

print("UNFUSED ROPE SELFTEST PASS")
