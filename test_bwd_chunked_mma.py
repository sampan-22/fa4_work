"""Correctness checks for SM90 sparse backward N64 -> N128 chunking."""

import os

import torch

from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd


BLK = 64
D = 128
ENV = "FLASH_ATTN_SM90_SPARSE_BWD_M64N128_CHUNKED_MMA"


def make_sparse(selected):
    """Build matching forward and transposed-backward full-block metadata."""
    batch, heads, nq, topk = selected.shape
    nkv = int(selected.max().item()) + 1
    # Tests pass nkv explicitly by ensuring the last block occurs at least once.
    fwd_cnt = torch.full((batch, heads, nq), topk, dtype=torch.int32, device="cuda")
    zero_cnt = torch.zeros_like(fwd_cnt)
    fwd = BlockSparseTensorsTorch(
        zero_cnt,
        torch.zeros_like(selected),
        fwd_cnt,
        selected,
        (BLK, BLK),
    )

    per_n = [[[] for _ in range(nkv)] for _ in range(heads)]
    for h in range(heads):
        for m in range(nq):
            for n in selected[0, h, m].tolist():
                per_n[h][n].append(m)
    max_count = max(1, max(len(x) for by_n in per_n for x in by_n))
    bwd_idx = torch.zeros((batch, heads, nkv, max_count), dtype=torch.int32, device="cuda")
    bwd_cnt = torch.zeros((batch, heads, nkv), dtype=torch.int32, device="cuda")
    for h in range(heads):
        for n in range(nkv):
            vals = per_n[h][n]
            bwd_cnt[0, h, n] = len(vals)
            if vals:
                bwd_idx[0, h, n, : len(vals)] = torch.tensor(vals, dtype=torch.int32, device="cuda")
    bwd = BlockSparseTensorsTorch(
        torch.zeros_like(bwd_cnt),
        torch.zeros_like(bwd_idx),
        bwd_cnt,
        bwd_idx,
        (BLK, BLK),
    )
    return fwd, bwd


def reference(q, k, v, dout, selected):
    qr = q.float().detach().requires_grad_(True)
    kr = k.float().detach().requires_grad_(True)
    vr = v.float().detach().requires_grad_(True)
    batch, seqlen, heads, _ = qr.shape
    nkv_heads = kr.shape[2]
    assert heads % nkv_heads == 0
    kx = kr.repeat_interleave(heads // nkv_heads, dim=2)
    vx = vr.repeat_interleave(heads // nkv_heads, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", qr, kx) * (D ** -0.5)
    keep = torch.zeros_like(scores, dtype=torch.bool)
    for h in range(heads):
        for m in range(seqlen // BLK):
            for n in selected[0, h, m].tolist():
                keep[0, h, m * BLK : (m + 1) * BLK, n * BLK : (n + 1) * BLK] = True
    probs = scores.masked_fill(~keep, -torch.inf).softmax(-1)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, vx)
    return torch.autograd.grad(out, (qr, kr, vr), dout.float())


def run_case(name, selected, hkv, mixed_lists=False):
    torch.manual_seed(123)
    _, heads, nq, _ = selected.shape
    nkv = int(selected.max().item()) + 1
    assert nkv % 2 == 0
    q = torch.randn(1, nq * BLK, heads, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, nkv * BLK, hkv, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    dout = torch.randn_like(q)
    fwd_sparse, bwd_sparse = make_sparse(selected)
    if mixed_lists:
        full_cnt = bwd_sparse.full_block_cnt.clone()
        full_idx = bwd_sparse.full_block_idx.clone()
        mask_cnt = (full_cnt > 0).to(torch.int32)
        mask_idx = torch.zeros_like(full_idx)
        mask_idx[..., 0] = full_idx[..., 0]
        if full_idx.shape[-1] > 1:
            full_idx[..., :-1] = full_idx[..., 1:].clone()
        full_cnt -= mask_cnt
        bwd_sparse = BlockSparseTensorsTorch(
            mask_cnt, mask_idx, full_cnt, full_idx, (BLK, BLK)
        )

    os.environ["FLASH_ATTN_SM90_SPARSE_M64N128_CHUNKED_MMA"] = "0"
    out, lse = _flash_attn_fwd(
        q, k, v, tile_mn=(BLK, BLK), block_sparse_tensors=fwd_sparse,
        causal=False, return_lse=True,
    )
    os.environ[ENV] = "0"
    baseline = _flash_attn_bwd(
        q, k, v, out, dout, lse, causal=False, block_sparse_tensors=bwd_sparse
    )
    os.environ[ENV] = "1"
    chunked = _flash_attn_bwd(
        q, k, v, out, dout, lse, causal=False, block_sparse_tensors=bwd_sparse
    )
    ref = reference(q, k, v, dout, selected)
    torch.cuda.synchronize()

    cb = [float((c.float() - b.float()).abs().max()) for c, b in zip(chunked, baseline)]
    cr = [float((c.float() - r).abs().max()) for c, r in zip(chunked, ref)]
    br = [float((b.float() - r).abs().max()) for b, r in zip(baseline, ref)]
    finite = all(torch.isfinite(x).all() for x in chunked)
    # Chunked and N64 baseline change WGMMA accumulation order.  Require the
    # chunked error to stay within a small bf16 allowance of baseline.
    ok = finite and all(c <= b + 0.08 for c, b in zip(cr, br)) and max(cb) < 0.16
    print(
        f"{'PASS' if ok else 'FAIL'} {name}: "
        f"chunk-baseline={cb} chunk-ref={cr} baseline-ref={br} finite={finite}"
    )
    return ok


def main():
    # Ensure all eight KV cubes are represented so metadata has an even N extent.
    selected_h1 = torch.tensor(
        [[[[0, 1, 7], [0, 2, 7], [3, 4, 7], [4, 5, 7], [1, 6, 7], [2, 6, 7]]]],
        dtype=torch.int32,
        device="cuda",
    )
    selected_gqa = selected_h1.expand(1, 2, -1, -1).contiguous()
    passed = [
        run_case("single_head_overlap_and_half_only", selected_h1, hkv=1),
        run_case("gqa_two_heads", selected_gqa, hkv=1),
        run_case("mixed_partial_and_full_lists", selected_h1, hkv=1, mixed_lists=True),
    ]
    raise SystemExit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
