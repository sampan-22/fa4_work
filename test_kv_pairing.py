"""Correctness harness for the SM90 KV-pairing path (kv_pair_factor=2).

Compares _flash_attn_fwd with tile_mn=(64,128) + sparse block_size=(64,64)
(the paired path) against:
  1. the unpaired path (tile_mn=(64,64), same sparse tensors), and
  2. an fp32 torch reference over the selected cubes.

Covers even topk, odd topk (duplicate-tail masking), topk=1, GQA (Hq=8,
Hkv=1, mirroring the VSA bench shapes) and MHA.
"""

import torch

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

BLK = 64


def make_sparse(sel, nkv):
    """sel: (B, H, nq, topk) sorted int32 cube indices -> BlockSparseTensorsTorch."""
    B, H, nq, topk = sel.shape
    dev = sel.device
    full_block_idx = torch.zeros((B, H, nq, nkv), dtype=torch.int32, device=dev)
    full_block_idx[..., :topk] = sel
    full_block_cnt = torch.full((B, H, nq), topk, dtype=torch.int32, device=dev)
    mask_block_idx = torch.zeros((B, H, nq, nkv), dtype=torch.int32, device=dev)
    mask_block_cnt = torch.zeros((B, H, nq), dtype=torch.int32, device=dev)
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_block_cnt,
        mask_block_idx=mask_block_idx,
        full_block_cnt=full_block_cnt,
        full_block_idx=full_block_idx,
        block_size=(BLK, BLK),
    )


def ref_attention(q, k, v, sel):
    B, S, H, D = q.shape
    Hkv = k.shape[2]
    nq = S // BLK
    group = H // Hkv
    out = torch.zeros(B, S, H, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        for h in range(H):
            hk = h // group
            for qc in range(nq):
                qs = q[b, qc * BLK : (qc + 1) * BLK, h].float()
                kk = torch.cat(
                    [k[b, c * BLK : (c + 1) * BLK, hk] for c in sel[b, h, qc].tolist()]
                ).float()
                vv = torch.cat(
                    [v[b, c * BLK : (c + 1) * BLK, hk] for c in sel[b, h, qc].tolist()]
                ).float()
                s = qs @ kk.T * (D ** -0.5)
                out[b, qc * BLK : (qc + 1) * BLK, h] = s.softmax(-1) @ vv
    return out


def run_case(B, H, Hkv, nq, nkv, topk, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    D = 128
    q = torch.randn(B, nq * BLK, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, nkv * BLK, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, nkv * BLK, Hkv, D, dtype=torch.bfloat16, device=dev)
    # random distinct cube selections per (b, h, qcube), sorted like fa4_hybrid
    sel = torch.stack(
        [
            torch.sort(torch.randperm(nkv, device=dev)[:topk]).values
            for _ in range(B * H * nq)
        ]
    ).view(B, H, nq, topk).to(torch.int32)

    bs = make_sparse(sel, nkv)
    out_unpaired, _ = _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
    out_paired, _ = _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 128))
    ref = ref_attention(q, k, v, sel)

    err_unpaired = (out_unpaired.float() - ref).abs().max().item()
    err_paired = (out_paired.float() - ref).abs().max().item()
    err_cross = (out_paired.float() - out_unpaired.float()).abs().max().item()
    tag = f"B={B} H={H} Hkv={Hkv} nq={nq} nkv={nkv} topk={topk}"
    # bf16 tolerance: paired must be as accurate as unpaired (same math,
    # different tiling); both are compared to the fp32 reference.
    ok = err_paired < max(2.5 * err_unpaired, 1e-2) and err_paired < 0.1
    print(
        f"{'PASS' if ok else 'FAIL'} {tag}: max|paired-ref|={err_paired:.3e} "
        f"max|unpaired-ref|={err_unpaired:.3e} max|paired-unpaired|={err_cross:.3e}"
    )
    return ok


def main():
    cases = [
        # GQA, VSA-like: Hq=8, Hkv=1; even topk
        dict(B=2, H=8, Hkv=1, nq=16, nkv=16, topk=4, seed=0),
        # odd topk (duplicate-tail mask path)
        dict(B=2, H=8, Hkv=1, nq=16, nkv=16, topk=3, seed=1),
        # odd, single block (pair = duplicated block only)
        dict(B=1, H=8, Hkv=1, nq=8, nkv=8, topk=1, seed=2),
        # odd, larger, uneven nkv
        dict(B=1, H=8, Hkv=1, nq=12, nkv=33, topk=33, seed=3),
        # MHA
        dict(B=2, H=2, Hkv=2, nq=8, nkv=16, topk=5, seed=4),
    ]
    results = [run_case(**c) for c in cases]
    print("ALL PASS" if all(results) else "SOME FAILED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
