"""Correctness harness for fa4_hybrid_v2 (SM90 ping-pong block-sparse path).

Compares _flash_attn_fwd with FLASH_ATTN_SM90_PP=1 (two consumer warpgroups
splitting each query cube's block list by stage parity, SplitKV-style merge)
against the stock single-consumer-WG path and an fp32 torch reference over
the selected cubes, on identical inputs/selection.

Covers even topk, odd topk, topk=1 (WG1 idle), topk=2 (WG1 gets exactly one
tile), GQA (Hq=8, Hkv=1, the VSA bench shape) and MHA.
"""

import os
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


def run_kernel(q, k, v, bs, ping_pong):
    os.environ["FLASH_ATTN_SM90_PP"] = "1" if ping_pong else "0"
    out, _ = _flash_attn_fwd(q, k, v, block_sparse_tensors=bs, tile_mn=(64, 64))
    return out


def run_case(B, H, Hkv, nq, nkv, topk, seed):
    torch.manual_seed(seed)
    dev = "cuda"
    D = 128
    q = torch.randn(B, nq * BLK, H, D, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, nkv * BLK, Hkv, D, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, nkv * BLK, Hkv, D, dtype=torch.bfloat16, device=dev)
    sel = torch.stack(
        [
            torch.sort(torch.randperm(nkv, device=dev)[:topk]).values
            for _ in range(B * H * nq)
        ]
    ).view(B, H, nq, topk).to(torch.int32)

    bs = make_sparse(sel, nkv)
    out_stock = run_kernel(q, k, v, bs, ping_pong=False)
    out_v2 = run_kernel(q, k, v, bs, ping_pong=True)
    ref = ref_attention(q, k, v, sel)

    err_stock = (out_stock.float() - ref).abs().max().item()
    err_v2 = (out_v2.float() - ref).abs().max().item()
    err_cross = (out_v2.float() - out_stock.float()).abs().max().item()
    tag = f"B={B} H={H} Hkv={Hkv} nq={nq} nkv={nkv} topk={topk}"
    # v2 changes the accumulation split (two partials merged once in f32),
    # so results are bit-different but must stay as accurate as stock.
    ok = err_v2 < max(2.0 * err_stock, 1e-2) and err_v2 < 0.1
    print(
        f"{'PASS' if ok else 'FAIL'} {tag}: max|v2-ref|={err_v2:.3e} "
        f"max|stock-ref|={err_stock:.3e} max|v2-stock|={err_cross:.3e}"
    )
    return ok


def main():
    cases = [
        # GQA, VSA-like: Hq=8, Hkv=1; even topk (both WGs equal tiles)
        dict(B=2, H=8, Hkv=1, nq=16, nkv=16, topk=4, seed=0),
        # odd topk (WG0 gets one more tile than WG1)
        dict(B=2, H=8, Hkv=1, nq=16, nkv=16, topk=3, seed=1),
        # topk=1: WG1 has no tiles, must contribute neutral partial
        dict(B=1, H=8, Hkv=1, nq=8, nkv=8, topk=1, seed=2),
        # topk=2: WG1 gets exactly one tile (first == last)
        dict(B=1, H=8, Hkv=1, nq=8, nkv=8, topk=2, seed=5),
        # larger, odd, uneven nkv (VSA 16k-ish topk)
        dict(B=1, H=8, Hkv=1, nq=12, nkv=33, topk=33, seed=3),
        # MHA
        dict(B=2, H=2, Hkv=2, nq=8, nkv=16, topk=5, seed=4),
    ]
    results = [run_case(**c) for c in cases]
    print("ALL PASS" if all(results) else "SOME FAILED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
