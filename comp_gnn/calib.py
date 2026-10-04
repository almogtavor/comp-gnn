import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.utils.model_definitions.text_automodel_wrapper import get_model_path
from comp_gnn.components import geom, is_moe, moe_forward, parts, weights
from comp_gnn.ntp import Feats, chunks
from comp_gnn.run import DEV

P = 256


def heads_mlp(z, mlp, W, R, H):
    """z (B, T, nl, H*hd), mlp (B, T, nl, D), W (nl, D, H*hd), R (D, P) -> unit-norm JL contributions (B*T, nl*(H+1), P)."""
    B, T, nl, _ = z.shape
    WR = torch.einsum("ldhk,dp->lhkp", W.reshape(nl, W.shape[1], H, -1), R)
    c = torch.cat([torch.einsum("btlhk,lhkp->btlhp", z.reshape(B, T, nl, H, -1), WR), (mlp @ R)[:, :, :, None]], 3)
    return torch.nn.functional.normalize(c.reshape(B * T, nl * (H + 1), P), dim=-1)


def xedges(G, per, k):
    """G (n, n) token-mean cosine; per = components per block (None: any other component, for comp_similarity).
    Top-k positive partners in other blocks, symmetric union."""
    blk = torch.arange(len(G)) // (per or 1)
    s = G.clamp(min=0).masked_fill(blk[:, None] == blk[None], 0)
    v, j = s.topk(k, 1)
    A = torch.zeros_like(s, dtype=torch.bool)
    A[torch.arange(len(s))[:, None].expand_as(j)[v > 0], j[v > 0]] = True
    return (A | A.T).nonzero().T


def lift(C, cnt, n):
    """C (m, m) co-routing counts over n tokens, cnt (m,) routing counts -> log lift (-inf where never co-routed)."""
    return torch.log(C * n / (cnt[:, None] * cnt[None]).clamp(min=1))


def remap(e, per, off, per_to):  # component ids in blocks of `per` -> slot off.. of blocks of `per_to`
    return e // per * per_to + off + e % per


def calibrate(batches, W, H, k=4, seed=0):
    """batches: iterable of (z, mlp) per-token features. Returns (xedges, G)."""
    R = torch.randn(W.shape[1], P, generator=torch.Generator().manual_seed(seed)).div(P ** 0.5).to(W.device)
    G, n = 0, 0
    for z, mlp in batches:
        c = heads_mlp(z, mlp, W, R, H)
        G, n = G + torch.einsum("tnp,tmp->nm", c, c), n + len(c)
    G = (G / n).cpu()
    return xedges(G, H + 1, k), G


def selfcheck():
    g = torch.Generator().manual_seed(0)
    nl, H, hd, D = 4, 3, 5, 16
    W = torch.randn(nl, D, H * hd, generator=g)
    W[2, :, hd:2 * hd] = W[0, :, :hd]  # twin: head (0, 0) copied (z and W_O block) into slot (2, 1)
    W[3, :, :hd] = W[0, :, hd:2 * hd]  # anti: head (0, 1) negated into slot (3, 0)
    batches = []
    for _ in range(3):
        z, mlp = torch.randn(2, 50, nl, H * hd, generator=g), torch.randn(2, 50, nl, D, generator=g)
        z[:, :, 2, hd:2 * hd] = z[:, :, 0, :hd]
        z[:, :, 3, :hd] = -z[:, :, 0, hd:2 * hd]
        batches.append((z, mlp))
    e, G = calibrate(batches, W, H, k=2)
    per = H + 1
    twin, anti = (0, 2 * per + 1), (1, 3 * per)
    assert G[twin].item() > 0.99 and G[anti].item() < -0.99, (G[twin], G[anti])
    assert G[0, per:].argmax().item() + per == twin[1], "twin is not head (0, 0)'s top cross-block partner"
    E = set(map(tuple, e.T.tolist()))
    assert twin in E and twin[::-1] in E and anti not in E and anti[::-1] not in E
    assert all((a, b)[::-1] in E for a, b in E) and all(a // per != b // per for a, b in E)  # symmetric, cross-block only
    M = (torch.rand(1000, 3 * 4, generator=g) < 0.3).float()  # 3 blocks x 4 experts
    M[:, 9] = M[:, 1]  # expert (0, 1) always co-routed with (2, 1)
    Le = set(map(tuple, xedges(lift(M.T @ M, M.sum(0), len(M)), 4, 1).T.tolist()))
    assert (1, 9) in Le and (9, 1) in Le and all(a // 4 != b // 4 for a, b in Le)
    assert remap(torch.tensor([0, 4, 5]), 4, 0, 9).tolist() == [0, 9, 10] and remap(torch.tensor([9]), 4, 5, 9).tolist() == [24]
    S = set(map(tuple, xedges(G, None, 2).T.tolist()))
    assert twin in S and anti not in S and all(a != b for a, b in S) and any(a // per == b // per for a, b in S)  # same block allowed
    print("calib ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family")
    ap.add_argument("--model_size")
    ap.add_argument("--out")
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    path = get_model_path(a.model_family, a.model_size)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(DEV)
    feats, X = Feats(model), chunks(tok, "fineweb", 0, a.tokens)
    W, H = weights(model)[0].to(DEV), geom(model)[0]
    moe, cur, co = is_moe(model), {}, {"C": 0, "cnt": 0, "n": 0}
    if moe:
        for i, layer in enumerate(parts(model)[0]):
            moe_forward(layer.mlp, i, lambda x: x, cur)

    def batches():  # position 0 (BOS / attention sink) is dropped: its huge, input-independent contributions would dominate
        for i in range(0, len(X), a.bs):
            cur.update({k: [] for k in ("gate", "exp", "shared", "routed")})
            f = feats(X[i:i + a.bs].to(DEV))
            if not moe:
                yield f["z"][:, 1:].float(), f["mlp"][:, 1:].float()
                continue
            M = (torch.stack(cur["gate"], 2)[:, 1:] > 0).flatten(0, 1).flatten(1).float()  # (tokens, nl*E) routed
            co.update(C=co["C"] + M.T @ M, cnt=co["cnt"] + M.sum(0), n=co["n"] + len(M))
            yield f["z"][:, 1:].float(), torch.stack(cur["shared"], 2)[:, 1:]
    e, G = calibrate(batches(), W, H, a.k)
    deg = torch.bincount(e[0], minlength=len(G)).float()
    print(f"[calib] {a.model_family}: tokens={X[:, 1:].numel()} components={len(G)} edges={e.shape[1] // 2} "
          f"degree mean={deg.mean():.1f} max={deg.max():.0f} edge cos median={G[e[0], e[1]].median():.3f}", flush=True)
    np.save(a.out, e.numpy())
    sim = xedges(G, None, a.k)  # comp_similarity: same k, no layer structure
    np.save(a.out.removesuffix(".npy") + "_sim.npy", sim.numpy())
    if moe:
        E = model.config.get_text_config().num_experts
        Lf, per = lift(co["C"], co["cnt"], co["n"]).cpu(), H + 1 + E
        for name, base, le in (("_exp", e, xedges(Lf, E, a.k)), ("_sim_exp", sim, xedges(Lf, None, a.k))):
            np.save(a.out.removesuffix(".npy") + name + ".npy", torch.cat([remap(base, H + 1, 0, per), remap(le, E, H + 1, per)], 1).numpy())
            print(f"[calib] {name}: expert edges={le.shape[1] // 2} lift median={Lf[le[0], le[1]].exp().median():.2f}", flush=True)
    os._exit(0)  # the FineWeb streaming threads abort the interpreter at shutdown (PyGILState_Release), failing the job


if __name__ == "__main__":
    main()
