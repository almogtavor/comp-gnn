"""CompGNN: the component-level GNN with learned type / layer / head embeddings and a per-type mean readout (768-d).
Used by the gated-LoRA gate (ntp.GATE) and the earlier comp_gnn / comp_noedge / comp_cayley / comp_resonly / comp_noexp
rows (dropped from the tables: more parameters than gin_cayley). Routed from run.build.
"""
import itertools

import torch
import torch.nn as nn

from comp_gnn.run import cayley_sl2

# ------- component-level GNN: residual nodes r_0..r_L, head nodes (l,h), MLP (MoE block) nodes l, expert nodes (l,e) -------
def component_graph(L, nl, nh, edges=True, residual_only=False, ne=0):
    """Node order: residual r_0..r_{L-1}, heads (l, h) row-major, MLPs, then experts (l, e) row-major (type 3, read r_l
    and write r_{l+1} like heads; id nh+1+e). Returns dense A, type, layer, head/expert ids."""
    nr = L
    n = nr if residual_only else nr + nl * nh + nl + nl * ne
    A = torch.zeros(n, n)
    off = L - 1 - nl  # r_{off+l} is block l's input, r_{off+l+1} its output (off = 0 when r_0 is the embedding)
    typ = torch.zeros(n, dtype=torch.long)
    lay = torch.zeros(n, dtype=torch.long)
    hid = torch.full((n,), nh, dtype=torch.long)
    lay[:nr] = torch.arange(nr)
    for i in range(nr - 1):
        A[i, i + 1] = A[i + 1, i] = 1
    if not residual_only:
        for l in range(nl):
            src, dst = off + l, off + l + 1
            comps = [nr + l * nh + h for h in range(nh)] + [nr + nl * nh + l] + [nr + nl * (nh + 1) + l * ne + e for e in range(ne)]
            for j, c in enumerate(comps):
                A[src, c] = A[c, src] = A[dst, c] = A[c, dst] = 1
                typ[c] = 1 if j < nh else 2 if j == nh else 3
                lay[c] = l + 1 + off
                hid[c] = j
    if not edges:
        A.zero_()
    return A, typ, lay, hid


class CompGNN(nn.Module):
    """ILSE-style GIN (proj_in, ReLU, dropout, [GIN(MLP) -> LayerNorm -> dropout] x nl) over the component graph,
    plus learned type/layer/head embeddings and a per-type mean readout. Inputs are LayerNormed per node because
    head/MLP outputs and the residual stream have very different scales (ponytail: no learned per-type scale)."""

    def __init__(self, L, D, C, nl, dropout, W_O, nheads=16, edges=True, residual_only=False, cayley=False, nexp=0, kexp=32):
        super().__init__()
        self.residual_only, self.nexp = residual_only, nexp
        self.ntypes = 1 if residual_only else 4 if nexp else 3
        nblk = W_O.shape[0]
        A, typ, lay, hid = component_graph(L, nblk, nheads, edges, residual_only, nexp)
        self.nreal, self.rtypes = len(typ), typ.unique().tolist()
        if cayley:  # same nodes on ILSE's SL(2,Z_n) Cayley graph; zero-feature virtual nodes get type ntypes, not read out
            n = len(typ)
            ei = next(e for e in map(cayley_sl2, itertools.count(2)) if e.max() + 1 >= n)  # ILSE's smallest SL(2,Z_k) >= n
            m = int(ei.max()) + 1
            A = torch.zeros(m, m).index_put_((ei[1], ei[0]), torch.ones(ei.shape[1]), accumulate=True)  # GIN sum over edges
            typ, lay, hid = torch.cat([typ, typ.new_full((m - n,), self.ntypes)]), torch.cat([lay, lay.new_zeros(m - n)]), torch.cat([hid, hid.new_full((m - n,), nheads)])
        # ponytail: sparse aggregation only for the MoE-size graphs (~8k nodes); small graphs keep the dense einsum
        self.sparse = len(typ) > 4096
        self.register_buffer("A", A.to_sparse() if self.sparse else A, persistent=not self.sparse)
        for k, v in (("typ", typ), ("lay", lay), ("hid", hid)):
            self.register_buffer(k, v)
        if not residual_only:
            hd = W_O.shape[2] // nheads  # o_proj input H*hd can differ from D (Gemma2: 2048 vs 2304)
            self.register_buffer("W", W_O.reshape(nblk, D, nheads, hd))  # (nl, D, H, hd)
        self.nheads = nheads
        H = 256
        self.norm_in = nn.LayerNorm(D, elementwise_affine=False)
        self.proj_in = nn.Linear(D, H)
        self.temb, self.lemb, self.hemb = nn.Embedding(self.ntypes + cayley, H), nn.Embedding(L, H), nn.Embedding(nheads + 1 + nexp, H)
        if nexp:  # expert node input: LayerNormed JL projection of g_e * E_e(x), plus the mean gate g_e
            self.norm_e, self.proj_e = nn.LayerNorm(kexp, elementwise_affine=False), nn.Linear(kexp + 1, H)
        self.mlps = nn.ModuleList([nn.Sequential(nn.Linear(H, H), nn.ReLU()) for _ in range(nl)])
        self.eps = nn.Parameter(torch.zeros(nl))
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(nl)])
        self.drop = nn.Dropout(dropout)
        self.out_dim = H * len(self.rtypes)
        self.head = nn.Linear(self.out_dim, C)

    def nodes(self, b):
        if self.residual_only:
            return b["r"]
        B, nl, _ = b["z"].shape  # z width is H*hd, not D (Gemma2)
        z = b["z"].float().reshape(B, nl, self.nheads, -1)
        heads = torch.einsum("blhk,ldhk->blhd", z, self.W).reshape(B, nl * self.nheads, -1)
        return torch.cat([b["r"], heads, b["mlp"].float()], 1)

    def agg(self, x):
        if not self.sparse:
            return torch.einsum("ij,bjd->bid", self.A, x)
        B, n, H = x.shape
        return torch.sparse.mm(self.A, x.transpose(0, 1).reshape(n, B * H)).reshape(n, B, H).transpose(0, 1)

    def embed(self, b):
        x = self.proj_in(self.norm_in(self.nodes(b)))
        B = x.shape[0]
        if self.nexp:
            e = b["exp"].float().reshape(B, -1, self.norm_e.normalized_shape[0])
            x = torch.cat([x, self.proj_e(torch.cat([self.norm_e(e), b["gate"].float().reshape(B, -1, 1)], -1))], 1)
        # Cayley virtual nodes: zero input, i.e. proj_in(norm_in(0)) = proj_in.bias
        x = torch.cat([x, self.proj_in.bias.expand(B, len(self.typ) - self.nreal, -1)], 1)
        x = self.drop(torch.relu(x + self.temb(self.typ) + self.lemb(self.lay) + self.hemb(self.hid)))
        for mlp, norm, eps in zip(self.mlps, self.norms, self.eps):
            x = self.drop(norm(mlp((1 + eps) * x + self.agg(x))))
        return torch.cat([x[:, self.typ == t].mean(1) for t in self.rtypes], -1)

def build(method, L, D, C, nl, dropout, W_O, nheads, nexp):
    if method == "comp_noexp":  # MoE ablation: same graph without the expert nodes
        return CompGNN(L, D, C, nl, dropout, W_O, nheads=nheads)
    kw = {"comp_noedge": {"edges": False}, "comp_cayley": {"cayley": True}, "comp_resonly": {"residual_only": True}}.get(method, {})
    return CompGNN(L, D, C, nl, dropout, W_O, nheads=nheads, nexp=0 if method == "comp_resonly" else nexp, **kw)


if __name__ == "__main__":
    A, typ, lay, hid = component_graph(25, 24, 16)
    assert A.shape == (433, 433) and (A == A.T).all() and (typ == 1).sum() == 384 and (typ == 2).sum() == 24
    assert A[0, 25] == 1 and A[1, 25] == 1 and A[2, 25] == 0  # head (0,0) reads r_0, writes r_1
    assert A[24, 25 + 384 + 23] == 1  # last MLP writes r_24
    g = CompGNN(25, 8, 2, 1, 0.0, torch.zeros(24, 8, 16 * 2), cayley=True)
    assert g.A.shape == (648, 648) and (g.A == g.A.T).all() and (g.typ == 3).sum() == 648 - 433 and g.A.sum(1).min() > 0
    assert g.embed({"r": torch.randn(3, 25, 8), "z": torch.randn(3, 24, 32), "mlp": torch.randn(3, 24, 8)}).shape == (3, 3 * 256)
    # MoE graph (Qwen3-30B-A3B size): expert (0, 0) reads r_0 / writes r_1, sparse aggregation == dense
    A, typ, lay, hid = component_graph(49, 48, 32, ne=128)
    e00 = 49 + 48 * 33
    assert len(typ) == 49 + 48 * (32 + 1 + 128) and (typ == 3).sum() == 48 * 128 and hid[e00] == 33 and A[0, e00] == A[1, e00] == 1
    m = CompGNN(49, 8, 2, 2, 0.0, torch.zeros(48, 8, 32 * 2), nheads=32, nexp=128)
    b = {"r": torch.randn(2, 49, 8), "z": torch.randn(2, 48, 64), "mlp": torch.randn(2, 48, 8),
         "gate": torch.rand(2, 48, 128), "exp": torch.randn(2, 48, 128 * 32)}
    x = torch.randn(2, len(typ), 4)
    assert m.sparse and torch.allclose(m.agg(x), torch.einsum("ij,bjd->bid", A, x), atol=1e-4)
    assert m.embed(b).shape == (2, 4 * 256)
    print("compgnn ok")
