import itertools
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.aggr import MeanAggregation, SumAggregation

from experiments.utils.model_definitions.gnn.gnn_datasets import SingleGraphDataset
from experiments.utils.model_definitions.gnn.gnn_models import LayerGINEncoder
from comp_gnn.run import CompGIN, cayley_sl2, jl


def build(method, L, D, C, nl, dropout, W_O, nheads, nexp):
    if method.endswith("_e10"):
        assert nexp, "type weights need expert nodes"
        method, tw = method.removesuffix("_e10"), torch.tensor([.45, .45, .1])
        if method.startswith("comp_hier_"):
            xe = torch.from_numpy(np.load(os.environ["XEDGES"])) if method == "comp_hier_xl" else None
            return HierGIN(L, D, C, nl, dropout, W_O, nheads, nexp, variant="bal", xedges=xe, tw=tw)
        g = build(method, L, D, C, nl, dropout, W_O, nheads, nexp)
        g.register_buffer("tw", tw.to(g.ei.device))
        return g
    if method == "comp_cayley_ln":
        return CompGINLN(L, D, C, nl, dropout, W_O, nheads, nexp)
    if method in ("comp_noedge_ln", "comp_resonly_ln"):
        g = (CompGINLN if method == "comp_noedge_ln" else ResOnlyGIN)(L, D, C, nl, dropout, W_O, nheads, nexp)
        if method == "comp_resonly_ln":
            g.n_real, g.tid = L, None
        path = torch.arange(g.n_real - 1)
        ei = torch.stack([path, path + 1]) if method == "comp_resonly_ln" else torch.zeros(2, 0, dtype=torch.long)
        g.ei, g.n_nodes = torch.cat([ei, ei.flip(0)], 1).to(g.ei.device), g.n_real
        return g
    if method == "comp_similarity":
        g = CompGINLN(L, D, C, nl, dropout, W_O, nheads, nexp)
        g.ei, g.n_nodes = torch.from_numpy(np.load(os.environ["SEDGES"])).long().to(g.ei.device), g.n_real
        assert g.ei.max() < g.n_real, "similarity edges from a different model"
        return g
    if method == "comp_hier_xl":  # $XEDGES is set per model; no file = misconfigured run, fail loudly
        return HierGIN(L, D, C, nl, dropout, W_O, nheads, nexp, xedges=torch.from_numpy(np.load(os.environ["XEDGES"])))
    return HierGIN(L, D, C, nl, dropout, W_O, nheads, nexp, variant=method.removeprefix("comp_hier_"))


class CompGINLN(CompGIN):
    def nodes(self, b):
        x = CompGIN.nodes(self, b)
        return F.layer_norm(x, x.shape[-1:])


class ResOnlyGIN(CompGIN):
    def nodes(self, b):
        return F.layer_norm(b["r"].float(), b["r"].shape[-1:])


class WeightedSum(SumAggregation):  # sum of messages scaled by the per-edge weights w set before each forward
    def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
        return super().forward(x * self.w[:, None], index, ptr, dim_size, dim)


class HierGIN(CompGIN):
    def __init__(self, L, D, C, nl, dropout, W_O, nheads, nexp=0, kexp=32, variant="cayley", xedges=None, tw=None):
        nn.Module.__init__(self)
        nblk = W_O.shape[0]
        self.register_buffer("W", W_O.reshape(nblk, D, nheads, W_O.shape[2] // nheads))
        self.nheads, self.nexp, self.kexp = nheads, nexp, kexp
        if nexp:
            self.register_buffer("P", jl(nblk, D, kexp))
        self.n_real = nblk * (nheads + 1 + nexp)  # component count; CompGIN.nodes reshapes to it
        ds = SingleGraphDataset([np.zeros((L, 1), np.float32)], [0], "cayley", keep_embedding_layer=True)  # = gin_cayley's graph
        assert ds.items[0].shape[0] == L
        self.L, self.m = L, ds.cayley_num_nodes or L
        comp = torch.arange(self.n_real)
        # block l (0-based) writes r_{L-nblk+l}: r_0 is the embedding output when L = nblk + 1
        star = torch.stack([self.m + comp, L - nblk + comp // (nheads + 1 + nexp)])
        xe = torch.zeros(2, 0, dtype=torch.long) if xedges is None else xedges.long() + self.m  # component ids -> node ids
        assert xe.numel() == 0 or xe.max() < self.m + self.n_real, "xedges from a different model"
        self.register_buffer("ei", torch.cat([ds.edge_index, star, star.flip(0), xe], 1))
        self.n_nodes = self.m + self.n_real
        assert variant in ("cayley", "readout", "deep", "mean", "bal", "gate"), variant
        assert variant != "gate" or nexp, "router gates need expert nodes"
        self.variant = variant
        self.enc = LayerGINEncoder(D, 256, nl + (variant == "deep"), dropout, 1, "mean", "cayley", pool_real_nodes_only=False,
                                   train_eps=True)
        if variant == "mean":
            for c in self.enc.gnn_layers:
                c.aggr, c.aggr_module = "mean", MeanAggregation()
        if variant in ("bal", "gate"):  # component -> mega edges weighted 1 / (count of that type in the block); all others 1
            k = comp % (nheads + 1 + nexp)
            w = torch.where(k < nheads, 1 / nheads, torch.where(k == nheads, 1.0, 1 / max(nexp, 1)))
            if tw is not None:  # _e10: per-type totals 3 x tw instead of 1 each
                w = w * 3 * tw[torch.where(k < nheads, 0, torch.where(k == nheads, 1, 2))]
            rest = self.ei.shape[1] - ds.edge_index.shape[1] - self.n_real  # mega -> component edges and xedges
            self.register_buffer("ew", torch.cat([torch.ones(ds.edge_index.shape[1]), w, torch.ones(rest)]))
            self.register_buffer("gpos", ds.edge_index.shape[1] + comp[k > nheads])  # gate: expert edge slots in ew
            self.bal = WeightedSum()
            for c in self.enc.gnn_layers:
                c.aggr_module = self.bal
        self.out_dim = 512 if variant == "readout" else 256
        self.head = nn.Linear(self.out_dim, C)

    def embed(self, b):
        r = b["r"].float()
        B, L, D = r.shape
        x = torch.cat([r, r.new_zeros(B, self.m - L, D), F.layer_norm(CompGIN.nodes(self, b), (D,))], 1)
        off = torch.arange(B, device=x.device) * self.n_nodes
        gid = torch.arange(B, device=x.device)[:, None].expand(B, self.n_nodes).clone()
        gid[:, L:] = 2 * B  # virtual nodes (and components, unless pooled below) go to a dummy graph
        if self.variant == "readout":
            gid[:, self.m:] += torch.arange(B, device=x.device)[:, None] - B  # components of graph b -> pooled graph B + b
        g = SimpleNamespace(x=x.reshape(-1, D), edge_index=(self.ei[:, None, :] + off[None, :, None]).reshape(2, -1),
                            batch=gid.reshape(-1), num_graphs=B)
        if self.variant == "bal":
            self.bal.w = self.ew.repeat(B)
        elif self.variant == "gate":  # gate (B, nblk, nexp) is ordered like the expert nodes (block-major)
            w = self.ew.repeat(B, 1)
            w[:, self.gpos] = b["gate"].float().reshape(B, -1)
            self.bal.w = w.reshape(-1)
        p = self.enc(g)
        return torch.cat([p[:B], p[B:2 * B]], 1) if self.variant == "readout" else p[:B]


if __name__ == "__main__":
    dense = lambda e, n: torch.zeros(n, n).index_put_((e[1], e[0]), torch.ones(e.shape[1]), accumulate=True)
    for (L, nblk, nh, ne), v in itertools.product(((25, 24, 16, 0), (27, 26, 8, 0), (33, 32, 32, 0), (5, 4, 2, 3)),
                                                  ("cayley", "readout", "deep", "mean", "bal")):
        D = 8
        g = HierGIN(L, D, 2, 2, 0.0, torch.randn(nblk, D, nh * 2), nh, ne, kexp=3, variant=v)
        assert len(g.enc.gnn_layers) == 2 + (v == "deep")
        m, nc = g.m, nblk * (nh + 1 + ne)
        mega = g.ei[:, (g.ei < m).all(0)]
        assert torch.equal(dense(mega, m), dense(next(e for e in map(cayley_sl2, itertools.count(2)) if e.max() + 1 == m), m))
        assert g.ei.shape[1] == mega.shape[1] + 2 * nc and not ((g.ei >= m).all(0)).any()  # no component-component edges
        A = dense(g.ei, m + nc)
        assert (A[m:].sum(1) == 1).all() and (A[:, m:].sum(0) == 1).all()  # one in- and one out-edge per component...
        blk = torch.arange(nc) // (nh + 1 + ne)
        assert (A[m + torch.arange(nc), L - nblk + blk] == 1).all() and (A[:L - nblk, m:] == 0).all()  # ...to r_{l+1}
        b = {"r": torch.randn(3, L, D), "z": torch.randn(3, nblk, nh * 2), "mlp": torch.randn(3, nblk, D)}
        if ne:
            b |= {"exp": torch.randn(3, nblk, ne * 3), "gate": torch.rand(3, nblk, ne)}
        g.eval()
        e = g.embed(b)
        assert e.shape == (3, g.out_dim)
        b["mlp"][0] += 10 * torch.randn(nblk, D)  # not a constant shift (LayerNorm removes that); components reach the readout through message passing, per graph
        e2 = g.embed(b)
        assert not torch.allclose(e[0], e2[0]) and torch.allclose(e[1:], e2[1:], atol=1e-5)
        if v == "readout":  # mega half must match the mega-only readout of the same weights (dummy-graph id differs only)
            g.variant = "cayley"
            assert torch.allclose(g.embed(b), e2[:, :256], atol=1e-5)
        if v == "mean":  # conv output = (1 + eps) x + mean of neighbors
            c, xx = g.enc.gnn_layers[0], torch.randn(3, 256)
            star = torch.tensor([[1, 2], [0, 0]])
            assert torch.allclose(c(xx, star)[0], c.nn((1 + c.eps) * xx[0] + xx[1:].mean(0)), atol=1e-5)
        if v == "bal":  # into each mega-node, each present component type's weights sum to 1
            star = g.ei[:, g.ei.shape[1] - 2 * nc:g.ei.shape[1] - nc]
            typ = (star[0] - m) % (nh + 1 + ne)
            typ = torch.where(typ < nh, 0, torch.where(typ == nh, 1, 2))
            tot = torch.zeros(L, 3).index_put_((star[1], typ), g.ew[mega.shape[1]:mega.shape[1] + nc], accumulate=True)
            assert torch.allclose(tot[L - nblk:, :2 + (ne > 0)], torch.ones(())) and (tot[:L - nblk] == 0).all()
            assert (g.ew[:mega.shape[1]] == 1).all() and (g.ew[mega.shape[1] + nc:] == 1).all()
            c, xx = g.enc.gnn_layers[0], torch.randn(4, 256)
            g.bal.w = torch.tensor([0.5, 0.5, 1.0])
            star = torch.tensor([[1, 2, 3], [0, 0, 0]])
            assert torch.allclose(c(xx, star)[0], c.nn((1 + c.eps) * xx[0] + xx[1:3].mean(0) + xx[3]), atol=1e-5)
    xe = torch.tensor([[0, 17], [17, 0]])  # head (0, 0) <-> head (1, 0), as calib.py writes them (both directions)
    g, g0 = (HierGIN(25, 8, 2, 2, 0.0, torch.randn(24, 8, 32), 16, xedges=e) for e in (xe, None))
    assert torch.equal(g.ei[:, :g0.ei.shape[1]], g0.ei) and torch.equal(g.ei[:, g0.ei.shape[1]:], xe + g.m)
    g = build("comp_cayley_ln", 25, 8, 2, 2, 0.0, torch.randn(24, 8, 32), 16, 0)
    b = {"r": torch.randn(3, 25, 8), "z": 100 * torch.randn(3, 24, 32), "mlp": 100 * torch.randn(3, 24, 8)}
    x = g.nodes(b)
    assert x.shape == (3, 24 * 17, 8) and x.mean(-1).abs().max() < 1e-4 and (x.std(-1, unbiased=False) - 1).abs().max() < 1e-3
    assert g.eval().embed(b).shape == (3, 256)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npy") as f:
        np.save(f.name, np.array([[0, 17], [17, 0]]))
        os.environ["SEDGES"] = f.name
        g = build("comp_similarity", 25, 8, 2, 2, 0.0, torch.randn(24, 8, 32), 16, 0).eval()
    assert g.n_nodes == g.n_real == 24 * 17 and g.ei.tolist() == [[0, 17], [17, 0]]
    e = g.embed(b)
    b["z"][0, 1] += 10 * torch.randn(32)  # node 17 (block 1, head 0) is isolated from node 0 except via its one edge
    assert e.shape == (3, 256) and not torch.allclose(g.embed(b)[0], e[0]) and torch.allclose(g.embed(b)[1:], e[1:], atol=1e-5)
    base = sum(p.numel() for p in __import__("comp_gnn.run").run.build("gin_cayley", 25, 8, 77, 2, 0.0, torch.randn(24, 8, 32), 16).parameters())
    for m in ("comp_noedge_ln", "comp_resonly_ln"):
        g = build(m, 25, 8, 77, 2, 0.0, torch.randn(24, 8, 32), 16, 0).eval()
        assert sum(p.numel() for p in g.parameters()) == base, m
        b = {"r": torch.randn(3, 25, 8), "z": torch.randn(3, 24, 32), "mlp": torch.randn(3, 24, 8)}
        assert g.embed(b).shape == (3, 256)
        for k, seen in (("z", m == "comp_noedge_ln"), ("r", m == "comp_resonly_ln")):  # components only / residual only
            e = g.embed(b)
            b[k][0] += 10 * torch.randn(b[k].shape[1:])
            e2 = g.embed(b)
            assert torch.allclose(e2[0], e[0], atol=1e-5) != seen and torch.allclose(e2[1:], e[1:], atol=1e-5), (m, k)
    assert build("comp_resonly_ln", 25, 8, 2, 2, 0.0, torch.randn(24, 8, 32), 16, 0).ei.shape == (2, 48)
    b = {"r": torch.randn(2, 5, 8), "z": torch.randn(2, 4, 4), "mlp": torch.randn(2, 4, 8), "exp": torch.randn(2, 4, 96)}
    for m in ("comp_hier_bal", "comp_noedge_ln"):  # _e10: same params; per-type totals / pool weights 45 / 45 / 10%
        g, g10 = (build(m + s, 5, 8, 2, 2, 0.0, torch.randn(4, 8, 4), 2, 3).eval() for s in ("", "_e10"))
        assert sum(p.numel() for p in g.parameters()) == sum(p.numel() for p in g10.parameters())
        if m == "comp_hier_bal":
            E, nc = g.ei.shape[1] - 2 * g.n_real, g.n_real
            typ = torch.arange(nc) % 6
            typ = torch.where(typ < 2, 0, torch.where(typ == 2, 1, 2))
            r = g10.ew[E:E + nc] / g.ew[E:E + nc]
            assert torch.allclose(r, 3 * torch.tensor([.45, .45, .1])[typ]) and torch.equal(g10.ew[:E], g.ew[:E]) and torch.equal(g10.ew[E + nc:], g.ew[E + nc:])
        else:
            g10.load_state_dict(g.state_dict(), strict=False)  # ew is a buffer: load only here
            seen = []
            h = g10.enc.register_forward_hook(lambda mod, i, o: seen.append(o))
            e = g10.embed(b)
            h.remove()
            assert torch.allclose(e, (seen[0][:6].view(2, 3, -1) * torch.tensor([.45, .45, .1])[:, None]).sum(1), atol=1e-6)
            assert not torch.allclose(e, g.embed(b))
    gb, gg = (build(m, 5, 8, 2, 2, 0.0, torch.randn(4, 8, 4), 2, 3).eval() for m in ("comp_hier_bal", "comp_hier_gate"))
    gg.load_state_dict(gb.state_dict())  # same parameters and buffers
    b["gate"] = torch.full((2, 4, 3), 1 / 3)  # uniform gates = comp_hier_bal's 1 / n_experts
    assert torch.allclose(gg.embed(b), gb.embed(b), atol=1e-6)
    b["gate"] = F.one_hot(torch.randint(3, (2, 4)), 3).float()  # top-1 routing: unchosen experts send nothing
    b2 = dict(b, exp=b["exp"] + 10 * (1 - b["gate"]).repeat_interleave(32, -1))
    assert torch.allclose(gg.embed(b), gg.embed(b2), atol=1e-5) and not torch.allclose(gb.embed(b), gb.embed(b2))
    print("hier ok")
