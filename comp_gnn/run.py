"""ILSE reproduction on precomputed Pythia-410M h5 files + a component-level (residual / attention-head / MLP) GNN.

Run from the ILSE-main root:  python -m comp_gnn.run --emb_dir <out>/Pythia_410m_mean_pooling --task X --out res.json
ILSE's per-task selected config (FIXED) and trainer for every method (Adam, CE, ReduceLROnPlateau(3, 0.5), 50 epochs, best-val ckpt).
"""
import argparse
import copy
import glob
import itertools
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from experiments.utils.precompute.h5_utils import load_embeddings_from_h5
from experiments.utils.model_definitions.gnn.gnn_datasets import SingleGraphDataset
from experiments.utils.model_definitions.gnn.gnn_models import get_cayley_graph, LayerGINEncoder, MLPEncoder, DeepSetEncoder, LearnedWeightingEncoder, DWAttEncoder
from experiments.utils.model_definitions.gnn.evaluate_linear_gcn import undersample_data

DEV = "cuda"
SAMPLES_PER_LABEL = {"EmotionClassification": 16}  # MTEB 1.38 task metadata; 8 otherwise
PAPER = {  # ILSE paper table rows per LLM: Cayley, FC, LastLayer, DeepSet, MLP-last
    "Pythia": {"Banking77Classification": (89.43, 90.65, 61.17, 84.23, 83.84), "EmotionClassification": (73.83, 75.61, 33.48, 47.89, 33.99),
               "MTOPDomainClassification": (98.77, 98.68, 80.88, 97.59, 96.97), "MTOPIntentClassification": (94.72, 95.04, 66.97, 92.21, 83.75),
               "PoemSentimentClassification": (70.87, 75.77, 42.40, 73.37, 75.00)},
    "Gemma2": {"Banking77Classification": (92.58, 92.39, 62.16, 90.03, 87.47), "EmotionClassification": (69.60, 79.90, 26.89, 78.77, 59.05),
               "MTOPDomainClassification": (99.16, 99.07, 80.79, 98.92, 98.37), "MTOPIntentClassification": (96.43, 95.34, 68.02, 94.37, 92.73),
               "PoemSentimentClassification": (83.27, 77.12, 35.67, 78.56, 71.63)},
    "Llama3": {"Banking77Classification": (92.85, 92.38, 68.25, 87.62, 86.70), "EmotionClassification": (73.43, 71.64, 34.23, 71.04, 67.67),
               "MTOPDomainClassification": (99.03, 98.99, 84.42, 98.77, 98.58), "MTOPIntentClassification": (96.46, 96.19, 73.39, 95.43, 92.09),
               "PoemSentimentClassification": (79.04, 77.98, 40.96, 77.02, 75.00)}}


def load(tdir):
    d, raw = {}, {sp: load_embeddings_from_h5(os.path.join(tdir, f"{sp}.h5")) for sp in ("train", "validation", "test")}
    classes = np.unique(np.concatenate([np.asarray(y) for _, y in raw.values()]))
    for split, (emb, y) in raw.items():
        s = {"r": torch.tensor(np.asarray(emb), dtype=torch.float32, device=DEV),
             "y": torch.tensor(np.searchsorted(classes, np.asarray(y)), device=DEV).long()}
        zp = os.path.join(tdir, f"{split}_z.npy")
        if os.path.exists(zp):
            s["z"] = torch.tensor(np.load(zp), device=DEV)
            s["mlp"] = torch.tensor(np.load(os.path.join(tdir, f"{split}_mlp.npy")), device=DEV)
            for k in ("gate", "exp"):  # MoE only
                if os.path.exists(os.path.join(tdir, f"{split}_{k}.npy")):
                    s[k] = torch.tensor(np.load(os.path.join(tdir, f"{split}_{k}.npy")), device=DEV)
        d[split] = s
    wp = os.path.join(tdir, "W_O.npy")
    d["W_O"] = torch.tensor(np.load(wp), device=DEV).float() if os.path.exists(wp) else None
    mp = os.path.join(tdir, "meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {"nheads": 16}  # Pythia-410M runs predate meta.json
    d["nheads"], d["nexp"] = meta["nheads"], meta.get("nexperts", 0)
    return d


# ---------------- ILSE GIN with fast manual batching (identical math to PyG DataLoader batching) ----------------
class GIN(nn.Module):
    def __init__(self, L, D, C, nl, dropout, graph_type, legacy=False):
        super().__init__()
        cayley = graph_type == "cayley"
        ds = SingleGraphDataset([np.zeros((L, 1), np.float32)], [0], graph_type, keep_embedding_layer=not legacy)
        self.drop_first = ds.items[0].shape[0] < L  # legacy cayley drops the embedding layer
        self.n_real = ds.items[0].shape[0]
        self.n_nodes = ds.cayley_num_nodes or self.n_real
        self.register_buffer("ei", ds.edge_index.clone())
        self.pool_real = cayley and not legacy
        # real-node-only pooling (official cayley setting) is done in embed() by routing virtual nodes to a dummy graph
        self.enc = LayerGINEncoder(D, 256, nl, dropout, 1, "mean", graph_type, pool_real_nodes_only=False, train_eps=cayley)
        self.out_dim = 256
        self.head = nn.Linear(256, C)

    def nodes(self, b):
        return b["r"][:, 1:] if self.drop_first else b["r"]

    def embed(self, b):
        x = self.nodes(b)
        B, L, D = x.shape
        if self.n_nodes > L:
            x = torch.cat([x, x.new_zeros(B, self.n_nodes - L, D)], 1)
        off = torch.arange(B, device=x.device) * self.n_nodes
        gid = torch.arange(B, device=x.device)[:, None].expand(B, self.n_nodes).clone()
        tid, T = getattr(self, "tid", None), 1  # tid (MoE components): mean of per-type means (heads, MLP, experts)
        if tid is not None:
            T = 3
            gid[:, :self.n_real] = gid[:, :self.n_real] * T + tid
        if self.pool_real:
            gid[:, self.n_real:] = T * B
        g = SimpleNamespace(x=x.reshape(-1, D), edge_index=(self.ei[:, None, :] + off[None, :, None]).reshape(2, -1),
                            batch=gid.reshape(-1), num_graphs=T * B)
        o = self.enc(g)[:T * B].view(B, T, -1)
        return o.mean(1) if getattr(self, "tw", None) is None else (o * self.tw[:, None]).sum(1)  # tw: hier.py _e10


def jl(nblk, D, K):
    """components.py's per-layer JL matrices (seed = layer), (nblk, D, K)."""
    return torch.stack([torch.randn(D, K, generator=torch.Generator().manual_seed(l)) / K ** 0.5 for l in range(nblk)])


class CompGIN(GIN):
    """ILSE's Cayley encoder unchanged (LayerGINEncoder, train_eps, real-node-only mean pooling, SL(2,Z_k) graph) with the
    layer nodes swapped for per-block head + MLP (+ expert) output nodes, block-major like ILSE's layer order."""
    def __init__(self, L, D, C, nl, dropout, W_O, nheads, nexp=0, kexp=32):
        nn.Module.__init__(self)
        nblk = W_O.shape[0]
        self.register_buffer("W", W_O.reshape(nblk, D, nheads, W_O.shape[2] // nheads))  # o_proj input H*hd can differ from D
        self.nheads, self.nexp, self.kexp, self.drop_first, self.pool_real = nheads, nexp, kexp, False, True
        self.n_real = nblk * (nheads + 1 + nexp)
        if nexp:
            self.register_buffer("P", jl(nblk, D, kexp))
            self.register_buffer("tid", torch.tensor([0] * nheads + [1] + [2] * nexp).repeat(nblk))
        # ILSE's get_cayley_graph caps at 1000 nodes; cayley_sl2 is the same graph uncapped (Llama3 needs 1056 nodes)
        ei = next(e for e in map(cayley_sl2, itertools.count(2)) if e.max() + 1 >= self.n_real)
        self.n_nodes = int(ei.max()) + 1
        self.register_buffer("ei", ei)
        self.enc = LayerGINEncoder(D, 256, nl, dropout, 1, "mean", "cayley", pool_real_nodes_only=False, train_eps=True)
        self.out_dim = 256
        self.head = nn.Linear(256, C)

    def nodes(self, b):
        B, nblk, _ = b["z"].shape
        z = b["z"].float().reshape(B, nblk, self.nheads, -1)
        parts = [torch.einsum("blhk,ldhk->blhd", z, self.W), b["mlp"].float()[:, :, None]]
        if self.nexp:  # expert node = P_l s: the JL sketch s = P_l^T (g_e E_e(x)) mapped back to D (unbiased, E[P P^T] = I)
            parts.append(torch.einsum("blek,ldk->bled", b["exp"].float().reshape(B, nblk, self.nexp, self.kexp), self.P))
        return torch.cat(parts, 2).reshape(B, self.n_real, -1)


class MLPLast(nn.Module):
    def __init__(self, L, D, C, nl, dropout):
        super().__init__()
        self.enc = MLPEncoder(D, 256, nl, dropout)
        self.out_dim = self.enc.out_dim
        self.head = nn.Linear(self.out_dim, C)

    def embed(self, b):
        return self.enc(b["r"][:, -1])


class DeepSet(nn.Module):
    def __init__(self, L, D, C, nl, dropout):
        super().__init__()
        self.enc = DeepSetEncoder(L, D, 256, pre_pooling_layers=0, post_pooling_layers=nl, dropout=dropout)
        self.out_dim = 256
        self.head = nn.Linear(self.out_dim, C)

    def embed(self, b):
        return self.enc(b["r"])


class Wrap(nn.Module):
    """ILSE's Weighted (ELMo softmax over layers) and DWAtt (paper-faithful, hidden_dim=None) encoders + linear head."""
    def __init__(self, enc, C):
        super().__init__()
        self.enc, self.out_dim = enc, enc.out_dim if hasattr(enc, "out_dim") else enc.working_dim
        self.head = nn.Linear(self.out_dim, C)

    def embed(self, b):
        return self.enc(b["r"])


def cayley_sl2(k):
    """ILSE's get_cayley_graph(k) (same generators, BFS numbering, both-direction edges) minus its k<=10 / 1000-node caps;
    Llama3's 1089 component nodes need k=11 (1320 nodes)."""
    gens = [((1, 1), (0, 1)), ((1, k - 1), (0, 1)), ((1, 0), (1, 1)), ((1, 0), (k - 1, 1))]
    mul = lambda a, b: tuple(tuple((a[i][0] * b[0][j] + a[i][1] * b[1][j]) % k for j in range(2)) for i in range(2))
    order = [((1, 0), (0, 1))]
    idx = {order[0]: 0}
    for a in order:  # BFS: the list grows while iterating
        for g in gens:
            if mul(a, g) not in idx:
                idx[mul(a, g)] = len(order)
                order.append(mul(a, g))
    e = [(i, idx[mul(a, g)]) for i, a in enumerate(order) for g in gens]
    return torch.tensor(e + [(j, i) for i, j in e]).T


# ILSE's own selected config per task: best_val_acc row of gin + cayley in their published Optuna dump
# (ILSE-main/scripts_and_jobs/scripts/eval/eval_pipeline/combined_table.csv, 1024-dim model = Pythia-410m).
# Same config for every method on that task (theirs and ours), no tuning on our side.
FIXED = {"Banking77Classification": {"lr": 1e-3, "wd": 1e-4, "dropout": 0.1, "nl": 1},
         "EmotionClassification": {"lr": 1e-3, "wd": 1e-4, "dropout": 0.1, "nl": 2},
         "MTOPDomainClassification": {"lr": 1e-3, "wd": 1e-4, "dropout": 0.3, "nl": 2},
         "MTOPIntentClassification": {"lr": 1e-3, "wd": 1e-4, "dropout": 0.2, "nl": 1},
         "PoemSentimentClassification": {"lr": 1e-3, "wd": 1e-3, "dropout": 0.2, "nl": 2}}


def build(method, L, D, C, nl, dropout, W_O, nheads=16, nexp=0):
    if method.endswith("_flat"):  # MoE ablation: same expert P_l s nodes, plain mean pool (no per-type balancing)
        g = build(method.removesuffix("_flat"), L, D, C, nl, dropout, W_O, nheads, nexp)
        g.tid = None
        return g
    if method == "gin_cayley":
        return GIN(L, D, C, nl, dropout, "cayley")
    if method == "gin_fc":
        return GIN(L, D, C, nl, dropout, "fully_connected")
    if method == "mlp_last":
        return MLPLast(L, D, C, nl, dropout)
    if method == "deepset":
        return DeepSet(L, D, C, nl, dropout)
    if method == "weighted":
        return Wrap(LearnedWeightingEncoder(L, D), C)
    if method == "dwatt":
        return Wrap(DWAttEncoder(L, D, None, 0.5, 24, dropout), C)
    if method in ("comp_gnn", "comp_noexp", "comp_noedge", "comp_cayley", "comp_resonly"):  # CompGNN (compgnn.py)
        return __import__("comp_gnn.compgnn").compgnn.build(method, L, D, C, nl, dropout, W_O, nheads, nexp)
    if method == "comp_cayley_nores":
        return CompGIN(L, D, C, nl, dropout, W_O, nheads, nexp)
    if method.startswith("comp_hier_") or method.removesuffix("_e10") in ("comp_cayley_ln", "comp_similarity", "comp_noedge_ln", "comp_resonly_ln"):  # live in hier.py
        return __import__("comp_gnn.hier").hier.build(method, L, D, C, nl, dropout, W_O, nheads, nexp)
    raise ValueError(method)


def batch(s, idx):
    return {k: v[idx] for k, v in s.items()}


@torch.no_grad()
def embed_all(model, s, bs=64):  # 512 OOMs the weighted-sum Hier rows on Qwen
    model.eval()
    return torch.cat([model.embed(batch(s, torch.arange(i, min(i + bs, len(s["y"])), device=DEV)))
                      for i in range(0, len(s["y"]), bs)])


def accuracy(model, s):
    return (model.head(embed_all(model, s)).argmax(-1) == s["y"]).float().mean().item()


def train(method, d, nl, dropout, lr, seed, epochs, wd=1e-4, bs=64):
    torch.manual_seed(seed)
    tr = d["train"]
    _, L, D = tr["r"].shape
    C = int(max(d[k]["y"].max().item() for k in ("train", "validation", "test")) + 1)
    model = build(method, L, D, C, nl, dropout, d["W_O"], d["nheads"], d["nexp"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=3, factor=0.5)
    best, best_state, g = -1.0, None, torch.Generator(device=DEV).manual_seed(seed)
    curve = {"train_loss": [], "val_acc": []}
    for _ in range(epochs):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(tr["y"]), device=DEV, generator=g)
        for i in range(0, len(perm), bs):
            b = batch(tr, perm[i:i + bs])
            loss = nn.functional.cross_entropy(model.head(model.embed(b)), b["y"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b["y"])
        va = accuracy(model, d["validation"])
        curve["train_loss"].append(tot / len(perm)), curve["val_acc"].append(va)
        sched.step(va)
        if va > best:
            best, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best, curve


def kshot(Xtr, ytr, Xte, yte, k):
    """MTEB 1.38 classification protocol (10 undersampled experiments, LR max_iter=100), via ILSE's own helper."""
    accs, idxs = [], None
    for _ in range(10):
        xs, ys, idxs = undersample_data(Xtr, ytr, samples_per_label=k, seed=42, idxs=idxs)
        clf = LogisticRegression(max_iter=100, random_state=42).fit(xs, ys)
        accs.append(float((clf.predict(Xte) == yte).mean()))
    return float(np.mean(accs))


def legacy_eval(model, d, k):
    """Official GNNWrapper bug: same trained weights, but the graph is rebuilt without keep_embedding_layer
    (embedding layer dropped, 24-node Cayley) and pooling runs over all nodes."""
    _, L, D = d["train"]["r"].shape
    C = model.head.out_features
    leg = GIN(L, D, C, len(model.enc.gnn_layers), 0.0, "cayley", legacy=True).to(DEV)
    leg.load_state_dict({k: v for k, v in model.state_dict().items() if k != "ei"}, strict=False)
    assert not leg.pool_real
    if not leg.drop_first:  # the bug drops the embedding layer only when L exceeds the legacy graph (Qwen3.6, L=41): n/a
        return None
    tr, te = d["train"], d["test"]
    return {"head_test": accuracy(leg, te),
            "kshot_test": kshot(embed_all(leg, tr).cpu().numpy(), tr["y"].cpu().numpy(),
                                embed_all(leg, te).cpu().numpy(), te["y"].cpu().numpy(), k)}


def wandb_name(out):
    """<family>/<TASK>__g<G>, i.e. the result json path relative to the results root."""
    return f"{os.path.basename(os.path.dirname(os.path.abspath(out)))}/{os.path.splitext(os.path.basename(out))[0]}"


def wandb_init(a):
    """Optional W&B mirror of the result json. No-op without WANDB_API_KEY;
    wandb itself reads WANDB_BASE_URL / WANDB_ENTITY / WANDB_PROJECT from the job env."""
    if not os.environ.get("WANDB_API_KEY"):
        return None
    name = wandb_name(a.out)
    try:
        import wandb
        run = wandb.init(name=name, id=re.sub(r"[^\w-]", "-", name), resume="allow",  # retries resume the run
                         config={**vars(a), "fixed": FIXED.get(a.task)})
    except Exception as e:  # the json is the source of truth: never fail a job over logging
        print(f"wandb disabled: {type(e).__name__}: {e}", flush=True)
        return None
    run.define_metric("curve/*", step_metric="epoch")
    print("wandb run:", run.url, flush=True)
    return run


def wandb_summary(run, res):
    """Idempotent: mirrors res headline numbers (incl. methods resumed from an earlier json) into the summary."""
    s = {"lastlayer_kshot_test": res["lastlayer_kshot_test"],
         **{f"bestlayer/{x}": res["bestlayer"][x] for x in ("layer", "kshot_test", "oracle_kshot_test")}}
    for m, r in res["methods"].items():
        s.update({f"{m}/{x}": r[x] for x in ("head_test", "kshot_test", "params", "minutes") if x in r})
        if "head_test_seeds" in r:
            s[f"{m}/head_test_mean"] = float(np.mean(r["head_test_seeds"]))
    run.summary.update(s)


def merge(out, parts):
    """Per-seed jsons (--seed 0, 1, ... run in parallel) -> the json a sequential run writes; minutes = slowest seed."""
    rs = [json.load(open(f)) for f in parts]
    for m, r in rs[0]["methods"].items():
        for o in rs[1:]:
            r["head_test_seeds"] += o["methods"][m]["head_test_seeds"]
            r["seed_curves"] += o["methods"][m]["seed_curves"]
            r["minutes"] = max(r["minutes"], o["methods"][m]["minutes"])
        r["parallel_seeds"] = len(parts)
    json.dump(rs[0], open(out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="gin_cayley,gin_fc,deepset,mlp_last,comp_gnn,comp_noedge,comp_resonly,comp_cayley")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--extra_seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=-1)  # train this one seed only (parallel seeds, joined by merge); -1: seed 0 + extra_seeds
    a = ap.parse_args()
    seeds = [a.seed] if a.seed >= 0 else list(range(1 + a.extra_seeds))
    d = load(os.path.join(a.emb_dir, a.task))
    k = SAMPLES_PER_LABEL.get(a.task, 8)
    tr, te = d["train"], d["test"]
    ytr, yte = tr["y"].cpu().numpy(), te["y"].cpu().numpy()
    last_tr, last_te = tr["r"][:, -1].cpu().numpy(), te["r"][:, -1].cpu().numpy()
    res = {"task": a.task, "L": int(tr["r"].shape[1]), "n": {s: len(d[s]["y"]) for s in ("train", "validation", "test")},
           "samples_per_label": k, "paper_cayley_fc_last_deepset_mlplast": PAPER.get(os.path.basename(os.path.normpath(a.emb_dir)).split("_")[0], {}).get(a.task),
           "methods": {}}
    # ponytail: resume from a previous partial out json (full-train LR dropped: CPU-bound for hours, not the paper protocol)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res.update({x: old[x] for x in ("lastlayer_kshot_test", "bestlayer", "methods", "grid_methods") if x in old})
    for m in [m for m, r in res["methods"].items() if "fixed_cfg" not in r]:  # earlier grid-selected runs: kept, not reported
        res.setdefault("grid_methods", {})[m] = res["methods"].pop(m)
    for f in sorted(glob.glob(os.path.join(os.path.dirname(a.out), f"{a.task}__*.json"))):  # bestlayer is per (model, task)
        sib = json.load(open(f)) if "bestlayer" not in res else {}
        res.update({x: sib[x] for x in ("lastlayer_kshot_test", "bestlayer") if "bestlayer" in sib})
    # Best Single Layer: MTEB k-shot LR per layer; picked on validation, oracle (best on test) reported too
    if "bestlayer" not in res:
        res["lastlayer_kshot_test"] = kshot(last_tr, ytr, last_te, yte, k)
        va, yva = d["validation"]["r"].cpu().numpy(), d["validation"]["y"].cpu().numpy()
        per = [(kshot(tr["r"][:, l].cpu().numpy(), ytr, va[:, l], yva, k), kshot(tr["r"][:, l].cpu().numpy(), ytr, te["r"][:, l].cpu().numpy(), yte, k))
               for l in range(tr["r"].shape[1])]
        bl = max(range(len(per)), key=lambda l: per[l][0])
        res["bestlayer"] = {"layer": bl, "kshot_test": per[bl][1], "oracle_kshot_test": max(p[1] for p in per), "per_layer_val_test": per}
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    run = wandb_init(a)
    if run:
        wandb_summary(run, res)
    for m in a.methods.split(","):
        if m in res["methods"] or (m.startswith("comp") and d["W_O"] is None):
            continue
        t0 = time.time()
        fx = FIXED[a.task]
        nl = 1 if m in ("weighted", "dwatt") else fx["nl"]
        best_model, va, curve = train(m, d, nl, fx["dropout"], fx["lr"], seeds[0], a.epochs, fx["wd"])
        cfg = {**fx, "nl": nl, "val": va, "test_head": accuracy(best_model, te), "curve": curve}
        Etr, Ete = embed_all(best_model, tr).cpu().numpy(), embed_all(best_model, te).cpu().numpy()
        r = {"fixed_cfg": cfg, "head_test": cfg["test_head"], "kshot_test": kshot(Etr, ytr, Ete, yte, k),
             "params": sum(p.numel() for p in best_model.parameters())}
        if m == "gin_cayley":
            r["official_eval_bug"] = legacy_eval(best_model, d, k)
        runs = [train(m, d, cfg["nl"], cfg["dropout"], cfg["lr"], s, a.epochs, cfg["wd"]) for s in seeds[1:]]
        seeds = [cfg["test_head"]] + [accuracy(x[0], te) for x in runs]
        r["head_test_seeds"], r["minutes"] = seeds, (time.time() - t0) / 60
        r["seed_curves"] = [cfg["curve"]] + [x[2] for x in runs]
        res["methods"][m] = r
        print(m, json.dumps({x: r[x] for x in r if x != "seed_curves"}), flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        if run:
            for e, (tl, v) in enumerate(zip(curve["train_loss"], curve["val_acc"], strict=True)):
                run.log({"epoch": e, f"curve/{m}/train_loss": tl, f"curve/{m}/val_acc": v})
            wandb_summary(run, res)
    if run:
        import wandb
        art = wandb.Artifact(re.sub(r"[^\w.-]", "-", run.name), type="result")
        art.add_file(a.out)
        run.log_artifact(art)
        run.finish()


if __name__ == "__main__":
    dense = lambda e: torch.zeros(648, 648).index_put_((torch.as_tensor(e[1]), torch.as_tensor(e[0])), torch.ones(e.shape[1]), accumulate=True)
    assert torch.equal(dense(cayley_sl2(9)), dense(get_cayley_graph(9)))  # matches ILSE's graph where theirs is defined
    g = CompGIN(25, 8, 2, 1, 0.0, torch.zeros(24, 8, 16 * 2), 16)  # 408 component nodes -> SL(2,Z_9), ILSE's encoder
    assert g.n_real == 408 and g.n_nodes == 648 and torch.equal(dense(g.ei), dense(get_cayley_graph(9)))
    assert g.embed({"r": torch.randn(3, 25, 8), "z": torch.randn(3, 24, 32), "mlp": torch.randn(3, 24, 8)}).shape == (3, 256)
    g = CompGIN(3, 8, 2, 1, 0.0, torch.zeros(2, 8, 4 * 2), 4, nexp=5, kexp=3)  # experts: P_l s, pooled per type
    v = torch.randn(1, 2, 5, 8)  # g_e E_e(x); components.py sketches it as v @ P_l, P_l of seed l
    s = torch.stack([v[:, l] @ torch.randn(8, 3, generator=torch.Generator().manual_seed(l)) / 3 ** 0.5 for l in range(2)], 1)
    b = {"r": torch.randn(1, 3, 8), "z": torch.randn(1, 2, 8), "mlp": torch.randn(1, 2, 8), "exp": s.reshape(1, 2, 15)}
    assert torch.allclose(g.nodes(b)[0, 5:10], s[0, 0] @ g.P[0].T, atol=1e-5)
    P = jl(1, 8, 4000)[0]
    assert (P @ P.T - torch.eye(8)).abs().max() < 0.1  # E[P P^T] = I: back-projection unbiased
    seen = []
    h = g.eval().enc.register_forward_hook(lambda m, i, o: seen.append((i[0].batch, o)))
    e = g.embed(b)
    h.remove()
    assert torch.equal(seen[0][0][:20], g.tid) and (seen[0][0][20:] == 3).all()  # heads / MLP / experts -> graphs 0 / 1 / 2
    assert torch.allclose(e, seen[0][1][:3].mean(0, keepdim=True), atol=1e-6)
    f = build("comp_cayley_nores_flat", 3, 8, 2, 1, 0.0, torch.zeros(2, 8, 4 * 2), 4, 5)  # _flat: same params, one pooled graph
    assert f.tid is None and sum(p.numel() for p in f.parameters()) == sum(p.numel() for p in g.parameters())
    g.tid, seen = None, []
    h = g.enc.register_forward_hook(lambda m, i, o: seen.append(i[0].batch))
    g.embed(b)
    h.remove()
    assert (seen[0][:20] == 0).all() and (seen[0][20:] == 1).all()
    assert wandb_name("results/Gemma2/Banking77Classification__g3.json") == "Gemma2/Banking77Classification__g3"
    tmp = __import__("tempfile").mkdtemp()
    for s, acc in enumerate((0.5, 0.6, 0.7)):
        json.dump({"bestlayer": 3, "methods": {"m": {"head_test": acc, "head_test_seeds": [acc], "seed_curves": [[s]], "minutes": s}}}, open(f"{tmp}/{s}", "w"))
    merge(f"{tmp}/o", [f"{tmp}/{s}" for s in range(3)])
    r = json.load(open(f"{tmp}/o"))["methods"]["m"]
    assert r["head_test"] == 0.5 and r["head_test_seeds"] == [0.5, 0.6, 0.7] and r["seed_curves"] == [[0], [1], [2]] and r["minutes"] == 2
    main()
