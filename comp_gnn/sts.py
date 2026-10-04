"""ILSE STS protocol (paper Sec. 5 / their run_optuna_trial_sts_*): train a siamese encoder on STSBenchmark train with
MSE on min + (cos+1)/2 * (max-min), select on STSBenchmark validation Spearman, report STSBenchmark test and zero-shot
transfer to STS12-16, BIOSSES, SICK-R (test splits; metric = Spearman(cos, gold), as MTEB cos_sim).
Run from the ILSE-main root:  python -m comp_gnn.sts --model_family F --model_size S --out res.json [--methods ...]
Features (layerwise residual r, head z, MLP outputs) are encoded in-process with components.run, kept on CPU in fp16.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from experiments.utils.model_definitions.gnn.gnn_datasets import load_task_data
from comp_gnn.components import run as encode, geom, weights, is_moe
from comp_gnn.run import build, DEV
from comp_gnn import wb

TRANSFER = ["STS12", "STS13", "STS14", "STS15", "STS16", "BIOSSES", "SICK-R"]
PAPER = {  # ILSE STS table (Spearman x100), Cayley row: STSB, STS12-16, BIOSSES, SICK-R
    "Pythia": (55.84, 56.0, 60.2, 56.65, 66.99, 58.63, 56.59, 55.34)}


def featurize(wrapper, task, split, bs):
    d = load_task_data(task, split)
    out = {"score": torch.tensor(np.asarray(d["original_scores"], np.float32))}
    for side, texts in (("a", d["text_a"]), ("b", d["text_b"])):
        f, lw = encode(wrapper, [str(t) for t in texts], bs)  # z, mlp (+ gate, exp for MoE)
        out[side] = {"r": torch.from_numpy(lw.astype(np.float16))} | {k: torch.from_numpy(v) for k, v in f.items()}
    print(f"[sts] {task}/{split}: n={len(out['score'])} r={tuple(out['a']['r'].shape)}", flush=True)
    return out


def side(s, idx):
    return {k: v[idx].to(DEV, non_blocking=True).float() for k, v in s.items()}


@torch.no_grad()
def cosines(model, p, bs=256):
    model.eval()
    n = len(p["score"])
    return torch.cat([nn.functional.cosine_similarity(model.embed(side(p["a"], slice(i, i + bs))),
                                                      model.embed(side(p["b"], slice(i, i + bs))))
                      for i in range(0, n, bs)]).cpu().numpy()


def rho(c, p):
    return float(spearmanr(c, p["score"].numpy()).correlation)


def train(method, d, nl, dropout, lr, seed, epochs, wd=1e-4, bs=64):
    torch.manual_seed(seed)
    tr, va = d["STSBenchmark/train"], d["STSBenchmark/validation"]
    _, L, D = tr["a"]["r"].shape
    model = build(method, L, D, 1, nl, dropout, d["W_O"], d["nheads"], d["nexp"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)  # theirs: on val loss
    best, best_state, g = -2.0, None, torch.Generator().manual_seed(seed)
    curve = {"train_loss": [], "val_rho": []}
    for _ in range(epochs):
        model.train()
        tot = 0.0
        perm = torch.randperm(len(tr["score"]), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            cos = nn.functional.cosine_similarity(model.embed(side(tr["a"], idx)), model.embed(side(tr["b"], idx)))
            loss = nn.functional.mse_loss(5 * (cos + 1) / 2, tr["score"][idx].to(DEV))  # STSB range [0, 5]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        c = cosines(model, va)
        curve["train_loss"].append(tot / len(perm)), curve["val_rho"].append(rho(c, va))
        wb.log(len(curve["val_rho"]), **{f"s{seed}/{k}": v[-1] for k, v in curve.items()})
        sched.step(float(((5 * (c + 1) / 2 - va["score"].numpy()) ** 2).mean()))
        if rho(c, va) > best:
            best, best_state = rho(c, va), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best, curve


def test_all(model, d):
    return {t: rho(cosines(model, d[f"{t}/test"]), d[f"{t}/test"]) for t in ["STSBenchmark"] + TRANSFER}


def layer_rho(p, l):
    return rho(nn.functional.cosine_similarity(p["a"]["r"][:, l].float(), p["b"]["r"][:, l].float()).numpy(), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_size", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="gin_cayley,gin_fc,deepset,mlp_last,weighted,dwatt,comp_gnn,comp_noedge,comp_resonly,comp_cayley")
    ap.add_argument("--epochs", type=int, default=25)  # their STS optuna setting
    ap.add_argument("--extra_seeds", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=128)
    a = ap.parse_args()
    wrapper = TextLayerwiseAutoModelWrapper(TextModelSpecifications(a.model_family, a.model_size, "main", ignore_checks=True),
                                            device_map="auto", evaluation_layer_idx=-1, use_memory_efficient_hooks=True)
    W, _ = weights(wrapper.model)
    d = {"W_O": W.to(DEV), "nheads": geom(wrapper.model)[0],
         "nexp": wrapper.model.config.num_experts if is_moe(wrapper.model) else 0}
    for t, sp in [("STSBenchmark", s) for s in ("train", "validation", "test")] + [(t, "test") for t in TRANSFER]:
        d[f"{t}/{sp}"] = featurize(wrapper, t, sp, a.batch_size)
    del wrapper
    torch.cuda.empty_cache()
    tasks = ["STSBenchmark"] + TRANSFER
    L = d["STSBenchmark/train"]["a"]["r"].shape[1]
    # Last Layer and Best Single Layer (raw cosine; layer picked on STSB validation, oracle per task reported too)
    per = {t: [layer_rho(d[f"{t}/test"], l) for l in range(L)] for t in tasks}
    val = [layer_rho(d["STSBenchmark/validation"], l) for l in range(L)]
    bl = int(np.nanargmax(val))
    res = {"model": f"{a.model_family}_{a.model_size}", "L": L, "paper_cayley": PAPER.get(a.model_family),
           "n": {k: len(v["score"]) for k, v in d.items() if isinstance(v, dict)},
           "lastlayer": {t: per[t][-1] for t in tasks},
           "bestlayer": {"layer": bl, "test": {t: per[t][bl] for t in tasks}, "oracle": {t: np.nanmax(per[t]) for t in tasks},
                         "val_per_layer": val, "test_per_layer": per},
           "methods": {m: r for m, r in (json.load(open(a.out))["methods"] if os.path.exists(a.out) else {}).items() if "fixed_cfg" in r}}  # resume; grid-era runs redone
    print("[sts] baselines " + json.dumps({k: res[k] for k in ("lastlayer",)} | {"bestlayer": res["bestlayer"]["test"]}), flush=True)
    for m in [m for m in a.methods.split(",") if m not in res["methods"]]:
        # no STS rows in ILSE's published Optuna dump: their basic_gin_trainer defaults (lr 1e-3, wd 1e-4, dropout 0.1);
        # ponytail: its 3 GIN layers is outside the paper's {1, 2}, so nl=2. Same config for every method, no tuning
        wb.init("sts", res["model"], "STS", m, vars(a) | {"method": m})
        t0, cfg = time.time(), {"lr": 1e-3, "dropout": 0.1, "nl": 1 if m in ("weighted", "dwatt") else 2}
        best_model, v, curve = train(m, d, cfg["nl"], cfg["dropout"], cfg["lr"], 0, a.epochs)
        cfg |= {"val": v, "curve": curve}
        r = {"fixed_cfg": cfg, "test": test_all(best_model, d), "params": sum(p.numel() for p in best_model.parameters())}
        runs = [train(m, d, cfg["nl"], cfg["dropout"], cfg["lr"], s, a.epochs) for s in range(1, 1 + a.extra_seeds)]
        r["test_seeds"] = [r["test"]] + [test_all(x[0], d) for x in runs]
        r["seed_curves"] = [cfg["curve"]] + [x[2] for x in runs]
        r["minutes"] = (time.time() - t0) / 60
        res["methods"][m] = r
        print(m, json.dumps({x: r[x] for x in ("fixed_cfg", "test", "minutes")}), flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        wb.finish({x: r[x] for x in r if x != "seed_curves"} | {"fixed_cfg": {k: v for k, v in cfg.items() if k != "curve"}})
    print("[sts] done", flush=True)


if __name__ == "__main__":
    p = {"a": {"r": torch.randn(50, 3, 4)}, "score": torch.rand(50)}
    p["b"] = {"r": torch.randn(50, 3, 4)}
    p["score"] = nn.functional.cosine_similarity(p["a"]["r"][:, 1], p["b"]["r"][:, 1])
    assert abs(layer_rho(p, 1) - 1) < 1e-6 and abs(layer_rho(p, 0)) < 0.6  # gold = cos at layer 1
    main()
