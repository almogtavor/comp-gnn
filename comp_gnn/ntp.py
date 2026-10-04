"""Next-token prediction with inter-layer readouts vs LoRA, on a frozen LLM.

Readout methods (ILSE encoders + ours, from run.build) read each token's own features - layerwise residuals r (L, D),
pre-o_proj head outputs z and MLP outputs - which are causal, so no token sees its future. A zero-initialised Linear maps
the readout embedding to a correction of the final (post-norm) hidden state: logits = lm_head(h_L + up(embed)), so every
readout starts exactly at the frozen model's perplexity. LoRA (ILSE's target modules) fine-tunes the LLM on the same
tokens. Gated LoRA (gnn_lora; controls noedge_lora, mlp_lora): LoRA whose rank-r update is gated per token,
W x_t + B(s_t * A x_t), with s_t = 1 + head(enc(token t's frozen-pass features)) - zero-init head, so it starts as LoRA;
causal since token t's features only see its prefix. lora_pm: plain LoRA at the smallest rank whose counted trainable params are
>= gnn_lora's (the parameter-matched baseline). Same token budget, 2048 tokens per optimizer step, one epoch; lr picked from 3 options by a
<=1 min probe (20 s of training each, dev ppl), shared by all seeds.
Run from the ILSE-main root:  python -m comp_gnn.ntp --model_family F --model_size S --dataset wikitext --out r.json
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.utils.model_definitions.text_automodel_wrapper import get_model_path
from comp_gnn.components import lm_only, finish, geom, hook, parts, scalars, weights
from comp_gnn.run import build, DEV
from comp_gnn import wb

# name: (hf path, config, text field, train/dev/test splits; None = carve from "train" by document index mod 20)
DATA = {"wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text", ("train", "validation", "test")),
        "code": ("codeparrot/codeparrot-clean-valid", None, "content", None),
        "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "text", None)}  # calib.py only
SEQ, TOK_STEP = 512, 2048
GATE = {"gnn_lora": "comp_gnn", "noedge_lora": "comp_noedge", "mlp_lora": "mlp_last"}  # gated-LoRA method: gate encoder


def targets(a):  # ILSE's LoRA target modules
    t = ["query_key_value", "dense"] if a.model_family == "Pythia" else ["q_proj", "v_proj", "k_proj", "o_proj"]
    if a.model_family == "Qwen36MoE":  # q/k/v/o exist on the 10 full-attn layers only; DeltaNet's q/k/v and o analogues
        t = t + ["in_proj_qkv", "out_proj"]
    return lm_only(t) if a.model_family == "Gemma4" else t


def chunks(tok, name, split, n_tokens):
    path, cfg, field, splits = DATA[name]
    ds = load_dataset(path, cfg, split=splits[split] if splits else "train", streaming=True)
    want = {0: 2, 1: 1, 2: 0}[split]  # carved: doc i mod 20 == 0 test, 1 dev, else train
    ids, step = [], SEQ - 1
    for i, ex in enumerate(ds):
        if splits is None and min(i % 20, 2) != want:
            continue
        if ex[field].strip():
            ids += tok(ex[field], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(ids) >= n_tokens // SEQ * step:
            break
    x = torch.tensor(ids[:len(ids) // step * step]).view(-1, step)
    return torch.cat([x.new_full((len(x), 1), tok.bos_token_id or tok.eos_token_id), x], 1)  # BOS at every chunk start


class Feats:
    """Per-token r (B, T, L, D), z (B, T, nl, H*hd), mlp (B, T, nl, D) from one no-grad forward (components.run, unpooled)."""

    def __init__(self, model):
        self.model, self.cur, self.on, self.S = model, {}, False, scalars(model)
        # LoRA forwards must not accumulate (graph-holding) tensors here
        hook(model, lambda k, x: self.cur.setdefault(k, []).append(x) if self.on else None)

    @torch.no_grad()
    def __call__(self, ids, **kw):  # kw: gsm8k's KV-cached decode (mask, positions, cache)
        self.cur, self.on = {}, True
        hs = self.model(ids, output_hidden_states=True, **kw).hidden_states
        self.on = False
        return {"r": torch.stack(hs, 2).float()} | finish({k: torch.stack(v, 2) for k, v in self.cur.items()}, self.S)


def head(model, h):
    logits = model.get_output_embeddings()(h.to(model.dtype)).float()
    cap = getattr(getattr(model.config, "text_config", model.config), "final_logit_softcapping", None)
    return cap * torch.tanh(logits / cap) if cap else logits


class Readout(nn.Module):
    def __init__(self, method, L, D, nl, dropout, W_O, nheads):
        super().__init__()
        self.enc = build(method, L, D, 1, nl, dropout, W_O, nheads)
        self.up = nn.Linear(self.enc.out_dim, D)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, f):
        B, T = f["r"].shape[:2]
        e = self.enc.embed({k: v.reshape(B * T, *v.shape[2:]) for k, v in f.items()})
        return f["r"][:, :, -1] + self.up(e).view(B, T, -1)


class GatedLoRA(nn.Module):
    """base(x) + alpha/r * B(A(drop(x)) * s), s (B, T, r) set by GateLoRA per forward; s None = base only."""

    def __init__(self, base, r, alpha=16, dropout=0.1):
        super().__init__()
        self.base, self.scale, self.s, self.drop = base, alpha / r, None, nn.Dropout(dropout)
        self.A = nn.Linear(base.in_features, r, bias=False, device=base.weight.device)
        self.B = nn.Linear(r, base.out_features, bias=False, device=base.weight.device)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))  # as peft
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        y = self.base(x)
        return y if self.s is None else y + (self.B(self.A(self.drop(x.float())) * self.s) * self.scale).to(y.dtype)


class GateLoRA(nn.Module):
    """Wraps the target Linears in GatedLoRA; one r-dim gate per (token, layer), shared by that layer's targets."""

    def __init__(self, model, enc_method, r, data, tm):
        super().__init__()
        layers = parts(model)[0]
        self.nl, self.r, self.mods, self.layer_of, self.undo = len(layers), r, nn.ModuleList(), [], []
        for i, l in enumerate(layers):
            for name, m in list(l.named_modules()):
                if name.split(".")[-1] in tm and isinstance(m, nn.Linear):
                    parent = l.get_submodule(name.rsplit(".", 1)[0]) if "." in name else l
                    g = GatedLoRA(m, r)
                    setattr(parent, name.split(".")[-1], g)
                    self.mods.append(g), self.layer_of.append(i), self.undo.append((parent, name.split(".")[-1], m))
        self.enc = build(enc_method, data["L"], data["D"], 1, 1, 0.1, data["W_O"], data["nheads"])
        self.head = nn.Linear(self.enc.out_dim, self.nl * r)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, model, feats, ids, fkw={}, mkw={}):  # fkw / mkw: the frozen / gated pass's KV-cache kwargs (gsm8k)
        f = feats(ids, **fkw)  # frozen pass: every gate is None here
        B, T = ids.shape
        s = 1 + self.head(self.enc.embed({k: v.reshape(B * T, *v.shape[2:]) for k, v in f.items()})).view(B, T, self.nl, self.r)
        for g, i in zip(self.mods, self.layer_of):
            g.s = s[:, :, i]
        try:
            return model(ids, **mkw).logits.float()
        finally:
            for g in self.mods:
                g.s = None

    def remove(self):
        for parent, name, m in self.undo:
            setattr(parent, name, m)


def pm_need(model, cfg, data, a):  # gnn_lora's total trainable params at this cfg (LoRA A/B + gate encoder + head), counted
    g = GateLoRA(model, GATE["gnn_lora"], cfg["r"], data, targets(a))
    g.remove()
    return sum(p.numel() for p in g.parameters() if p.requires_grad)


def logits(model, ro, feats, ids):  # ro=None means the LLM's own logits (frozen or LoRA)
    return ro(model, feats, ids) if isinstance(ro, GateLoRA) else head(model, ro(feats(ids))) if ro is not None else model(ids).logits.float()


def nll(model, ro, feats, ids, labels=None):
    """Summed next-token NLL and count over labels (default ids; -100 = ignored)."""
    lg, y = logits(model, ro, feats, ids), (ids if labels is None else labels)[:, 1:]
    return nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="sum"), int((y != -100).sum())


def batch(X, idx):
    """(ids, labels) on DEV: ntp X (N, T) token chunks, labels = ids; gsm8k X (N, 2, SEQ) ids + labels (right-padded, -100 on
    prompt and padding), trimmed to the batch's last labelled position (causal: right padding never reaches real tokens)."""
    x = X[idx]
    if x.dim() == 2:
        return x.to(DEV), None
    w = int((x[:, 1] != -100).any(0).nonzero().max()) + 1
    return x[:, 0, :w].to(DEV), x[:, 1, :w].to(DEV)


@torch.no_grad()
def ppl(model, ro, feats, X, bs):
    model.eval()
    if ro is not None:
        ro.eval()
    tot = n = 0
    for i in range(0, len(X), bs):
        s, c = nll(model, ro, feats, *batch(X, slice(i, i + bs)))
        tot, n = tot + s.item(), n + c
    return math.exp(tot / n)


def train(method, cfg, model, feats, data, seed, bs, a, probe=0, ckpt=None):  # ckpt: trained params, loaded if present, else saved
    t0 = time.time()
    torch.manual_seed(seed)
    if method in ("lora", "lora_pm"):
        from peft import LoraConfig, get_peft_model
        r, need = cfg["r"], pm_need(model, cfg, data, a) if method == "lora_pm" else 0
        while True:  # lora_pm: the smallest rank whose counted trainable params >= gnn_lora's, so the gated row is never larger
            pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=16, lora_dropout=0.1, target_modules=targets(a), bias="none"))
            if sum(p.numel() for p in pm.parameters() if p.requires_grad) >= need:
                break
            pm.unload()
            r += 1
        model, ro, params = pm, None, [p for p in pm.parameters() if p.requires_grad]
        for p in params:  # adapters train in fp32 on a bf16 base
            p.data = p.data.float()
    elif method in GATE:
        ro = GateLoRA(model, GATE[method], cfg["r"], data, targets(a)).to(DEV)
        params = [p for p in ro.parameters() if p.requires_grad]
        x0 = batch(data["dev"], slice(0, 1))[0]
        with torch.no_grad():  # B = 0 and gate = 1 at init: exactly the frozen model
            assert (ro.eval()(model, feats, x0) - model(x0).logits.float()).abs().max() < 1e-3, "gated LoRA does not start at the frozen model"
    else:
        ro = Readout(method, data["L"], data["D"], cfg["nl"], 0.1, data["W_O"], data["nheads"]).to(DEV)
        params = list(ro.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=1e-4 if isinstance(ro, Readout) else 0.0)
    acc, X = TOK_STEP // (bs * SEQ), data["train"]
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    curve = []  # mean train NLL/token per optimizer step
    pre = f"probe_lr{cfg['lr']}/" if probe else ""
    mod, ck = ro or model, torch.load(ckpt) if ckpt and os.path.exists(ckpt) else None
    if ck:  # same method, cfg and seed already trained (gsm8k: test seed 0 = the select run): skip the loop
        assert set(ck["state"]) == {n for n, p in mod.named_parameters() if p.requires_grad}, f"{ckpt} does not match {method}"
        mod.load_state_dict(ck["state"], strict=False)
        curve, perm = ck["curve"], perm[:0]
    for i in range(0, len(perm) - bs * acc + 1, bs * acc):
        (ro or model).train()  # a readout keeps the frozen LLM in eval mode
        opt.zero_grad()
        step = 0.0
        for j in range(acc):
            s, c = nll(model, ro, feats, *batch(X, perm[i + j * bs:i + (j + 1) * bs]))
            (s / (c * acc)).backward()
            step += s.item() / (c * acc)
        curve.append(step)
        wb.log(len(curve), **{pre + "train_nll": step})
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if probe and time.time() - t0 > probe:  # probe: fixed wall-clock budget, then dev only
            break
    if ckpt and not ck:  # atomic: a reader never sees a partial file
        state = {n: p.detach().cpu() for n, p in mod.named_parameters() if p.requires_grad}
        torch.save({"state": state, "curve": curve}, f"{ckpt}.{os.getpid()}")
        os.replace(f"{ckpt}.{os.getpid()}", ckpt)
    dev = ppl(model, ro, feats, data["dev"][:len(data["dev"]) // 4] if probe else data["dev"], bs)
    wb.log(len(curve), **{pre + "dev_ppl": dev})
    ev = getattr(a, "evaluate", None)  # gsm8k: test generation, run here while LoRA / gates are still attached
    res = {"dev": dev, "test": None if probe else ev(model, ro) if ev else ppl(model, ro, feats, data["test"], bs),
           "params": sum(p.numel() for p in params), "curve": {"train_nll": curve}}  # trainable: LoRA A/B + gate encoder/head, or readout
    if ck:
        res["from_ckpt"] = ckpt
    if method == "lora_pm":
        res["lora_rank"] = r
    if isinstance(ro, GateLoRA):
        ro.remove()
    if method in ("lora", "lora_pm"):
        model.unload()  # strip adapters, restore the frozen base for the next run
    return res


def frozen(a):  # tokenizer, frozen bf16 LLM, its feature extractor
    path = get_model_path(a.model_family, a.model_size)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="eager").to(DEV)
    model.requires_grad_(False)
    return AutoTokenizer.from_pretrained(path), model, Feats(model)


def geometry(model, feats, x):  # readout dims + W_O, checked on sample ids x
    f, ref = feats(x), model(x).logits.float()
    # hidden_states[-1] must be the post-norm state the LM head reads, else the readout corrects the wrong vector
    assert (head(model, f["r"][:, :, -1]) - ref).abs().max() < 1e-2 * ref.abs().max(), "last hidden state is not the LM head input"
    return {"L": f["r"].shape[2], "D": f["r"].shape[3], "W_O": weights(model)[0].to(DEV), "nheads": geom(model)[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_size", required=True)
    ap.add_argument("--dataset", choices=list(DATA), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="lora,lora_pm,mlp_last,weighted,dwatt,gin_cayley,gin_fc,deepset,comp_gnn,comp_noedge,comp_resonly")
    ap.add_argument("--train_tokens", type=int, default=2_000_000)
    ap.add_argument("--eval_tokens", type=int, default=262_144)
    ap.add_argument("--bs", type=int, default=4)  # sequences per micro-batch; gradient accumulation keeps TOK_STEP
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok, model, feats = frozen(a)
    data = {s: chunks(tok, a.dataset, i, n) for i, (s, n) in enumerate([("train", a.train_tokens), ("dev", a.eval_tokens // 4), ("test", a.eval_tokens)])}
    data |= geometry(model, feats, data["dev"][:1].to(DEV))
    res = {"model": f"{a.model_family}_{a.model_size}", "dataset": a.dataset, "tokens": {s: data[s].numel() for s in ("train", "dev", "test")},
           "frozen": {"dev": ppl(model, None, feats, data["dev"], a.bs), "test": ppl(model, None, feats, data["test"], a.bs)}, "methods": {}}
    print("[ntp] frozen", json.dumps(res["frozen"]), res["tokens"], flush=True)
    for m in a.methods.split(","):
        t0 = time.time()
        wb.init("ntp", res["model"], a.dataset, f"{m}_s{a.seed}", vars(a) | {"method": m, "frozen": res["frozen"]})
        # 3-option lr probe, 20 s of training each; the first process to finish it fixes the cfg for every seed (os.link is atomic)
        sp = a.out.rsplit("_s", 1)[0] + "__probe.json"
        if not os.path.exists(sp):
            probe = [c | {"dev": train(m, c, model, feats, data, 0, a.bs, a, probe=20)["dev"]}
                     for c in ({"lr": lr, "r": 2, "nl": 2} for lr in (1e-4, 5e-4, 1e-3))]
            print("[ntp] probe", m, probe, flush=True)
            with open(sp + f".{os.getpid()}", "w") as fo:
                json.dump({"probe": probe, "cfg": min(probe, key=lambda c: c["dev"])}, fo)
            try:
                os.link(sp + f".{os.getpid()}", sp)
            except FileExistsError:
                pass
            os.remove(sp + f".{os.getpid()}")
        pr = json.load(open(sp))
        res |= {"method": m, "cfg": pr["cfg"], "probe": pr["probe"], "seed": a.seed} | train(m, pr["cfg"], model, feats, data, a.seed, a.bs, a)
        res["minutes"] = (time.time() - t0) / 60
        print("[ntp] cell", json.dumps({x: res[x] for x in res if x != "curve"}), flush=True)
        with open(a.out, "w") as fo:
            json.dump(res, fo, indent=1)
        wb.finish({x: res[x] for x in res if x != "curve"})
    print("[ntp] done", flush=True)


if __name__ == "__main__":
    main()
