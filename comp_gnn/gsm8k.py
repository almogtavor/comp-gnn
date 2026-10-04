"""GSM8K (openai/gsm8k "main") with the ntp.py methods on a frozen LLM: readouts, LoRA, gated LoRA, frozen (0-shot), and frozen8
(the frozen model with 8 fixed train examples in the training format before each question as in lm-eval's gsm8k 8-shot; KV-cached greedy).

Train on `Question: {q}\\nAnswer: {gold rationale ending #### N}<eos>` (BOS first, right-padded to 512, loss on answer tokens
only) for one epoch of train minus a seeded 500-example dev split, with ntp.train's optimizer / grad accumulation.
Grid (4 cfgs): lr in (1e-4, 5e-4) x (r=2, nl=1 / r=8, nl=2) - r is the (gated) LoRA rank, nl the readout depth.
Two stages, one cell per process so a job array runs each in its own process; a cell whose json exists is skipped:
  select: (method, cfg) trained with seed 0 -> <out>/<m>__cfg<k>.json (dev answer NLL)
  test:   (method, seed) waits for all 4 select cells, the first to see them fixes the cfg in <m>__cfg.json (os.link is
          atomic), trains, then greedy-decodes the 1319 test questions (256 new tokens, full forward every step, so each new
          token gets its own readout / gate) -> <m>_s<seed>.json; metric = exact match of the number after ####, or, when a
          generation has no #### (frozen bases), of its last number before any "Question:" continuation (lm-eval's
          flexible-extract, same rule for every method; test_em_strict keeps the ####-only score). --rescore re-extracts saved jsons.
          --shard i --shards N splits one test cell's generation over N processes: shard 0 trains and saves the weights, the others
          load them (identical weights, greedy, so the gens equal the unsharded run's), each decodes its contiguous slice to
          <out>/shards/, and the shard that sees all N slices merges them into <m>_s<seed>.json.
Run from the ILSE-main root:  python -m comp_gnn.gsm8k --model_family F --model_size S --methods M --stage select --out DIR
"""
import argparse
import glob
import json
import math
import os
import re
import tempfile
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from comp_gnn import ntp, wb
from comp_gnn.ntp import Feats, GateLoRA, Readout, batch, frozen, geometry, head, logits, nll, ppl, train

SEQ, NDEV, NEW = 512, 500, 256
GRID = [{"lr": lr, "r": r, "nl": nl} for r, nl in ((2, 1), (8, 2)) for lr in (1e-4, 5e-4)]


def examples(split):  # dev: NDEV train questions picked with a fixed seed; train: the rest
    ds = load_dataset("openai/gsm8k", "main", split="test" if split == "test" else "train")
    dev = set(torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:NDEV].tolist())
    out = list(ds) if split == "test" else [ex for i, ex in enumerate(ds) if (i in dev) == (split == "dev")]
    cap = os.environ.get("GSM8K_N")  # "train,dev,test" caps (Qwen3.6 time budget, disclosed): a seeded subset of each split
    cap = int(cap.split(",")[("train", "dev", "test").index(split)]) if cap else len(out)
    return [out[i] for i in sorted(torch.randperm(len(out), generator=torch.Generator().manual_seed(1))[:cap].tolist())]


def shots(k=8):  # the first k train examples outside dev (uncapped, so every model sees the same ones), training format
    ds = load_dataset("openai/gsm8k", "main", split="train")
    dev = set(torch.randperm(len(ds), generator=torch.Generator().manual_seed(0))[:NDEV].tolist())  # = examples()' dev
    ex = [ds[i] for i in range(len(ds)) if i not in dev][:k]
    return "".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in ex)


@torch.no_grad()
def fewshot(model, tok, test, bs, pre=None, new=NEW):  # frozen8: left-padded KV-cached greedy (no readout or gate, so the cache is exact)
    tok.padding_side, pre, gens = "left", pre or shots(), []
    tok.pad_token = tok.pad_token or tok.eos_token  # Llama3 ships without one; masked out, so any id works
    for i in range(0, len(test), bs):
        P = [pre + f"Question: {ex['question']}\nAnswer:" for ex in test[i:i + bs]]
        x = tok(P, return_tensors="pt", padding=True).to(ntp.DEV)
        y = model.generate(**x, max_new_tokens=new, do_sample=False, pad_token_id=tok.eos_token_id, stop_strings=["\nQuestion:"], tokenizer=tok)
        gens += tok.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
    return score(gens, test)


def encode(tok, ex):  # (prompt ids, answer ids); concatenated, so the test prompt is exactly the training prefix
    p = [tok.bos_token_id or tok.eos_token_id] + tok(f"Question: {ex['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
    return p, tok(" " + ex["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]


def tensor(tok, exs):  # (N, 2, SEQ) ids + labels (-100 on prompt and padding), right-padded with eos; over-length dropped
    X, k = torch.full((len(exs), 2, SEQ), tok.eos_token_id), 0
    X[:, 1] = -100
    for p, y in map(lambda ex: encode(tok, ex), exs):
        if len(p) + len(y) <= SEQ:
            X[k, 0, :len(p) + len(y)], X[k, 1, len(p):len(p) + len(y)], k = torch.tensor(p + y), torch.tensor(y), k + 1
    print(f"[gsm8k] {k} examples, {len(exs) - k} over {SEQ} tokens dropped", flush=True)
    return X[:k]


def answer(s, flex=False):  # the number after the last ####, as float when parseable; None without ####, unless flex
    s = s.split("\nQuestion:")[0]  # a self-generated next question (frozen8 continues the few-shot pattern) is not the answer
    if "####" not in s:  # flex (generations only, every method): the last number before a "Question:" continuation, as
        n = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)  # lm-eval's flexible-extract; frozen bases never emit ####
        return float(n[-1].replace(",", "")) if flex and n else None
    t = s.split("####")[-1].strip().split("\n")[0].replace(",", "").replace("$", "").strip()
    m = re.match(r"-?\d+(\.\d+)?", t)
    return float(m[0]) if m else t.rstrip(".") or None


@torch.no_grad()
def generate(model, ro, feats, prompts, eos, bs, new=NEW):
    """Greedy, right-padded: each sequence's next token is read at its own last real position and written right after it."""
    model.eval()
    if ro is not None:
        ro.eval()
    out = []
    for i in range(0, len(prompts), bs):
        out += _gen(model, ro, feats, prompts[i:i + bs], eos, new)
    return out


def _gen(model, ro, feats, P, eos, new):
    n = torch.tensor([len(p) for p in P], device=ntp.DEV)
    x = torch.full((len(P), int(n.max()) + new), eos, device=ntp.DEV)
    for b, p in enumerate(P):
        x[b, :len(p)] = torch.tensor(p)
    n0, done, c = n.clone(), torch.zeros(len(P), dtype=torch.bool, device=ntp.DEV), len(P)
    for _ in range(new):  # ponytail: full recompute per token (no KV cache: readouts / gates change every position)
        live = (~done).nonzero()[:, 0]  # finished rows are dropped: rows are independent (causal, right-padded)
        while True:
            try:
                t = torch.cat([_step(model, ro, feats, x[r, :int(n[r].max())], n[r]) for r in live.split(c)]); break
            except torch.OutOfMemoryError:  # exact: halve the row chunk (kept for later steps), redo this step only
                assert c > 1
            c //= 2
            torch.cuda.empty_cache()  # outside the handler: the live traceback would pin the failed attempt's tensors
        x[live, n[live]] = t
        fin = t == eos
        done[live[fin]] = True
        n[live[~fin]] += 1
        if done.all():
            break
    return [x[b, n0[b]:n[b]].tolist() for b in range(len(P))]


def _step(model, ro, feats, ids, n):  # greedy next token at each row's last real position
    ar = torch.arange(len(ids), device=ids.device)
    if isinstance(ro, Readout):  # causal per-token features: the readout is needed at the last real position only
        return head(model, ro({k: v[ar, n - 1][:, None] for k, v in feats(ids).items()}))[:, 0].argmax(-1)
    return logits(model, ro, feats, ids)[ar, n - 1].argmax(-1)  # LLM logits / gated LoRA (gates at every position)


def evaluate(model, ro, feats, tok, test, bs, new=NEW, old=None, kv=False):
    """kv: KV-cached decode (_gen_kv). old (same weights only): this slice's gens from the 256-token run; only the ones that
    hit that cap (>= 240 re-encoded tokens) are re-decoded (greedy: the uncapped ones cannot change)."""
    P = [encode(tok, ex)[0] for ex in test]
    if old is None and not kv:
        return score([tok.decode(g, skip_special_tokens=True) for g in generate(model, ro, feats, P, tok.eos_token_id, bs, new)], test)
    idx = list(range(len(P))) if old is None else [i for i, g in enumerate(old) if len(tok(g, add_special_tokens=False)["input_ids"]) >= 240]
    gens = list(old or [""] * len(P))
    for j in range(0, len(idx), bs):
        for i, g in zip(idx[j:j + bs], _gen_kv(model, ro, feats, [P[i] for i in idx[j:j + bs]], tok.eos_token_id, new)):
            gens[i] = tok.decode(g, skip_special_tokens=True)
    return score(gens, test) | {"extended": len(idx)}


@torch.no_grad()
def _gen_kv(model, ro, feats, P, eos, new):
    """Left-padded KV-cached greedy for every method: readouts and gates are causal per-token functions of the frozen
    forward, so caching it (and, for gated LoRA, the gated forward) gives the same tokens as _gen's full recompute up to
    bf16 kernel order (selfcheck: equal on the tiny model)."""
    model.eval(), ro is not None and ro.eval()
    n = max(map(len, P))
    ids = torch.tensor([[eos] * (n - len(p)) + p for p in P], device=ntp.DEV)
    mask = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in P], device=ntp.DEV)
    pos, c1, c2 = (mask.cumsum(1) - 1).clamp(min=0), DynamicCache(config=model.config), DynamicCache(config=model.config)
    out, done = [[] for _ in P], torch.zeros(len(P), dtype=torch.bool, device=ntp.DEV)
    for _ in range(new):  # ponytail: finished rows keep decoding (masked out of the result); drop them from the cache if it matters
        kw = {"attention_mask": mask, "position_ids": pos, "use_cache": True, "logits_to_keep": 1}
        if isinstance(ro, Readout):
            t = head(model, ro({k: v[:, -1:] for k, v in feats(ids, past_key_values=c1, **kw).items()}))[:, 0].argmax(-1)
        elif isinstance(ro, GateLoRA):
            t = ro(model, feats, ids, kw | {"past_key_values": c1}, kw | {"past_key_values": c2})[:, -1].argmax(-1)
        else:
            t = model(ids, past_key_values=c1, **kw).logits[:, -1].argmax(-1)
        for b in (~done).nonzero()[:, 0].tolist():
            out[b] += [] if int(t[b]) == eos else [int(t[b])]
        done |= t == eos
        if done.all():
            break
        ids, mask, pos = t[:, None], torch.cat([mask, torch.ones_like(mask[:, :1])], 1), pos[:, -1:] + 1
    return out


def score(gens, test):  # gold is strict (always has ####); em_strict = the old ####-only metric, kept for the disclosure
    hit, strict = ([answer(g, f) is not None and answer(g, f) == answer(ex["answer"]) for g, ex in zip(gens, test)] for f in (True, False))
    if wb.RUN:
        import wandb
        wb.RUN.log({"examples": wandb.Table(columns=["question", "gold", "generation", "correct"],
                                            data=[[ex["question"], ex["answer"], g, h] for ex, g, h in list(zip(test, gens, hit))[:50]])})
    return {"em": sum(hit) / len(hit), "em_strict": sum(strict) / len(strict), "gens": gens}


def rescore(out):  # re-extract test_em from the saved generations of every <m>_s<seed>.json in out (same GSM8K_N cap)
    test = examples("test")
    for f in sorted(glob.glob(f"{out}/*_s[0-9]*.json")):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:  # a copy truncated mid-write: left as is
            print(f"[gsm8k] unreadable, skipped {f}", flush=True); continue
        assert len(r["gens"]) == len(test), f"{f}: {len(r['gens'])} gens vs {len(test)} test questions"
        e = score(r["gens"], test)
        r |= {"test_em": e["em"], "test_em_strict": e["em_strict"]}
        with open(f"{f}.{os.getpid()}", "w") as fo:
            json.dump(r, fo, indent=1)
        os.replace(f"{f}.{os.getpid()}", f)
        print(f"[gsm8k] rescored {f}: {r['test_em']:.4f} (strict {r['test_em_strict']:.4f})", flush=True)


def save(p, obj):  # atomic and first-writer-wins: a reader never sees a partial json
    with open(f"{p}.{os.getpid()}", "w") as fo:
        json.dump(obj, fo, indent=1)
    try:
        os.link(f"{p}.{os.getpid()}", p)
    except FileExistsError:
        pass
    os.remove(f"{p}.{os.getpid()}")


def merge(out, m, x, n, test):  # all n slice jsons present -> the full <m>_s<x>.json (shard 0 carries curve / dev / params)
    sh = [f"{out}/shards/{m}_s{x}_{i}of{n}.json" for i in range(n)]
    if not all(map(os.path.exists, sh)):
        return print(f"[gsm8k] {m}_s{x}: waiting for the other shards to merge", flush=True)
    r = [json.load(open(f)) for f in sh]
    gens = [g for ri in r for g in ri["gens"]]
    e = score(gens, test)
    save(f"{out}/{m}_s{x}.json", r[0] | {"test_em": e["em"], "test_em_strict": e["em_strict"], "gens": gens, "shards": n,
                                       "minutes": max(ri["minutes"] for ri in r)})


def chosen(out, m, ks):  # blocks until the select cells ks exist; the first test process to see them writes the choice
    sp, sel = f"{out}/{m}__cfg.json", [f"{out}/{m}__cfg{k}.json" for k in ks]
    while not os.path.exists(sp):
        if all(map(os.path.exists, sel)):
            r = [{k: v for k, v in json.load(open(f)).items() if k in ("cfg", "dev_nll")} for f in sel]
            save(sp, {"select": r, "cfg": min(r, key=lambda c: c["dev_nll"])["cfg"]})
        else:
            print(f"[gsm8k] {m}: waiting for select cells", flush=True)
            time.sleep(60)
    return json.load(open(sp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family")
    ap.add_argument("--model_size")
    ap.add_argument("--methods", default="frozen,lora,gnn_lora,gin_cayley,comp_cayley_nores,comp_cayley_ln,comp_similarity,comp_noedge,"
                                       "comp_hier_cayley,comp_hier_xl")  # comp_similarity / comp_hier_xl read $SEDGES / $XEDGES
    ap.add_argument("--stage", choices=("select", "test"))
    ap.add_argument("--cfgs", default="0,1,2,3")  # GRID indices: select trains them, test picks among them ("2,3" = new rows' short grid)
    ap.add_argument("--seeds", default="0,1,2")  # test: frozen / frozen8 run seed 0 only (no training, greedy)
    ap.add_argument("--out")  # results dir, shared by all cells of one model
    ap.add_argument("--bs", type=int, default=4)  # sequences per micro-batch; ntp.train accumulates to 2048 tokens of padded width
    ap.add_argument("--gen_bs", type=int, default=32)
    ap.add_argument("--shard", type=int, default=0)  # test: this process decodes slice shard of shards (see module doc)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--new", type=int, default=NEW)  # test new-token budget, every method (8000 = room for Qwen3.6's thinking trace, disclosed)
    ap.add_argument("--kv", action="store_true")  # test: KV-cached decode (see _gen_kv)
    ap.add_argument("--extend", default=None)  # test, with --kv: dir of the 256-token run; re-decode only its capped gens (see evaluate)
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--rescore", action="store_true")  # re-extract test_em of every test json in --out, no GPU
    a = ap.parse_args()
    if a.selfcheck or a.rescore:
        return selfcheck() if a.selfcheck else rescore(a.out)
    ms, sel = a.methods.split(","), a.stage == "select"
    cells = [(m, int(k)) for m in ms if m[:6] != "frozen" for k in a.cfgs.split(",")] if sel else \
        [(m, int(s)) for m in ms for s in (a.seeds.split(",") if m[:6] != "frozen" else [0])]
    sh = lambda m, x: f"{a.out}/shards/{m}_s{x}_{a.shard}of{a.shards}.json"
    cells = [(m, x) for m, x in cells if not os.path.exists(f"{a.out}/{m}__cfg{x}.json" if sel else f"{a.out}/{m}_s{x}.json")
             and (sel or a.shards == 1 or not os.path.exists(sh(m, x)))]
    if not cells:
        return print("[gsm8k] all cells done, skip", flush=True)
    os.makedirs(f"{a.out}/shards", exist_ok=True)
    pick = {} if sel else {m: chosen(a.out, m, a.cfgs.split(",")) for m, _ in cells if m[:6] != "frozen"}  # wait before taking the GPU's memory
    tok, model, feats = frozen(a)
    data = {s: tensor(tok, examples(s)) for s in ("train", "dev")}
    data |= geometry(model, feats, batch(data["dev"], slice(0, 1))[0])
    full, name, args = None if sel else examples("test"), f"{a.model_family}_{a.model_size}", dict(vars(a))
    test = full and full[a.shard * len(full) // a.shards:(a.shard + 1) * len(full) // a.shards]
    old = lambda m, x: a.extend and json.load(open(f"{a.extend}/{m}_s{x}.json"))["gens"][a.shard * len(full) // a.shards:(a.shard + 1) * len(full) // a.shards]
    for m, x in cells:
        a.evaluate = (lambda model, ro: None) if sel else (lambda model, ro: evaluate(model, ro, feats, tok, test, a.gen_bs, a.new, old(m, x) or None, a.kv))
        t0, cfg = time.time(), GRID[x] if sel else pick.get(m, {}).get("cfg")
        wb.init("gsm8k", name, "gsm8k", f"{m}_cfg{x}" if sel else f"{m}_s{x}", args | {"method": m, "cfg": cfg} | ({"seed": x} if not sel else {}))
        ck = None if m[:6] == "frozen" or (not sel and x and a.shards == 1) else \
            f"{a.out}/{m}__cfg{GRID.index(cfg)}" + ("" if sel or x == 0 else f"_s{x}") + ".pt"  # select trains with seed 0  # seeds > 0 keep weights only to share them across shards
        while ck and a.shard and not os.path.exists(ck):  # shards > 0 decode with shard 0's weights
            print(f"[gsm8k] {m}_s{x}: waiting for shard 0's weights", flush=True)
            time.sleep(30)
        res = train(m, cfg, model, feats, data, 0 if sel else x, a.bs, a, ckpt=ck) if m[:6] != "frozen" else \
            {"dev": ppl(model, None, feats, data["dev"], a.bs), "params": 0, "curve": {"train_nll": []},
             "test": a.evaluate(model, None) if m == "frozen" else fewshot(model, tok, test, a.gen_bs, new=a.new)}
        res = {"model": name, "method": m, "cfg": cfg, "dev_nll": math.log(res.pop("dev"))} | res | {"minutes": (time.time() - t0) / 60}
        T = len(res["curve"]["train_nll"])
        wb.log(T, dev_nll=res["dev_nll"])
        if not sel:
            res |= {"seed": x, "select": pick.get(m, {}).get("select"), "test_em": res["test"]["em"], "test_em_strict": res["test"]["em_strict"],
                    "extended": res["test"].get("extended"), "new": a.new, "gens": res.pop("test")["gens"]}
            wb.log(T, test_em=res["test_em"])
        print("[gsm8k] cell", json.dumps({k: v for k, v in res.items() if k not in ("curve", "gens", "test")}), flush=True)
        if sel or a.shards == 1:
            save(f"{a.out}/{m}__cfg{x}.json" if sel else f"{a.out}/{m}_s{x}.json", res)
        else:
            save(sh(m, x), res | {"shard": a.shard})
            merge(a.out, m, x, a.shards, full)
        wb.finish({k: v for k, v in res.items() if k not in ("curve", "gens", "test")})
    print("[gsm8k] done", flush=True)


def selfcheck():  # CPU, offline tiny model: answer parser, label masking, padded NLL, batched greedy == one-by-one greedy
    import comp_gnn.run as run
    run.DEV = ntp.DEV = "cpu"
    for s, v in [("so 5 #### 1,234", 1234.0), ("#### $18.", 18.0), ("#### -3.5\nmore", -3.5), ("no answer 42", None),
                 ("#### 7 #### 8", 8.0), ("####", None), ("#### 0.50", 0.5), ("#### abc.", "abc"), ("#### 72.00", 72.0)]:
        assert answer(s) == v, (s, answer(s), v)
    for s, v in [(" 112", 112.0), (" $1,000\n\nQuestion: 7 apples?\nAnswer: 7", 1000.0), ("so 3 then 4.5.", 4.5), ("none", None),
                 ("x 2 #### 9", 9.0), ("so 2 #### abc.", "abc"), ("#### 5\n\nQuestion: q\nAnswer: #### 9", 5.0)]:  # flex: #### still wins when present
        assert answer(s, True) == v, (s, answer(s, True), v)
    assert score(["#### 3", " 3", "4"], [{"answer": "#### 3"}] * 3) | {"gens": 0} == {"em": 2 / 3, "em_strict": 1 / 3, "gens": 0}
    path = "hf-internal-testing/tiny-random-Gemma2ForCausalLM"
    tok, model = AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path, attn_implementation="eager")
    feats = Feats(model.requires_grad_(False).eval())
    exs = [{"question": "q " * k, "answer": f"step {k}\n#### {k}"} for k in (1, 5, 9)] + [{"question": "x " * SEQ, "answer": "#### 1"}]
    X = tensor(tok, exs)
    assert len(X) == 3, "over-length example kept"
    for x, ex in zip(X, exs):
        (p, y), m = encode(tok, ex), x[1] != -100
        assert m.nonzero()[:, 0].tolist() == list(range(len(p), len(p) + len(y))) and (x[0][m] == x[1][m]).all() and x[0, :len(p)].tolist() == p
    ids, lab = batch(X, slice(0, 3))
    assert lab.shape[1] == int((X[:, 1] != -100).nonzero()[:, 1].max()) + 1, "batch not trimmed to the last label"
    one = [nll(model, None, feats, *batch(X, slice(i, i + 1))) for i in range(3)]
    s, c = nll(model, None, feats, ids, lab)
    assert c == sum(n for _, n in one) and abs(s - sum(v for v, _ in one)) < 1e-3 * abs(s), "right padding changes the NLL"
    g = geometry(model, feats, ids[:1])
    ro = Readout("mlp_last", g["L"], g["D"], 1, 0.0, g["W_O"], g["nheads"])
    torch.nn.init.normal_(ro.up.weight, std=0.5)  # non-trivial readout: tokens differ from the frozen model's
    P = [encode(tok, ex)[0] for ex in exs]
    for r in (None, ro):
        e = generate(model, r, feats, P[:1], -1, 1, 3)[0][1]  # a token sequence 0 emits: exercises eos stopping mid-batch
        for eos in (tok.eos_token_id, e):
            out = generate(model, r, feats, P, eos, 4, 12)
            assert out == [o for p in P for o in generate(model, r, feats, [p], eos, 1, 12)], "batched greedy != one-by-one"
            assert all(len(o) <= 12 and eos not in o for o in out)
            global _step
            step, _step = _step, lambda *a: (_ for _ in ()).throw(torch.OutOfMemoryError()) if len(a[3]) > 1 else step(*a)
            try:  # every multi-row step OOMs: the per-step halving must still give the same tokens
                assert generate(model, r, feats, P, eos, 4, 12) == out, "OOM row split != unsplit"
            finally:
                _step = step
    gate = GateLoRA(model, "mlp_last", 2, g, ["q_proj", "v_proj"])
    for prm in gate.parameters():
        torch.nn.init.normal_(prm, std=0.3) if prm.requires_grad else None
    for r in (None, ro, gate):  # KV-cached left-padded decode (--extend) == the full-recompute decode
        assert _gen_kv(model, r, feats, P, tok.eos_token_id, 12) == generate(model, r, feats, P, tok.eos_token_id, 4, 12), f"cache != recompute: {type(r)}"
    gate.remove()
    pre, q = "Question: 1+1\nAnswer: 2 #### 2\n\n", [{"question": "q " * k, "answer": "#### 1"} for k in (1, 6, 3)]
    pad, tok.pad_token = tok.pad_token, None  # as Llama3: fewshot must supply one
    gb = fewshot(model, tok, q, 3, pre)["gens"]  # frozen8: left-padded batch == one-by-one == the uncached greedy
    assert gb == fewshot(model, tok, q, 1, pre)["gens"], "left padding changes frozen8 generations"
    P1 = [tok(pre + f"Question: {e['question']}\nAnswer:")["input_ids"] for e in q]
    assert [g.split("\nQuestion:")[0] for g in gb] == [tok.decode(o, skip_special_tokens=True).split("\nQuestion:")[0] for o in generate(model, None, feats, P1, tok.eos_token_id, 1)]
    tok.padding_side, tok.pad_token = "right", pad
    n, ar = torch.tensor([3, 7, 5]), torch.arange(3)  # last-position readout (generate's fast path) == full-sequence logits
    assert torch.allclose(head(model, ro({k: v[ar, n - 1][:, None] for k, v in feats(ids).items()}))[:, 0],
                          logits(model, ro, feats, ids)[ar, n - 1], atol=1e-4)
    d, ck = {"train": torch.cat([X] * 3), "dev": X} | g, f"{tempfile.mkdtemp()}/ck.pt"  # 9 sequences = 2 optimizer steps at bs 1
    a = argparse.Namespace(evaluate=lambda model, ro: None)
    r0 = train("mlp_last", GRID[1], model, feats, d, 0, 1, a, ckpt=ck)
    r1 = train("mlp_last", GRID[1], model, feats, d, 1, 1, a, ckpt=ck)  # seed 1 would train differently: equal dev => loaded
    a.model_family = "Gemma2"  # lora_pm: smallest rank with counted params >= gnn_lora's (LoRA params are linear in r)
    gl, pm = (train(m, GRID[0], model, feats, d, 0, 1, a) for m in ("gnn_lora", "lora_pm"))
    assert pm["params"] >= gl["params"] > pm["params"] * (pm["lora_rank"] - 1) / pm["lora_rank"] and pm["lora_rank"] > GRID[0]["r"], (gl["params"], pm)
    assert not any(p.requires_grad for p in model.parameters()) and "lora" not in str(model), "LoRA left attached"
    assert len(r0["curve"]["train_nll"]) == 2 and "from_ckpt" not in r0 and r1["from_ckpt"] == ck
    assert r1["dev"] == r0["dev"] and r1["curve"] == r0["curve"], "checkpoint reload != the trained run"
    o, T = tempfile.mkdtemp(), [{"answer": f"#### {k}"} for k in range(5)]  # merge: slices concatenate in order, rescored on all
    os.makedirs(f"{o}/shards")
    for i, sl in enumerate((slice(0, 2), slice(2, 5))):
        save(f"{o}/shards/m_s1_{i}of2.json", {"gens": [f"#### {k}" if k != 3 else "x" for k in range(5)][sl], "minutes": i, "curve": i})
        merge(o, "m", 1, 2, T)
        assert os.path.exists(f"{o}/m_s1.json") == (i == 1), "merged before all shards finished"
    r = json.load(open(f"{o}/m_s1.json"))
    assert r["gens"] == ["#### 0", "#### 1", "#### 2", "x", "#### 4"] and r["test_em"] == 0.8 and r["curve"] == 0 and r["minutes"] == 1
    print("gsm8k ok", flush=True)


if __name__ == "__main__":
    main()
