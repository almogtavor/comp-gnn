"""Paper tables (booktabs LaTeX) + status.md from synced result jsons (fixed-config runs only).
Usage: python agg_paper.py [results_root] [out_dir]   (defaults: ../results, this dir; status.md goes to ../)
       python agg_paper.py --selftest
Reads, per run dir <results_root>/<run>/ (one dir is enough; several machines' copies can sit side by side):
  results/<Fam>/<Task>Classification__*.json   run.py, methods with "fixed_cfg"
  results/<Fam>/sts__g*.json                   sts.py, methods with "fixed_cfg"
  lora/<Fam>/<Task>Classification/fixed_{,bs256_}seed*.json   lora.py (bs 32 = Cayley-Encoder exact, main; bs 256 = disclosed ablation)
  results/<Fam>/ntp_fix/wikitext__<m>_s<seed>.json, wikitext__<m>__probe.json   ntp.py
  results/<Fam>/gsm8k_fixed/<m>_s<seed>.json (test), <m>__cfg<k>.json (dev select)   gsm8k.py
Rows in DROPPED exceeded their matched baseline's parameter count and are excluded; check_budget() fails the run if any
reported row of ours has more trainable params than gin_cayley (readouts) or lora_pm (gated LoRA).
Everything else is listed as ignored in status.md. Same relative path in several run dirs: newest mtime wins.
"""
import ast, csv, glob, json, os, re, statistics as st, sys, time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
MODELS = [("Pythia", "Pythia-410M"), ("Gemma2", "Gemma-2-2B"), ("Llama3", "Llama-3-8B"), ("Qwen36MoE", "Qwen3.6-35B-A3B")]
GEN_MODELS = MODELS[1:]  # generation (NTP, GSM8K) skips Pythia
SKIP_FAMS = {"Gemma4": "Gemma4, dropped from the paper", "Qwen3MoE": "Qwen3-30B-A3B, superseded"}
TASKS = ["Banking77", "Emotion", "MTOPDomain", "MTOPIntent", "PoemSentiment"]
ILSE = [("mlp_last", "MLP (last layer)"), ("weighted", "Weighted"), ("dwatt", "DWAtt"), ("deepset", "DeepSet"),
        ("gin_fc", "FC-Encoder"), ("gin_cayley", "Cayley-Encoder")]
OURS = [("comp_cayley_nores", "Comp Cayley, no residual"), ("comp_cayley_ln", "Comp Cayley, no residual, LayerNorm"),
        ("comp_hier_cayley", "Hierarchical Cayley"), ("comp_hier_mean", "Hier. mean-aggr"), ("comp_hier_xl", "Hier. Cayley + cross-layer"),
        ("comp_similarity", "Comp similarity edges"), ("comp_noedge_ln", "Comp. DeepSets"), ("comp_resonly_ln", "Comp residual-only")]
DROPPED = {"comp_gnn", "comp_cayley", "comp_noedge", "comp_resonly", "comp_hier_readout", "comp_hier_deep", "comp_noexp"}
GATED = ["mlp_lora", "noedge_lora", "gnn_lora"]
FLAT = [(m + "_flat", n + " (no type balancing)") for m, n in OURS if m in ("comp_cayley_nores", "comp_cayley_ln", "comp_similarity", "comp_noedge_ln")]  # MoE-only ablation
BAL = [("comp_hier_bal", "Hier. type-balanced aggr"), ("comp_hier_gate", "Hier. router-gate-weighted aggr")]  # MoE-only: per-type (or router-gate) weighted mean of component messages into mega-nodes
E10 = [(m + "_e10", n + " (45/45/10 type weights)") for m, n in OURS + BAL if m in ("comp_hier_bal", "comp_hier_xl", "comp_similarity", "comp_cayley_ln", "comp_noedge_ln")]  # MoE-only, cls
CLS_M = ILSE + OURS + BAL + FLAT + E10
STS_M = ILSE + OURS + BAL + FLAT
MOE_ONLY = {m for m, _ in BAL + FLAT + E10}
for_fam = lambda fam, ms: [x for x in ms if "MoE" in fam or x[0] not in MOE_ONLY]  # these rows exist for the MoE model only
NOT_RUN = {"Qwen36MoE": {"comp_hier_mean"}}  # cls showed type-balanced aggr is the MoE variant; mean-aggr control not run on NTP
STS_T = ["STSBenchmark", "STS12", "STS13", "STS14", "STS15", "STS16", "BIOSSES", "SICK-R"]
NTP_M = [("lora", "LoRA ($r{=}2$)"), ("lora_pm", "LoRA ($r$ matched)"), ("mlp_lora", "MLP-gated LoRA"),
         ("noedge_lora", "Comp. DeepSets gated LoRA"), ("gnn_lora", "Comp-GNN gated LoRA")] + [(m, n + " readout") for m, n in ILSE + OURS]
LRS = [1e-4, 5e-4, 1e-3]
PAPER_ROWS = ["Cayley-Encoder", "FC-Encoder", "Last layer", "DeepSet", "MLP (last layer)"]  # order of paper_cayley_fc_last_deepset_mlplast


def fixed_dicts():
    """FIXED dicts straight from the repo sources, so the hyperparameter table cannot drift from the code."""
    out = {}
    for f in ("run.py", "lora.py"):
        for node in ast.parse(open(os.path.join(REPO, "ilse_repro", f)).read()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "FIXED":
                out[f] = ast.literal_eval(node.value)
    return out


def readable(f):
    try:
        json.load(open(f)); return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def latest_files(root):
    """{relpath: (abs path, run)}; newest readable copy wins across run dirs (a copy can be truncated mid-write)."""
    best = {}
    for f in glob.glob(f"{root}/*/**/*.json", recursive=True):
        if os.path.basename(f).startswith("._"):
            continue
        cl, rel = os.path.relpath(f, root).split(os.sep, 1)
        if rel not in best or (readable(f), os.path.getmtime(f)) > (readable(best[rel][0]), os.path.getmtime(best[rel][0])):
            best[rel] = (f, cl)
    return best


def load(root):
    D = {"cls": defaultdict(dict), "base": {}, "sts": defaultdict(dict), "lora": defaultdict(dict), "lora256": defaultdict(dict),
         "ntp": defaultdict(dict), "gsm8k": defaultdict(dict), "params": defaultdict(dict), "frozen": {}, "probe": {}, "ignored": [], "where": defaultdict(set)}
    ign = lambda rel, why: D["ignored"].append((rel, why))
    for rel, (f, cl) in sorted(latest_files(root).items(), key=lambda kv: os.path.getmtime(kv[1][0])):
        parts = rel.split(os.sep)
        if parts[0] == "results" and len(parts) > 3 and parts[2] == "lora":  # some runs write results/<Fam>/lora/<Task>/...
            parts = ["lora", parts[1]] + parts[3:]
        fam = parts[1] if len(parts) > 2 else None
        if fam in SKIP_FAMS:
            ign(rel, SKIP_FAMS[fam]); continue
        if fam not in dict(MODELS):
            ign(rel, "unknown model family"); continue
        try:
            r = json.load(open(f))
        except (json.JSONDecodeError, UnicodeDecodeError):
            ign(rel, "unreadable/partial json"); continue
        name = parts[-1]
        if parts[0] == "lora":
            if name.startswith(("fixed_seed", "fixed_bs256_seed")):
                D["lora256" if "bs256" in name else "lora"][fam, parts[2].replace("Classification", "")][r["seed"]] = 100 * r["test_at_best_val"]
                D["where"]["LoRA"].add(cl)
            else:
                ign(rel, "LoRA Optuna-era (not fixed cfg)")
        elif parts[2:3] == ["ntp_fix"]:
            m = re.match(r"(\w+?)__(\w+?)(?:_s(\d+)|__probe)\.json$", name)
            if not m or m.group(1) != "wikitext":
                ign(rel, "ntp_fix non-wikitext dataset"); continue
            meth = m.group(2)
            if meth in DROPPED:
                ign(rel, "over parameter budget, dropped"); continue
            if m.group(3) is None:
                D["probe"][fam, meth] = r
            else:
                D["ntp"][fam, meth][int(m.group(3))] = r
                D["params"]["ntp", fam][meth] = r["params"]
                D["probe"].setdefault((fam, meth), {"probe": r["probe"], "cfg": r["cfg"]})
                D["frozen"][fam] = r["frozen"]
            D["where"]["NTP"].add(cl)
        elif parts[2:3] == ["gsm8k"] and name in ("frozen_s0.json", "frozen8_s0.json", "frozen8_8k_s0.json"):  # no trainable size, so the earlier run is the same eval
            D["gsm8k"][fam, name[:-8]][0] = r["test_em"]; D["params"]["gsm8k", fam][name[:-8]] = 0
        elif parts[2:3] == ["gsm8k"]:
            ign(rel, "GSM8K with dev-selected size (superseded by gsm8k_fixed)")
        elif parts[2:3] == ["gsm8k_fixed"]:
            m = re.match(r"(\w+?)(?:_s(\d+)|__cfg(\d+))\.json$", name)
            if not m or m.group(1) in DROPPED:
                ign(rel, "over parameter budget, dropped" if m else "gsm8k unrecognised file"); continue
            if m.group(2) is not None:
                D["gsm8k"][fam, m.group(1)][int(m.group(2))] = r["test_em"]
                D["params"]["gsm8k", fam][m.group(1)] = r.get("params", 0)
            else:
                D["params"][f"gsm8k_cfg{m.group(3)}", fam][m.group(1)] = r["params"]
            D["where"]["GSM8K"].add(cl)
        elif parts[2:3] == ["ntp_fan"] or name.startswith("ntp_"):
            ign(rel, "old NTP run (ntp_fan / per-method grid)")
        elif name.startswith("sts__"):
            n = 0
            for meth, v in r.get("methods", {}).items():
                if meth in DROPPED:
                    ign(f"{rel}:{meth}", "over parameter budget, dropped")
                elif "fixed_cfg" in v:
                    D["sts"][fam][meth] = v["test_seeds"]; n += 1
                    D["params"]["sts", fam][meth] = v["params"]
                    D["where"]["STS"].add(cl)
                else:
                    ign(f"{rel}:{meth}", "STS grid-selected (no fixed_cfg)")
            D["base"].setdefault(("sts", fam), {"Last layer": r["lastlayer"], "Best single layer (dev)": r["bestlayer"]["test"]})
        elif "Classification__" in name:
            t = r["task"].replace("Classification", "")
            D["base"][fam, t] = (r.get("lastlayer_kshot_test"), r.get("paper_cayley_fc_last_deepset_mlplast"))
            for meth, v in r.get("methods", {}).items():
                if meth in DROPPED:
                    ign(f"{rel}:{meth}", "over parameter budget, dropped")
                elif "fixed_cfg" in v:
                    D["cls"][fam, t][meth] = ([100 * x for x in v["head_test_seeds"]], v["fixed_cfg"])
                    D["params"]["cls", fam, t][meth] = v["params"]
                    D["where"]["Classification"].add(cl)
                else:
                    ign(f"{rel}:{meth}", "classification Optuna/grid-era (no fixed_cfg)")
            for meth in r.get("grid_methods", {}):
                ign(f"{rel}:grid_methods.{meth}", "classification grid search (not reported)")
        else:
            ign(rel, "unrecognised format")
    return D


def check_budget(D):
    """Never more params than the matched baseline: ours <= gin_cayley, gated LoRA of ours <= lora_pm (same cfg/task)."""
    bad = []
    for key, d in D["params"].items():
        for ref, ms in (("gin_cayley", [m for m, _ in OURS + BAL + FLAT + E10]), ("lora_pm", GATED[1:])):
            bad += [f"{key} {m} {d[m]} > {ref} {d[ref]}" for m in ms if ref in d and m in d and d[m] > d[ref]]
    assert not bad, "parameter budget violated:\n" + "\n".join(bad)


def fmtp(ps):
    """Trainable params (mean over the given counts, e.g. tasks whose heads differ by #classes), in millions."""
    ps = [p for p in ps if p is not None]
    return f"{st.mean(ps) / 1e6:.2f}M" if ps else "--"


# ---------- LaTeX ----------
def cell(xs, rank=None, dig=2, full=3):
    """rank 0 = best (bold), 1 = second best (underlined). A str (e.g. a params column) is printed as is."""
    if isinstance(xs, str):
        return xs
    if not xs:
        return "--"
    s = f"{st.mean(xs):.{dig}f}" + (f"{{\\scriptsize$\\pm${st.stdev(xs):.{dig}f}}}" if len(xs) > 1 else "")
    s = {0: f"\\textbf{{{s}}}", 1: f"\\underline{{{s}}}"}.get(rank, s)
    return s + (f"\\textsuperscript{{({len(xs)})}}" if len(xs) < full else "")


def best_idx(rows, ci, hi, dig=2):
    """{row index: rank} for column ci: 0 (bold) for every row tied at the best printed value, 1 (underline) for the next value."""
    v = {i: round(st.mean(r[ci]), dig) for i, r in enumerate(rows) if isinstance(r[ci], list) and r[ci]}
    top = sorted(set(v.values()), reverse=hi)[:2]
    return {i: top.index(x) for i, x in v.items() if x in top}


def ref(xs, n):
    """A reference row (one value per column, None = missing) in block()'s format, padded with -- to n columns."""
    return [[] if x is None else [x] for x in xs] + [""] * (n - len(xs))  # blank: references have no trainable params


def avg(r):
    """Avg column: per-seed mean over the columns (first n common seeds); -- unless every column has results."""
    return [st.mean(c) for c in zip(*r)] if all(r) else []


def block(rows, names, hi=True, dig=2, nref=0, nfree=0):
    """rows: list of per-column value lists (hi: bool or one bool per column); bold best, underline second best per column, over every row.
    The first nref rows are single-value references (see ref()), followed by a cmidrule; the first nfree of them are not ranked
    and get a dagger where they beat the bold."""
    hs = [hi[c] if isinstance(hi, list) else hi for c in range(len(rows[0]))] if rows else []
    b = [{i + nfree: k for i, k in best_idx(rows[nfree:], c, h, dig).items()} for c, h in enumerate(hs)]
    top = [max((round(st.mean(r[c]), dig) * (1 if h else -1) for r in rows[nfree:] if isinstance(r[c], list) and r[c]), default=None) for c, h in enumerate(hs)]
    dag = lambda i, c, x: (r"$^\dagger$" if i < nfree and isinstance(x, list) and x and top[c] is not None
                           and round(st.mean(x), dig) * (1 if hs[c] else -1) > top[c] else "")
    out = [f"{n} & " + " & ".join(cell(x, b[c].get(i), dig, full=1 if i < nref else 3) + dag(i, c, x) for c, x in enumerate(r)) + r" \\"
           for i, (n, r) in enumerate(zip(names, rows))]
    return out[:nref] + [r"\cmidrule(l){1-%d}" % (len(rows[0]) + 1)] * (0 < nref < len(out)) + out[nref:]


def env(cap, lab, colspec, header, body, wide=True):
    e = "table*" if wide else "table"
    return [f"\\begin{{{e}}}[tp]", r"\centering", r"\small", f"\\caption{{{cap}}}", f"\\label{{{lab}}}",
            r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.92\textheight}", f"\\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule",
            *body, r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", f"\\end{{{e}}}", ""]


def tables(D, FX):
    T = []
    SEEDNOTE = "Mean{\\scriptsize$\\pm$std} over seeds; a superscript $(n)$ marks cells with fewer than 3 seeds; -- = missing or still running."
    # (a) classification
    body = []
    for fam, mname in MODELS:
        body.append(f"\\multicolumn{{{len(TASKS) + 3}}}{{l}}{{\\textit{{{mname}}}}} \\\\")
        refs = []
        for i, pr in enumerate(PAPER_ROWS):
            v = [D["base"].get((fam, t), (None, None))[1] for t in TASKS]
            v = [x[i] if x and i < len(x) else None for x in v]
            refs.append((f"\\quad {pr} (Cayley-Encoder reported)", v + [None if None in v else st.mean(v)]))
        refs = [x for x in refs if any(y is not None for y in x[1])]  # Cayley-Encoder published no Qwen3.6 numbers
        npaper = len(refs)
        v = [D["base"].get((fam, t), (None,))[0] for t in TASKS]
        refs.append((r"\quad Last layer, 8-shot LR (ours)", [None if x is None else 100 * x for x in v] + [None if None in v else 100 * st.mean(v)]))
        cm = for_fam(fam, CLS_M)
        rows = [[D["cls"].get((fam, t), {}).get(m, ([], None))[0] for t in TASKS] for m, _ in cm]
        rows = [r + [avg(r), fmtp(D["params"]["cls", fam, t].get(m) for t in TASKS)] for r, (m, _) in zip(rows, cm)]
        body += block([ref(v, len(TASKS) + 2) for _, v in refs] + rows,
                      [n for n, _ in refs] + [f"\\quad {n}" + (" (ours)" if m.startswith("comp") else "") for m, n in cm], nref=len(refs), nfree=npaper)
        body.append(r"\midrule")
    T += env("Classification test accuracy (\\%) with Cayley-Encoder's fixed per-task hyperparameters (Table~\\ref{tab:hparams}); trained head, best-val checkpoint. "
             "Cayley-Encoder reported = numbers from the Cayley-Encoder paper (MTEB 8-shot protocol) for reference, not ranked; $^\\dagger$ = above the best ranked value. "
             "Bold = best, underline = second best per model and column over all other rows (ties share the mark). "
             "Params = trainable parameters, mean over the five tasks (heads differ by number of classes). " + SEEDNOTE,
             "tab:cls", "l" + "c" * (len(TASKS) + 1) + "r", "Method & " + " & ".join(t + "$\\uparrow$" for t in TASKS) + " & Avg$\\uparrow$ & Params", body[:-1])
    # (b) STS
    body = []
    for fam, mname in MODELS:
        body.append(f"\\multicolumn{{{len(STS_T) + 3}}}{{l}}{{\\textit{{{mname}}}}} \\\\")
        refs = (D["base"].get(("sts", fam)) or {}).items()
        rows = [ref([100 * d[t] for t in STS_T] + [100 * st.mean(d[t] for t in STS_T)], len(STS_T) + 2) for _, d in refs]
        for m, _ in for_fam(fam, STS_M):
            seeds = D["sts"][fam].get(m, [])
            rows.append([[100 * s[t] for s in seeds] for t in STS_T] + [[100 * st.mean(s[t] for t in STS_T) for s in seeds]] + [fmtp([D["params"]["sts", fam].get(m)])])
        body += block(rows, [f"\\quad {bn}" for bn, _ in refs] + [f"\\quad {n}" + (" (ours)" if m.startswith("comp") else "") for m, n in for_fam(fam, STS_M)], nref=len(refs))
        body.append(r"\midrule")
    T += env("STS test Spearman correlation ($\\times$100), fixed config for every method (Cayley-Encoder trainer defaults: lr $10^{-3}$, wd $10^{-4}$, dropout 0.1, "
             "2 GNN/MLP layers, 1 for Weighted/DWAtt; 25 epochs), best-dev checkpoint. Last/best single layer rows are unsupervised references. Bold = best, underline = second best per model and column over all rows (ties share the mark). " + SEEDNOTE,
             "tab:sts", "l" + "c" * (len(STS_T) + 1) + "r", "Method & " + " & ".join(t + "$\\uparrow$" for t in STS_T) + " & Avg$\\uparrow$ & Params", body[:-1])
    # (c) LoRA
    rows = [[list(D[k].get((fam, t), {}).values()) for t in TASKS] for fam, _ in MODELS for k in ("lora", "lora256")]
    body = [f"{n}{b} & " + " & ".join(map(cell, r + [avg(r)])) + r" \\"
            for (n, b), r in zip([(n, b) for _, n in MODELS for b in ("", " (bs 256)")], rows) if any(r)]  # bs 256 was not run on Qwen3.6
    T += env("LoRA fine-tuning test accuracy (\\%) at the best-validation epoch with Cayley-Encoder's LoRA trainer defaults (rank 2, $\\alpha$ 16, dropout 0.1, "
             "wd $10^{-4}$, batch 32, lr $5\\cdot10^{-4}$, 20 epochs; no tuning; Qwen3.6-35B-A3B: 3 epochs on a 1000-example training subset, for compute). ``(bs 256)'' rows are a disclosed earlier run with batch 256 and "
             "lr $5\\cdot10^{-4}\\sqrt{8}$ (sqrt scaling) for compute; it diverged on Llama3 (majority-class collapse on Poem/Emotion), so it is not the main row. Rows are models, not compared against each other, so nothing is bolded. " + SEEDNOTE,
             "tab:lora", "l" + "c" * (len(TASKS) + 1), "Model & " + " & ".join(TASKS) + " & Avg", body, wide=False)
    # (d) WikiText PPL
    rows = [[x for f, _ in GEN_MODELS for x in ([D["frozen"][f]["test"]] if f in D["frozen"] else [], "0")]]
    rows += [[x for f, _ in GEN_MODELS for x in (("n/r", "n/r") if m in NOT_RUN.get(f, ()) else
              ([c["test"] for c in D["ntp"].get((f, m), {}).values()], fmtp([D["params"]["ntp", f].get(m)])))] for m, _ in NTP_M]
    names = ["Frozen"] + [n + (" (ours)" if "comp" in m or "Comp" in n else "") for m, n in NTP_M]
    body = block(rows, names, hi=False, nref=1)
    T += env("WikiText-103 test perplexity (lower is better) of a frozen LLM with a trained readout correction or (gated) LoRA, 500k training tokens, "
             "lr from the probe in Table~\\ref{tab:probe}. Frozen is the unmodified model (deterministic, one value). Bold = best, underline = second best per model. "
             "Params = trainable parameters. n/r = not run: on Qwen3.6 the mean-aggr control was dropped after classification, where type-balanced aggregation is the MoE variant (its NTP graph has no expert nodes, so it reduces to Hierarchical Cayley). The two plain-LoRA rows differ only in rank (same target modules, $\\alpha$ 16, dropout 0.1, lr selection): LoRA ($r{=}2$) uses Cayley-Encoder's rank, LoRA ($r$ matched) the smallest rank whose counted trainable parameters are $\\geq$ Comp-GNN gated LoRA's ($r$ = 5 Pythia, 4 Gemma2, 4 Llama3, 3 Qwen3.6). " + SEEDNOTE,
             "tab:ntp", "l" + "cr" * len(GEN_MODELS), "Method & " + " & ".join(f"{n} PPL$\\downarrow$ & Params" for _, n in GEN_MODELS), body, wide=False)
    # (e) LR probe
    body = []
    for fam, mname in GEN_MODELS:
        body.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{mname}}}}} \\\\")
        for m, n in NTP_M:
            if m in NOT_RUN.get(fam, ()): continue
            p = D["probe"].get((fam, m))
            if not p:
                body.append(f"\\quad {n} & -- & -- & -- \\\\"); continue
            dev = {c["lr"]: c["dev"] for c in p["probe"]}
            body.append(f"\\quad {n} & " + " & ".join(
                "--" if lr not in dev else (f"\\textbf{{{dev[lr]:.2f}}}" if lr == p["cfg"]["lr"] else f"{dev[lr]:.2f}") for lr in LRS) + r" \\")
        body.append(r"\midrule")
    T += env("Learning-rate probe for Table~\\ref{tab:ntp}: dev perplexity after 20\\,s of training for each of 3 options lr $\\in\\{10^{-4}, 5\\cdot10^{-4}, 10^{-3}\\}$ "
             "(seed 0, evaluated on 1/4 of the dev slice). The chosen lr (bold, lowest dev ppl) is shared by all 3 seeds of the full run (500k training tokens). "
             "LoRA rank 2 and readout depth 2 are fixed. -- = probe not finished.", "tab:probe", "lccc",
             "Method & $10^{-4}$ & $5\\cdot10^{-4}$ & $10^{-3}$", body[:-1], wide=False)
    # (f) hyperparameters
    body = [f"{t.replace('Classification', '')} & {c['lr']:g} & {c['wd']:g} & {c['dropout']:g} & {c['nl']} \\\\" for t, c in FX["run.py"].items()]
    lc = FX["lora.py"]
    T += env("Fixed hyperparameters (no search on our side). Classification: the per-task configurations Cayley-Encoder selected with Optuna on Pythia-410M, "
             "reused unchanged for every model and every method (ours included); Weighted and DWAtt always use 1 layer. "
             "Adam, cross-entropy, ReduceLROnPlateau (patience 3, factor 0.5), 50 epochs, best-val checkpoint. "
             "STS: Cayley-Encoder's published Optuna table has no STS rows, so we use their GIN trainer defaults (lr $10^{-3}$, wd $10^{-4}$, dropout 0.1) "
             "with 2 layers instead of the trainer default of 3 (outside the paper's search range $\\{1,2\\}$), 1 for Weighted/DWAtt, 25 epochs. "
             f"LoRA classification: Cayley-Encoder LoRA trainer defaults (rank {lc['lora_r']}, $\\alpha$ {lc['lora_alpha']}, dropout {lc['lora_dropout']}, "
             f"lr {lc['lr']:g}, wd {lc['weight_decay']:g}, batch 32, 20 epochs, exactly as Cayley-Encoder, except Qwen3.6: 3 epochs, 1000 training examples; a batch-256 ablation is in Table~\\ref{{tab:lora}}). NTP: lr chosen by the probe of Table~\\ref{{tab:probe}}, "
             "AdamW (wd $10^{-4}$ readouts, 0 LoRA), LoRA rank 2, $\\alpha$ 16, dropout 0.1; readout depth 2, dropout 0.1.",
             "tab:hparams", "lcccc", "Task & lr & wd & dropout & layers", body, wide=False)
    return T


MAIN_E10 = {"Qwen36MoE": {"comp_noedge_ln", "comp_cayley_ln", "comp_similarity"}}  # main-table rows shown with 45/45/10 type weights
MAIN_OURS = [("comp_noedge_ln", "Comp. DeepSets"), ("comp_cayley_ln", "Comp-Cayley"), ("comp_hier_cayley", "Hier-Cayley"), ("comp_hier_xl", "Hier-Cayley + cross-layer"), ("comp_similarity", "Comp-Similarity")]


def main_tables(D):
    """The paper's two main tables: Cayley-Encoder-style classification + STS, and generation (WikiText PPL, GSM8K EM)."""
    body = []
    for fam, mname in MODELS:
        body.append(f"\\multicolumn{{{len(TASKS) + 4}}}{{l}}{{\\textit{{{mname}}}}} \\\\")
        sl = (D["base"].get(("sts", fam)) or {}).get("Last layer")
        ll = [D["base"].get((fam, t), (None,))[0] for t in TASKS] + [sl and st.mean(sl[t] for t in STS_T)]
        ll.append(None if None in ll[:-1] else st.mean(ll[:-1]))  # Avg excludes STS
        ms = ILSE + MAIN_OURS
        ks = [m + "_e10" if m in MAIN_E10.get(fam, ()) else m for m, _ in ms]  # result keys behind each row
        rows = [[D["cls"].get((fam, t), {}).get(m, ([], None))[0] for t in TASKS]
                + [[100 * st.mean(s[t] for t in STS_T) for s in D["sts"][fam].get(m, [])]] for m in ks]
        rows = [r + [avg(r[:-1]), fmtp(D["params"]["cls", fam, t].get(m) for t in TASKS)] for r, m in zip(rows, ks)]  # Avg excludes STS
        lines = block([ref([None if x is None else 100 * x for x in ll], len(TASKS) + 3)] + rows, [r"\quad Last layer"] + [f"\\quad {n}" for _, n in ms], nref=1)
        body += lines[:2 + len(ILSE)] + [r"\cmidrule(l){1-9}"] + [l.replace(" & ", " (ours) & ", 1) for l in lines[2 + len(ILSE):]] + [r"\midrule"]
    T = env("Classification accuracy (\\%) on Cayley-Encoder's five MTEB tasks and STS Spearman ($\\times$100, mean of STS12-16, STS-B, BIOSSES, SICK-R), "
            "our reproduction of every Cayley-Encoder baseline under Cayley-Encoder's fixed hyperparameters. Last layer = Cayley-Encoder's 8-shot logistic-regression probe "
            "(classification) / cosine of mean-pooled last-layer states (STS); unsupervised reference. "
            "Ours: Comp. DeepSets = the same attention-head/MLP component nodes with no message passing; "
            "Comp-Cayley = Cayley graph over the components (parameter-free LayerNorm on component inputs, as in Hier-Cayley); Hier-Cayley = components starred to Cayley-Encoder's layer Cayley graph, "
            "readout over layer nodes only; + cross-layer = fixed component edges from a FineWeb calibration pass. "
            "Avg = unweighted mean of the five classification columns (per seed); STS is excluded as too noisy. "
            "On Qwen3.6-MoE, Comp. DeepSets, Comp-Cayley and Comp-Similarity weight the per-type (heads / MLP / experts) pooled means 45 / 45 / 10\\% instead of a third each (same parameters; equal weights in the appendix). "
            "Params = trainable parameters of the classification head (mean over tasks); ours never exceed the Cayley-Encoder's. "
            "Bold = best, underline = second best per model and column over all rows (ties share the mark). Mean{\\scriptsize$\\pm$std} over 3 seeds; -- = not run.",
            "tab:main", "l" + "c" * (len(TASKS) + 2) + "r", "Method & " + " & ".join(t + "$\\uparrow$" for t in TASKS) + " & STS$\\uparrow$ & Avg$\\uparrow$ & Params", body[:-1])
    gm = [("frozen", "Frozen"), ("frozen8", "Frozen, 8-shot"), ("frozen8_8k", "Frozen, 8-shot, 8000 tokens")] + [(m, n) for m, n in NTP_M if m in ("lora", "lora_pm", "gnn_lora")] + [("gin_cayley", "Cayley-Encoder readout")] + MAIN_OURS
    rows = []
    for m, _ in gm:
        r = []
        for fam, _ in GEN_MODELS:
            ppl = ([D["frozen"][fam]["test"]] if fam in D["frozen"] else []) if m == "frozen" else [c["test"] for c in D["ntp"].get((fam, m), {}).values()]
            pn, pg = D["params"]["ntp", fam].get(m, 0 if m[:6] == "frozen" else None), D["params"]["gsm8k", fam].get(m)
            assert None in (pn, pg) or abs(pn - pg) <= 70_000, f"{fam} {m}: WikiText P {pn} != GSM8K P {pg}"  # same architecture on both
            r += [ppl, [100 * x for x in D["gsm8k"].get((fam, m), {}).values()], fmtp([pn if pn is not None else pg])]
        rows.append(r)
    body = block(rows, [f"{n}{' (ours)' if m.startswith('comp') else ''}" for m, n in gm], hi=[c % 3 == 1 for c in range(3 * len(GEN_MODELS))], nref=3, nfree=3)
    T += env("Generation: WikiText-103 test perplexity ($\\downarrow$) and GSM8K test exact match (\\%, $\\uparrow$, greedy, 256 new tokens). "
             "Every method corrects the frozen LLM's next-token logits per token (readouts) or gates per-layer LoRA per token (GNN-LoRA). "
             "Frozen = unmodified model, 0-shot (it never emits the trained \\#\\#\\#\\# answer marker, so EM uses lm-eval's flexible extract: the last number before any self-generated next question, the same rule for every method; Qwen3.6 opens with a thinking trace and rarely finishes it in 256 tokens). Frozen, 8-shot = the same model with 8 fixed GSM8K train examples in the training format before each question (lm-eval's gsm8k 8-shot setting; no PPL, as WikiText has no few-shot variant; Qwen3.6 still opens with a thinking trace, and only 2 of its 100 test answers finish it within 256 tokens). Frozen, 8-shot, 8000 tokens = the same with up to 8000 new tokens, so Qwen3.6's thinking trace can finish (a larger budget than any trained row gets). P = trainable parameters, the same on both benchmarks: every method keeps 2 GIN readout layers and LoRA rank 2 except LoRA ($r$ matched), and only the learning rate is picked on dev (WikiText from 1e-4 / 5e-4 / 1e-3, GSM8K from 1e-4 / 5e-4). "
             "The two plain-LoRA rows differ only in rank (same target modules, $\\alpha$ 16, dropout 0.1, lr selection): LoRA ($r{=}2$) uses Cayley-Encoder's rank, LoRA ($r$ matched) the smallest rank whose counted trainable parameters are $\\geq$ Comp-GNN gated LoRA's ($r$ = 5 Pythia, 4 Gemma2, 4 Llama3, 3 Qwen3.6). Bold = best, underline = second best per column among the trained rows (the frozen rows are not ranked; $^\\dagger$ = above the bold). Mean{\\scriptsize$\\pm$std} over 3 seeds; -- = not run.",
             "tab:gen", "l" + "ccr" * len(GEN_MODELS),
             "Method & " + " & ".join(f"{n} PPL$\\downarrow$ & {n} EM$\\uparrow$ & P" for _, n in GEN_MODELS), body)
    return T


def status(D, root):
    now = time.strftime("%Y-%m-%d %H:%M")
    L = [f"# ILSE repro vs component GNNs - status ({now})", "",
         f"Source: `{os.path.relpath(root, REPO)}/<run>`. Run dirs with data per experiment: "
         + ", ".join(f"{k}: {'/'.join(sorted(v))}" for k, v in sorted(D["where"].items())) + ".", "",
         "| Experiment | Model | complete (3 seeds) | partial | missing |", "|---|---|---|---|---|"]
    for fam, mname in MODELS:
        cells = [(f"{t}/{m}", len(D["cls"].get((fam, t), {}).get(m, ([],))[0])) for t in TASKS for m, _ in for_fam(fam, CLS_M)]
        cells_s = [(m, len(D["sts"][fam].get(m, []))) for m, _ in for_fam(fam, STS_M)]
        cells_l = [(t, len(D["lora"].get((fam, t), {}))) for t in TASKS]  # bs 32
        cells_n = [(m, len(D["ntp"].get((fam, m), {}))) for m, _ in NTP_M]
        cells_p = [(m, 3 * ((fam, m) in D["probe"])) for m, _ in NTP_M]
        cells_g = [(m, len(D["gsm8k"].get((fam, m), {}))) for m, _ in NTP_M]
        exps = [("Classification", cells), ("STS", cells_s), ("LoRA cls", cells_l)]
        for ex, cs in exps + [("NTP wikitext", cells_n), ("NTP lr probe", cells_p), ("GSM8K test", cells_g)] * (fam in dict(GEN_MODELS)):
            full = sum(n >= 3 for _, n in cs)
            part = [f"{k}({n})" for k, n in cs if 0 < n < 3]
            L.append(f"| {ex} | {mname} | {full}/{len(cs)} | {', '.join(part) or '-'} | {len(cs) - full - len(part)} |")
    L += ["", "## Automatic flags", ""]
    for (fam, t), d in sorted(D["cls"].items()):
        for m, (xs, _) in d.items():
            if xs and st.mean(xs) < 5:
                L.append(f"- classification {fam}/{t}/{m}: {st.mean(xs):.2f}% (near chance, likely diverged)")
    for fam, d in sorted(D["sts"].items()):
        for m, seeds in d.items():
            if len(seeds) > 1 and all(s == seeds[0] for s in seeds):
                L.append(f"- STS {fam}/{m}: all {len(seeds)} seeds give identical scores (seed has no effect; std 0 is not 3 independent runs)")
    by = defaultdict(list)
    for rel, why in D["ignored"]:
        by[why].append(rel)
    L += ["", "## Ignored inputs (not in tables)", ""]
    for why, rels in sorted(by.items()):
        L.append(f"- **{why}**: {len(rels)} - e.g. " + ", ".join(f"`{r}`" for r in rels[:4]) + (" ..." if len(rels) > 4 else ""))
    return L


def csvs(D, out):
    """Wide CSVs, one row per method; columns are `<model>_<col>` (mean over seeds) + `<model>_<col>_std`."""
    def write(k, cols, rows, std=True, models=MODELS):
        sfx = ("", "_std") if std else ("",)
        with open(os.path.join(out, f"{k}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method"] + [f"{f}_{c}{s}" for f, _ in models for c in cols for s in sfx])
            for m, get in rows:
                w.writerow([m] + [v for f, _ in models for c in cols for xs in [get(f, c)]
                                  for v in (f"{st.mean(xs):.4f}" if xs else "", f"{st.stdev(xs):.4f}" if len(xs) > 1 else "")[:len(sfx)]])
    write("cls", TASKS, [(m, lambda f, t, m=m: D["cls"].get((f, t), {}).get(m, ([],))[0]) for m, _ in CLS_M])
    sts = lambda f, t, m: [100 * (s[t] if t != "Avg" else st.mean(s[x] for x in STS_T)) for s in D["sts"][f].get(m, [])]
    write("sts", STS_T + ["Avg"], [(m, lambda f, t, m=m: sts(f, t, m)) for m, _ in STS_M])
    write("lora", TASKS, [(n, lambda f, t, k=k: list(D[k].get((f, t), {}).values())) for k, n in (("lora", "lora_bs32"), ("lora256", "lora_bs256"))])
    write("ntp", ["wikitext_test_ppl"], [("frozen", lambda f, _: [D["frozen"][f]["test"]] if f in D["frozen"] else [])]
          + [(m, lambda f, _, m=m: [c["test"] for c in D["ntp"].get((f, m), {}).values()]) for m, _ in NTP_M], models=GEN_MODELS)
    pr = lambda f, m: D["probe"].get((f, m), {"probe": [], "cfg": {}})
    write("probe", [f"dev_ppl_lr{lr:g}" for lr in LRS] + ["chosen_lr"],
          [(m, lambda f, c, m=m: [pr(f, m)["cfg"]["lr"]] if c == "chosen_lr" and pr(f, m)["cfg"] else
            [x["dev"] for x in pr(f, m)["probe"] if f"dev_ppl_lr{x['lr']:g}" == c]) for m, _ in NTP_M], std=False, models=GEN_MODELS)


def main(root, out):
    D = load(root)
    check_budget(D)
    os.makedirs(out, exist_ok=True)
    tex = ["% generated by agg_paper.py - needs \\usepackage{booktabs,adjustbox}", ""] + tables(D, fixed_dicts())
    open(os.path.join(out, "tables.tex"), "w").write("\n".join(tex))
    open(os.path.join(out, "main.tex"), "w").write("\n".join(["% generated by agg_paper.py - needs \\usepackage{booktabs,adjustbox}", ""] + main_tables(D)))
    csvs(D, out)
    open(os.path.join(os.path.dirname(out), "status.md"), "w").write("\n".join(status(D, root)) + "\n")
    return D


def selftest():
    import tempfile
    d = tempfile.mkdtemp()
    w = lambda p, o: (os.makedirs(os.path.dirname(f"{d}/res/{p}"), exist_ok=True), json.dump(o, open(f"{d}/res/{p}", "w")))
    pr = [{"lr": lr, "r": 2, "nl": 2, "dev": dv} for lr, dv in zip(LRS, (30.0, 20.0, 25.0))]
    for s in range(3):
        w(f"c1/results/Gemma2/ntp_fix/wikitext__lora_s{s}.json", {"frozen": {"dev": 40, "test": 41}, "cfg": pr[1], "probe": pr, "test": 21 + s, "dev": 20, "params": 7})
        w(f"c1/results/Llama3/gsm8k_fixed/gin_cayley_s{s}.json", {"test_em": 0.5 + s / 100, "params": 1_180_000})
        w(f"c1/lora/Pythia/EmotionClassification/fixed_bs256_seed{s}.json", {"seed": s, "test_at_best_val": 0.7})
        w(f"c1/lora/Pythia/EmotionClassification/fixed_seed{s}.json", {"seed": s, "test_at_best_val": 0.6})
    w("c1/results/Gemma2/ntp_fix/code__lora_s0.json", {})
    w("c2/results/Pythia/EmotionClassification__g0.json", {"task": "EmotionClassification", "lastlayer_kshot_test": 0.3, "methods": {
        "comp_cayley_ln": {"fixed_cfg": {"lr": 1e-3}, "params": 5, "head_test_seeds": [0.8, 0.82, 0.81]}, "gin_fc": {"head_test_seeds": [0.9]},
        "comp_gnn": {"fixed_cfg": {}, "params": 9, "head_test_seeds": [0.99] * 3}}})
    w("c2/results/Gemma4/sts__g0.json", {})
    D = main(f"{d}/res", f"{d}/out/pt")
    tex = open(f"{d}/out/pt/tables.tex").read()
    assert "22.00{\\scriptsize$\\pm$1.00}" in tex and "\\textbf{20.00}" in tex and "\\underline{41.00}" in tex, "ntp/probe"
    assert "70.00{\\scriptsize$\\pm$0.00}" in tex and "\\textbf{81.00" in tex, "lora/cls"
    whys = {w for _, w in D["ignored"]}
    assert {"ntp_fix non-wikitext dataset", "Gemma4, dropped from the paper", "classification Optuna/grid-era (no fixed_cfg)"} <= whys, whys
    gen = open(f"{d}/out/pt/main.tex").read()
    assert "51.00{\\scriptsize$\\pm$1.00}} & 1.18M" in gen and "99.00" not in tex, "gsm8k loader / dropped rows"
    assert "over parameter budget, dropped" in whys
    D["params"]["ntp", "X"] = {"gin_cayley": 10, "comp_cayley_ln": 11}
    try:
        check_budget(D); raise RuntimeError("budget check missed ours > gin_cayley")
    except AssertionError:
        pass
    assert "--" in tex and "–" not in tex and "—" not in tex
    ntp = open(f"{d}/out/pt/ntp.csv").read()
    assert ntp.startswith("method,Gemma2_wikitext_test_ppl,Gemma2_wikitext_test_ppl_std,Llama3_") and "\nlora,22.0000,1.0000,," in ntp \
        and "\nfrozen,41.0000,," in ntp, ntp
    assert "\nlora,30.0000,20.0000,25.0000,0.0005," in open(f"{d}/out/pt/probe.csv").read()
    lora = open(f"{d}/out/pt/lora.csv").read()
    assert lora.startswith("method,Pythia_Banking77,Pythia_Banking77_std,Pythia_Emotion,") and "\nlora_bs256,,,70.0000,0.0000," in lora and "\nlora_bs32,,,60.0000,0.0000," in lora, lora
    o = block([ref([90.0, 50.0], 2), [[80.0], [70.0]], [[85.0], [60.0]]], ["a", "b", "c"], nref=1, nfree=1)  # reference unranked, dagger above the bold
    assert o[0] == r"a & 90.00$^\dagger$ & 50.00 \\" and r"\textbf{70.00}" in o[2] and r"\textbf{85.00}" in o[3]
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results")),
             os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else HERE))
