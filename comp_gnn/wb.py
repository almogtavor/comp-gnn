"""Optional W&B logging (entity/project from $WANDB_ENTITY / $WANDB_PROJECT).
No-op without WANDB_API_KEY, so everything runs offline by default.
One run per (kind, model, task, method[, seed]); a deterministic id makes a restarted job replace its run.
Everything is logged against "t" (epoch or optimizer step), so per-seed series overlay instead of concatenating.
Backfill (runs that finished without W&B, and run.py classification, which has no live hooks):
    python -m comp_gnn.wb --backfill results/<Fam>/<file>.json ...
"""
import hashlib
import json
import os
import sys

RUN = None


def init(kind, fam, task, method, config):
    global RUN
    if not os.environ.get("WANDB_API_KEY"):
        return
    import wandb
    finish()
    name = f"{kind}-{fam}-{task}-{method}"
    # a restarted job retrains the cell from scratch: resuming would append a second curve and W&B adds ~a unix timestamp to
    # _runtime on resume, so the stale attempt is replaced instead
    if old := _api_run(name):
        old.delete()
    for attempt in range(3):  # a slow/unreachable W&B server must not kill the run: results land on disk anyway
        try:
            RUN = _init(wandb, name, kind, fam, task, config)
            break
        except Exception as e:
            print(f"[wb] init attempt {attempt} failed: {e}", flush=True)
            wandb.teardown()  # a timed-out init still holds the run id in the local wandb service ("run ID is in use")
    else:
        print("[wb] W&B unavailable, continuing without it", flush=True)
        return
    RUN.define_metric("*", step_metric="t")


def _init(wandb, name, kind, fam, task, config):
    return wandb.init(entity=os.environ.get("WANDB_ENTITY"),
                      project=os.environ.get("WANDB_PROJECT", "comp-gnn"),
                      group=f"{kind}/{fam}/{task}", name=name, id=hashlib.md5(name.encode()).hexdigest()[:16], resume="allow",
                      job_type=kind, tags=[kind, fam, task], config=config,
                      settings=wandb.Settings(init_timeout=300))


def log(t, **kv):
    if RUN:
        RUN.log({"t": t} | kv)


def finish(summary=None):
    global RUN
    if RUN:
        RUN.summary.update(summary or {})
        RUN.finish()
        RUN = None


def _api_run(name):  # the existing W&B run for name, else None
    import wandb
    try:
        api = wandb.Api(timeout=60)
        return api.run(f"{os.environ.get('WANDB_ENTITY', api.default_entity)}/{os.environ.get('WANDB_PROJECT', 'comp-gnn')}/"
                                         f"{hashlib.md5(name.encode()).hexdigest()[:16]}")
    except Exception:
        return None


def exists(name):  # True if the run already has history (backfill is not re-logged)
    r = _api_run(name)
    return r is not None and r.lastHistoryStep >= 0


def backfill(path):
    d, fam = json.load(open(path)), path.rstrip("/").split("/")[-2 - ("ntp_fix" in path)]
    fam = d.get("model") or {"Pythia": "Pythia_410m", "Gemma2": "Gemma2_2B", "Llama3": "Llama3_8B"}.get(fam, fam)
    if "curve" in d:  # ntp.py cell: one method, one seed
        cells = [("ntp", d["dataset"], f"{d['method']}_s{d['seed']}", [d["curve"]], {x: d[x] for x in d if x != "curve"})]
    else:  # run.py (classification) / sts.py: all seeds of each method
        kind, task = ("cls", d["task"]) if "task" in d else ("sts", "STS")
        cells = [(kind, task, m, r["seed_curves"], {x: r[x] for x in r if x != "seed_curves"}) for m, r in d["methods"].items()
                 if "seed_curves" in r]
    for kind, task, m, curves, summary in cells:
        if exists(f"{kind}-{fam}-{task}-{m}"):
            print(f"[wb] {kind}-{fam}-{task}-{m} already logged", flush=True)
            continue
        init(kind, fam, task, m, {"method": m, "source": path})
        if not RUN:
            print(f"[wb] no W&B, {kind}-{fam}-{task}-{m} not backfilled", flush=True)
            continue
        RUN.tags = RUN.tags + ("backfill",)
        for s, c in enumerate(curves):
            pre = f"s{s}/" if len(curves) > 1 or kind != "ntp" else ""
            for i in range(max(map(len, c.values()))):
                log(i + 1, **{pre + k: v[i] for k, v in c.items() if i < len(v)})
        finish(summary)
        print(f"[wb] backfilled {kind}-{fam}-{task}-{m} from {path}", flush=True)


if __name__ == "__main__":
    assert sys.argv[1] == "--backfill", "usage: python -m comp_gnn.wb --backfill <result.json> ..."
    for f in sys.argv[2:]:
        backfill(f)
