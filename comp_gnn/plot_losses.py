"""Training curves from result jsons -> PNGs. Usage: python -m comp_gnn.plot_losses <results_dir> [figures_dir]
One figure per json: a panel per method with the train loss of each seed (solid) and the val metric (dashed, right axis).
Reads run.py/sts.py/ntp.py "seed_curves", ntp fan-out cells ("curve") and lora test_seed*.json ("epoch_logs").
Jsons from before curves were recorded are skipped.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def curves(res):
    """{method: [curve dict per seed]}, a curve dict maps metric name -> list per epoch/step."""
    if "epoch_logs" in res:  # lora: ILSE's per-epoch logs
        logs = res["epoch_logs"]
        return {"lora": [{k: [e[k] for e in logs] for k in ("train_loss", "val_loss", "val_acc")}]}
    if "curve" in res:  # ntp fan-out cell
        return {f'{res["method"]} cfg{res["cfg"]} s{res["seed"]}': [res["curve"]]}
    return {m: r["seed_curves"] for m, r in res.get("methods", {}).items() if "seed_curves" in r}


def plot(res, out):
    cs = curves(res)
    if not cs:
        return False
    fig, axs = plt.subplots(1, len(cs), figsize=(4 * len(cs), 3.2), squeeze=False)
    for ax, (m, seeds) in zip(axs[0], cs.items()):
        tw = None
        for s, c in enumerate(seeds):
            for k, v in c.items():
                if "loss" in k or "nll" in k:
                    ax.plot(v, color=f"C{s}", ls="-" if k.startswith("train") else ":", label=f"s{s} {k}")
                else:
                    tw = tw or ax.twinx()
                    tw.plot(v, color=f"C{s}", ls="--", label=f"s{s} {k}")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel("step" if any("nll" in k for k in seeds[0]) else "epoch")
        ax.legend(fontsize=6, loc="upper right")
        if tw:
            tw.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return True


def main(src, dst):
    n = 0
    for root, _, files in os.walk(src):
        for f in files:
            if f.endswith(".json"):
                try:
                    res = json.load(open(os.path.join(root, f)))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(res, dict):
                    n += plot(res, os.path.join(dst, os.path.relpath(root, src), f[:-5] + ".png"))
    print(f"[plot] {n} figures -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) == 1:  # self-check on synthetic results of each kind
        import tempfile
        d = tempfile.mkdtemp()
        c = {"train_loss": [1.0, 0.5], "val_acc": [0.3, 0.6]}
        for name, r in {"a.json": {"methods": {"gin": {"seed_curves": [c, c]}, "old": {}}},
                        "b.json": {"method": "lora", "cfg": {"lr": 1e-4}, "seed": 0, "curve": {"train_nll": [3.0, 2.5]}},
                        "c.json": {"epoch_logs": [{"train_loss": 1, "val_loss": 1, "val_acc": 0.5}]},
                        "d.json": {"methods": {"x": {"head_test": 0.5}}}}.items():
            json.dump(r, open(os.path.join(d, name), "w"))
        main(d, os.path.join(d, "fig"))
        assert sorted(os.listdir(os.path.join(d, "fig", "."))) == ["a.png", "b.png", "c.png"]
    else:
        main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures")
