import argparse
import json
import os
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from comp_gnn.components import lm_only
from comp_gnn import wb

import experiments.utils.model_definitions.gnn.optuna_runs.run_optuna_trial_lora as L
from experiments.utils.model_definitions.gnn.gnn_datasets import load_task_data
from experiments.utils.model_definitions.text_automodel_wrapper import get_model_path

try:  # Qwen3.6 has no HF seq-cls class: the same generic one HF uses for Llama/Qwen3 (last non-pad token -> score); the HF
    # checkpoint is multimodal (Qwen3_5MoeConfig), its model.* keys load into the generic class's AutoModel backbone as-is
    from transformers import AutoModelForSequenceClassification, Qwen3_5MoeConfig, Qwen3_5MoeTextConfig
    from transformers.modeling_layers import GenericForSequenceClassification
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoePreTrainedModel

    for _c in (Qwen3_5MoeConfig, Qwen3_5MoeTextConfig):  # .register() silently skips configs from transformers itself
        AutoModelForSequenceClassification._model_mapping._extra_content[_c] = type("Qwen3_5MoeForSequenceClassification", (
            GenericForSequenceClassification, Qwen3_5MoePreTrainedModel), {"config_class": _c})
    Qwen3_5MoeConfig.pad_token_id = None  # transformers 5.18 dropped it from the wrapper config; ILSE reads it before setting it
except ImportError:  # transformers 4 (dense models)
    pass
L.CLASSIFICATION_TASKS.append("PoemSentimentClassification")  # ponytail: in the paper's LoRA table, missing from their list
_peft = L.get_peft_model
STASH = {}


def _stash(*a, **k):  # keep a handle on the model their train loop builds
    if hasattr(getattr(a[0], "model", None), "language_model"):  # Gemma4: keep LoRA off the vision/audio towers
        a[1].target_modules = lm_only(a[1].target_modules)
    if a[0].config.model_type.startswith("qwen3_5_moe"):  # ILSE's q_proj/v_proj exist on the 10 full-attn layers only;
        a[1].target_modules = set(a[1].target_modules) | {"in_proj_qkv"}  # DeltaNet's fused q/k/v projection (disclosed)
        tc = a[0].config.get_text_config()  # their pad fix sets the top-level config; the generic head reads text_config
        tc.pad_token_id = tc.pad_token_id if tc.pad_token_id is not None else a[0].config.pad_token_id
    if STASH.get("ckpt"):  # same math, activations recomputed: Llama3-8B at bs 256 OOMs an H200 otherwise
        a[0].gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    STASH["m"] = _peft(*a, **k)
    return STASH["m"]


L.get_peft_model = _stash
_load = L.load_task_data


def _load_keep_rng(*a, **k):  # mteb.get_task reseeds torch/np/random with 42, which erased their per-run seed
    st = torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate()
    out = _load(*a, **k)
    torch.set_rng_state(st[0]), torch.cuda.set_rng_state_all(st[1]), np.random.set_state(st[2]), random.setstate(st[3])
    return out


L.load_task_data = _load_keep_rng


FIXED = {"lora_r": 2, "lora_alpha": 16, "lora_dropout": 0.1, "lr": 5e-4, "weight_decay": 1e-4}  # lora_trainer.py defaults


class TestOnBestVal:
    """Stands in for optuna's trial: on each val improvement, evaluates the stashed LoRA model on test."""
    def __init__(self, loader):
        self.loader, self.best_val, self.test = loader, 0.0, None

    def report(self, val, step):
        wb.log(step, val_acc=val)
        if val <= self.best_val + 0.001:  # their early-stopping improvement rule (min_delta)
            return
        self.best_val, m = val, STASH["m"]
        m.eval()
        ok = n = 0
        with torch.no_grad():
            for b in self.loader:
                out = m(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda())
                ok += (out.logits.argmax(-1).cpu() == b["labels"]).sum().item()
                n += len(b["labels"])
        self.test = ok / n
        wb.log(step, test_at_best_val=self.test)

    def should_prune(self):
        return False


def main():
    ap = argparse.ArgumentParser()
    for k in ("task", "model_family", "model_size", "study_dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--worker", type=int, required=True)  # = seed
    ap.add_argument("--batch_size", type=int, default=32)  # ILSE: 32; larger = disclosed compute deviation
    ap.add_argument("--epochs", type=int, default=20)  # ILSE: 20; fewer = disclosed (Qwen3.6 time budget)
    a = ap.parse_args()
    cfg = FIXED | {"lr": FIXED["lr"] * (a.batch_size / 32) ** 0.5}  # sqrt lr scaling (Adam) off their bs32 default
    tag = "" if a.batch_size == 32 else f"bs{a.batch_size}_"
    STASH["ckpt"] = a.batch_size > 32 and a.model_family in ("Llama3", "Qwen36MoE")  # ponytail: hardcoded; key on param count if more big models join
    L.TASK_NAME, L.MODEL_FAMILY, L.MODEL_SIZE = a.task, a.model_family, a.model_size
    os.makedirs(a.study_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(get_model_path(a.model_family, a.model_size))
    tok.pad_token = tok.pad_token or tok.eos_token
    test = load_task_data(a.task, "test")
    # PoemSentiment test has no "mixed" examples (3 of 4 classes), as in the paper; only check label ids are valid
    assert set(test["labels"]) <= set(range(load_task_data(a.task, "train")["num_classes"])), "test labels out of range"
    wb.init("lora", f"{a.model_family}_{a.model_size}", a.task, f"lora_{tag}s{a.worker}", vars(a) | cfg)
    loader = DataLoader(L.TextClassificationDataset(test["text"], test["labels"], tok), batch_size=256, collate_fn=L.collate_fn)
    fake = TestOnBestVal(loader)
    args = argparse.Namespace(task=a.task, model_family=a.model_family, model_size=a.model_size, epochs=a.epochs, batch_size=a.batch_size,
                              save_dir="/tmp/lora", seed=a.worker, trial=fake, **cfg)
    res = L.train_and_eval_lora(args)
    out = {"fixed_cfg": cfg, "batch_size": a.batch_size, "seed": a.worker,
           "val": res["best_val_acc"], "test_at_best_val": fake.test, "params": res["param_count"],
           "epoch_logs": res["epoch_logs"]}
    print("[lora] result " + json.dumps(out), flush=True)
    with open(os.path.join(a.study_dir, f"fixed_{tag}seed{a.worker}.json"), "w") as f:
        json.dump(out, f)
    for i, e in enumerate(out["epoch_logs"]):
        wb.log(i, **{f"epoch/{k}": v for k, v in e.items() if isinstance(v, (int, float))})
    wb.finish({k: v for k, v in out.items() if k != "epoch_logs"})


if __name__ == "__main__":
    r = []
    for s in (1, 2):
        torch.manual_seed(s)
        _load_keep_rng("PoemSentimentClassification", "validation")
        r.append(torch.rand(1).item())
    assert r[0] != r[1], "data loading still resets the RNG"
    main()
