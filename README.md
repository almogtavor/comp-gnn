# comp-gnn

Code for the Comp-GNN paper: graph encoders over the components (attention heads, MLPs, MoE experts) of a frozen LLM, plus the
ILSE baselines, the AR readout and the gated LoRA, on text classification, STS, WikiText-103 and GSM8K.

## What is where

| Paper | File | Method keys |
|---|---|---|
| ILSE baselines (Cayley GIN, FC GIN, DeepSets, last-layer MLP, weighted, DWAtt) | `comp_gnn/run.py`, `sts.py` | `gin_cayley`, `gin_fc`, `deepset`, `mlp_last`, `weighted`, `dwatt` |
| Component features (heads, MLP, MoE experts via JL) | `comp_gnn/components.py` | - |
| Comp DeepSets / Comp-Cayley | `comp_gnn/compgnn.py` | `comp_noedge_ln`, `comp_resonly_ln`, `comp_cayley_nores`, `comp_cayley_ln` |
| Hier-Cayley, MoE type balancing | `comp_gnn/hier.py` | `comp_hier_*`, `comp_hier_xl` |
| Cross-layer FineWeb calibration, Comp-Similarity | `comp_gnn/calib.py`, `hier.py` | `comp_hier_xl`, `comp_similarity` |
| AR readout (WikiText-103) | `comp_gnn/ntp.py` | same keys, plus `lora`, `lora_pm`, `gnn_lora` |
| GSM8K, gated LoRA | `comp_gnn/gsm8k.py` | `frozen`, `lora`, `lora_pm`, `gnn_lora`, ... |
| LoRA classification baseline | `comp_gnn/lora.py` | - |
| Paper tables | `tables/agg_paper.py` | - |

Optional W&B logging (`comp_gnn/wb.py`) is a no-op unless `WANDB_API_KEY` is set (`WANDB_ENTITY`, `WANDB_PROJECT` optional).

## Install

The code runs on top of the ILSE codebase, which is not included here. Clone it, then from its root:

```bash
source /path/to/comp-gnn/comp_gnn/setup.sh   # patches ILSE in place and pip-installs requirements
export PYTHONPATH=/path/to/comp-gnn:/path/to/comp-gnn/stub:.
```

`setup.sh` swaps the gated Gemma2/Llama3 repos for ungated mirrors (`unsloth/gemma-2-2b`, `NousResearch/Meta-Llama-3-8B`) and adds
Qwen3.6-35B-A3B (`Qwen36MoE`). For Qwen, also run `tf5` (defined by `setup.sh`) to switch to transformers>=5.5 and flash-linear-attention.
Models: `--model_family Pythia|Gemma2|Llama3|Qwen36MoE`, sizes `410m|2B|8B|35B-A3B`.

Self-checks (no GPU): `python -m comp_gnn.hier`, `python -m comp_gnn.calib --selfcheck`,
`python -m comp_gnn.gsm8k --selfcheck`, `python tables/agg_paper.py --selftest`.

## Running

All commands run from the ILSE root. `F`/`S` are family/size, `N` a seed or group tag.

**1. Features** (classification and STS):

```bash
python -m experiments.utils.precompute.precompute_pipeline --model_family F --model_size S --tasks TASK --output_dir emb --pooling_method mean --batch_size 64
python -m comp_gnn.components --emb_dir emb/F_S_mean_pooling --task TASK --model_family F --model_size S --batch_size 128
```

**2. Cross-layer edges** (FineWeb calibration, needed by `comp_hier_xl` and `comp_similarity`):

```bash
python -m comp_gnn.calib --model_family F --model_size S --out xedges/F.npy
export XEDGES=xedges/F.npy SEDGES=xedges/F_sim.npy   # Qwen36MoE cls/STS: xedges/F_exp.npy, xedges/F_sim_exp.npy
```

**3. Classification and STS tables:**

```bash
python -m comp_gnn.run --emb_dir emb/F_S_mean_pooling --task TASK --out results/F/TASK__gN.json --methods gin_cayley,gin_fc,deepset,mlp_last
python -m comp_gnn.sts --model_family F --model_size S --out results/F/sts__gN.json --methods gin_cayley,gin_fc,deepset,mlp_last
python -m comp_gnn.lora --task TASK --model_family F --model_size S --study_dir results/F/lora/TASK --worker 1   # workers 1..3 = seeds
```

Other method groups: `weighted,dwatt`; `comp_cayley_nores`; `comp_hier_*`; `comp_hier_xl`; `comp_similarity`; `comp_noedge_ln,comp_resonly_ln`.

**4. WikiText-103** (one job per method and seed 0..2; `--bs` 4 Pythia, 2 Gemma2, 1 Llama3, 2 Qwen):

```bash
python -m comp_gnn.ntp --model_family F --model_size S --dataset wikitext --methods M --seed 0 --bs 4 --train_tokens 500000 --out results/F/ntp_fix/wikitext__M_s0.json
```

**5. GSM8K** (select one of 4 configs on dev, then test 3 seeds; `comp_noedge_ln`, `comp_resonly_ln`, `lora_pm` use `--cfgs 2,3`):

```bash
python -m comp_gnn.gsm8k --model_family F --model_size S --methods M --stage select --cfgs 0 --out results/F/gsm8k_fixed   # cfgs 0..3
python -m comp_gnn.gsm8k --model_family F --model_size S --methods M --stage test --seeds 0 --gen_bs 100 --out results/F/gsm8k_fixed   # --gen_bs 16 for gated LoRA
```

Dense models use `peft<0.15`.

**6. Tables:** `python tables/agg_paper.py results/` reads `results/<run>/results/<Fam>/...` and `results/<run>/lora/<Fam>/...`
(each `<run>` is one copy of the ILSE root's `results/` and `lora/` outputs) and writes the LaTeX tables.
