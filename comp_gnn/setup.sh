# Patches a fresh ILSE checkout and installs the dependencies. Source it from the ILSE-main root: `source /path/to/comp-gnn/comp_gnn/setup.sh`.
# ponytail: ILSE's train->val split drops texts (needed to recompute components); keep them, embeddings/split unchanged
sed -i -e '242s/labels, metadata = load_embeddings_from_h5(train_h5, load_metadata=True)/labels, texts, metadata = load_embeddings_from_h5(train_h5, load_texts=True, load_metadata=True)/' -e '282s/texts=None,/texts=[texts[i] for i in train_indices],/' -e '292s/texts=None,/texts=[texts[i] for i in val_indices],/' experiments/utils/precompute/precompute_pipeline.py
# Gated repos 403 with our token: ungated mirrors. NousResearch/Meta-Llama-3-8B is sha256-identical to meta-llama;
# unsloth/gemma-2-2b is a bf16 export (ILSE loads bf16 anyway) - disclosed in the write-up.
python3 - <<'PY'
p = "experiments/utils/model_definitions/text_automodel_wrapper.py"
s = open(p).read()
s = s.replace('return "google/gemma-2-2b"', 'return "unsloth/gemma-2-2b"', 1).replace('f"meta-llama/Meta-Llama-3-8B"', '"NousResearch/Meta-Llama-3-8B"', 1)
# MoE extension (ours, not in ILSE): Qwen3.6-35B-A3B, a multimodal checkpoint, so layer counts live in text_config and the
# text stack is model.model.language_model
s = s.replace('"Gemma2"]', '"Gemma2", "Qwen36MoE"]', 1).replace("'Gemma2': Gemma2_sizes,", "'Gemma2': Gemma2_sizes, 'Qwen36MoE': ['35B-A3B'],", 1)
s = s.replace('def get_model_path(name, size):', 'def get_model_path(name, size):\n    if name == "Qwen36MoE":\n        return "Qwen/Qwen3.6-35B-A3B"', 1)
s = s.replace("self.num_layers = self.config.num_hidden_layers + 1", "tc = getattr(self.config, 'text_config', self.config)\n        self.num_layers = tc.num_hidden_layers + 1", 1)
s = s.replace("self.config.num_hidden_layers = 1 ", "tc.num_hidden_layers = 1 ", 1).replace("self.config.num_hidden_layers = self.evaluation_layer_idx", "tc.num_hidden_layers = self.evaluation_layer_idx", 1)
s = s.replace("        model = self._get_model_with_forward_pass()\n", "        model = self._get_model_with_forward_pass()\n        if hasattr(getattr(model, 'model', None), 'language_model'):  # comp_gnn: Qwen3.6 text stack\n            model = __import__('types').SimpleNamespace(model=model.model.language_model)\n", 1)
assert s.count("tc.num_hidden_layers") == 3 and "Qwen3.6 text stack" in s and "'Qwen36MoE': ['35B-A3B']" in s and 'return "Qwen/Qwen3.6-35B-A3B"' in s
assert "NousResearch/Meta-Llama-3-8B" in s and "unsloth/gemma-2-2b" in s
open(p, "w").write(s)
PY
# ILSE_MAX_N (Qwen3.6 time budget, disclosed): every split of every task is a fixed seed-0 random subset of at most N examples
cat >> experiments/utils/model_definitions/gnn/gnn_datasets.py <<'PY2'


_load_task_data_full = load_task_data


def load_task_data(task_name, split="train"):  # comp_gnn: same splits, each capped to a seeded subset of ILSE_MAX_N
    d, cap = _load_task_data_full(task_name, split), int(__import__("os").environ.get("ILSE_MAX_N", 0))
    n = len(next(d[k] for k in ("labels", "scores", "original_scores", "text", "text_a") if k in d))
    if not cap or n <= cap:
        return d
    keep = np.sort(np.random.default_rng(0).permutation(n)[:cap])
    return {k: ([v[i] for i in keep] if isinstance(v, list) else v[keep]) if hasattr(v, "__len__") and not isinstance(v, str)
            and len(v) == n else v for k, v in d.items()}
PY2
# BLAS/OpenMP may see all host cores inside a container with a smaller CPU quota: cap threads or sklearn LR thrashes
NCPU=$(awk '$1!="max"{print int($1/$2)}' /sys/fs/cgroup/cpu.max 2>/dev/null); NCPU=${NCPU:-$(nproc)}; [ "$NCPU" -ge 1 ] || NCPU=1
export OMP_NUM_THREADS=$NCPU OPENBLAS_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU
export HF_DATASETS_TRUST_REMOTE_CODE=1
python3 -m pip install -q --prefer-binary "mteb<2" torch-geometric "${TRANSFORMERS_SPEC:-transformers>=4.42,<4.50}" "sentence-transformers<4" umap-learn matplotlib accelerate datasets h5py scikit-learn wandb
TASKS=(Banking77Classification EmotionClassification MTOPDomainClassification MTOPIntentClassification PoemSentimentClassification)
MODELS=(Gemma2:2B Llama3:8B)
# Qwen3.6 needs transformers 5 -> sentence-transformers 5; mteb 1.x still imports the removed AutoModelForVision2Seq.
# Call right after picking the model:  if [ "$FAM" = Qwen36MoE ]; then tf5; fi
tf5() {
  python3 -m pip install -q --prefer-binary "transformers>=5.5" sentence-transformers "peft>=0.18" flash-linear-attention
  sed -i 's/AutoModelForVision2Seq/AutoModelForImageTextToText/g' "$(python3 -m pip show mteb | sed -n 's/^Location: //p')/mteb/models/gme_v_models.py"
}
