"""Per-component (attention head z, MLP output) features, aligned row-by-row with ILSE's h5 files.

Run from the ILSE-main root:  python -m comp_gnn.components --emb_dir <out>/<Family>_<size>_mean_pooling --task X
Writes <task>/{split}_z.npy (N, nl, H*hd) pre-o_proj head outputs, {split}_mlp.npy (N, nl, D), W_O.npy (nl, D, H*hd),
b_O.npy (nl, D) and meta.json {nheads, nexperts, k}. Head h's residual contribution at block l =
z[:, l, h*hd:(h+1)*hd] @ W_O[l][:, h*hd:(h+1)*hd].T (mean pooling commutes with the linear map).
Gemma2 applies RMSNorm after attention/MLP: post_attn_norm(o) = o / rms(o) * (1 + w) with rms per token, so we store
z / rms(o) per token before pooling and fold (1 + w) into W_O (exact); the MLP feature is the post-FF-norm output.
Gemma4: same fold with w instead of 1 + w; heads (256- or 512-wide by layer) are zero-padded to the widest in z and W_O;
the per-layer-embedding (PLE) residual term is folded into the MLP node; layer_scalar s (r_{l+1} = s * (r_l + block))
is folded into W_O and the MLP feature, so they are each component's contribution to r_{l+1}.
MoE (Qwen3.6, Qwen3_5MoeSparseMoeBlock): {split}_gate.npy (N, nl, E) token-mean router weights g_e (0 when unrouted) and
{split}_exp.npy (N, nl, E*K) token-mean of a fixed JL projection (seed = layer, N(0, 1/K) entries, K=32) of each expert's
gated output g_e * E_e(x); the projection commutes with mean pooling. The MLP feature is the gated shared expert only, so
no aggregate node duplicates the experts (shared + routed == block output is asserted). Qwen3.6 DeltaNet layers have no
o_proj: their z is the linear_attn.out_proj input (32 value heads x 128, so each 256-wide slot = the 2 value heads that
share one key head); full-attention layers' z already includes the sigmoid output gate.
"""
import argparse
import json
import os
import types

import numpy as np
import torch

from experiments.utils.model_definitions.text_automodel_wrapper import TextModelSpecifications, TextLayerwiseAutoModelWrapper
from experiments.utils.precompute.h5_utils import load_embeddings_from_h5

CHUNK = 5000  # same chunking as precompute_pipeline, so padding (global per chunk) matches the h5
K = 32  # JL width per expert


def parts(model):
    """(layers, o_proj(layer), mlp_out_module(layer), post_attn_norm(layer) or None) for Pythia / Llama3 / Gemma2/4."""
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers, lambda l: l.attention.dense, lambda l: l.mlp, lambda l: None
    base = getattr(model.model, "language_model", model.model)  # Gemma4 is a multimodal wrapper
    gemma = hasattr(base.layers[0], "post_feedforward_layernorm")
    return (base.layers, lambda l: l.self_attn.o_proj if hasattr(l, "self_attn") else l.linear_attn.out_proj,  # Qwen3.6
            (lambda l: l.post_feedforward_layernorm) if gemma else (lambda l: l.mlp),
            (lambda l: l.post_attention_layernorm) if gemma else (lambda l: None))


def geom(model):
    """(nheads, padded head dim)."""
    layers, oproj = parts(model)[:2]
    H = getattr(model.config, "text_config", model.config).num_attention_heads
    return H, max(oproj(l).in_features for l in layers) // H


def pad(x, H, hd):  # (..., H*h) -> (..., H*hd), each head zero-padded
    x = x.unflatten(-1, (H, -1))
    return torch.nn.functional.pad(x, (0, hd - x.shape[-1])).flatten(-2)


def lm_only(t): return rf".*language_model.*\.({'|'.join(t)})"  # peft regex: text decoder only
def nscale(n):  # Gemma4RMSNorm scales by w, Gemma2RMSNorm by 1 + w
    return n.weight if hasattr(n, "with_scale") else 1 + n.weight


def scalars(model):
    return torch.tensor([float(getattr(l, "layer_scalar", 1.0)) for l in parts(model)[0]])


def hook(model, sink):
    """Registers per-token hooks calling sink(key, x) for z (normed, padded), mlp and Gemma4's ple. Returns handles."""
    layers, oproj, mlpmod, anorm = parts(model)
    H, hd = geom(model)

    def zhook(norm):
        def f(m, inp, out):
            x = inp[0].float()
            if norm is not None:  # per-token 1/rms(o), same eps as the norm; the scale goes into W_O
                x = x * torch.rsqrt(out.float().pow(2).mean(-1, keepdim=True) + norm.eps)
            sink("z", pad(x, H, hd))
        return f

    hs = []
    for l in layers:
        hs.append(oproj(l).register_forward_hook(zhook(anorm(l))))
        # the MoE block returns (out, router_logits)
        hs.append(mlpmod(l).register_forward_hook(lambda m, i, o: sink("mlp", (o[0] if isinstance(o, tuple) else o).float())))
        if hasattr(l, "post_per_layer_input_norm"):
            hs.append(l.post_per_layer_input_norm.register_forward_hook(lambda m, i, o: sink("ple", o.float())))
    return hs


def finish(t, S):
    """t: stacked (..., nl, D) features. Folds Gemma4's PLE term into the MLP node and scales it by layer_scalar S."""
    if "ple" in t:  # ponytail: PLE has no node of its own (disclosed); add one if it matters
        t["mlp"] = t["mlp"] + t.pop("ple")
    t["mlp"] = t["mlp"] * S.to(t["mlp"].device)[:, None]
    return t


def is_moe(model):
    return hasattr(parts(model)[0][0], "mlp") and hasattr(parts(model)[0][0].mlp, "experts")


def moe_forward(blk, l, pool, cur):
    """Qwen3_5MoeSparseMoeBlock.forward (transformers 5.x, same op order) that also pools router gates, JL-projected gated
    expert outputs, the gated shared expert and the routed sum into cur["gate"/"exp"/"shared"/"routed"]. The first call is
    checked against the original forward."""
    orig, P = blk.forward, {}

    def fwd(self, h):
        B, S, D = h.shape
        x = h.reshape(-1, D)
        w = torch.softmax(torch.nn.functional.linear(x, self.gate.weight), -1, dtype=torch.float)
        tw, ti = torch.topk(w, self.gate.top_k, -1)
        tw = (tw / tw.sum(-1, keepdim=True)).to(x.dtype)
        E, ex = self.gate.num_experts, self.experts
        if x.device not in P:
            P[x.device] = (torch.randn(D, K, generator=torch.Generator().manual_seed(l)) / K ** 0.5).to(x.device)
        out = torch.zeros_like(x)
        G = torch.zeros(len(x), E, device=x.device).scatter_(1, ti, tw.float())
        proj = torch.zeros(len(x), E, K, device=x.device)
        mask = torch.nn.functional.one_hot(ti, E).permute(2, 1, 0)
        for e in mask.sum((1, 2)).nonzero()[:, 0].tolist():
            k, t = torch.where(mask[e])
            g, u = torch.nn.functional.linear(x[t], ex.gate_up_proj[e]).chunk(2, -1)
            y = torch.nn.functional.linear(ex.act_fn(g) * u, ex.down_proj[e]) * tw[t, k, None]
            out.index_add_(0, t, y.to(x.dtype))
            proj[t, e] = y.float() @ P[x.device]
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        full = (out + shared).view(B, S, D)
        if not cur.get("checked"):
            ref = orig(h)
            assert (ref - full).abs().max() <= 1e-2 * ref.abs().max(), "MoE reimplementation differs"
            cur["checked"] = True
        cur["gate"].append(pool(G.view(B, S, E)))
        cur["exp"].append(pool(proj.view(B, S, E * K)))
        cur["shared"].append(pool(shared.view(B, S, D).float()))
        cur["routed"].append(pool(out.view(B, S, D).float()))
        return full
    blk.forward = types.MethodType(fwd, blk)
    return lambda: setattr(blk, "forward", orig)


def run(wrapper, texts, batch_size):
    """Returns ({"z", "mlp"[, "gate", "exp"]}: (N, nl, d) fp16 arrays, layerwise residuals (N, L, D))."""
    layers, S = parts(wrapper.model)[0], scalars(wrapper.model)
    keys = ["z", "mlp"] + (["gate", "exp", "shared", "routed"] if is_moe(wrapper.model) else [])
    raw = keys + (["ple"] if hasattr(layers[0], "post_per_layer_input_norm") else [])
    feats, cur = {k: [] for k in keys}, {k: [] for k in raw}

    def pool(x):
        return wrapper._get_pooled_hidden_states(x, wrapper._current_attention_mask, "mean").float().cpu()

    hooks = hook(wrapper.model, lambda k, x: cur[k].append(pool(x)))
    restore = [moe_forward(layer.mlp, i, pool, cur) for i, layer in enumerate(layers)] if "exp" in keys else []
    layerwise = []
    try:
        for s in range(0, len(texts), CHUNK):
            for k in raw:
                cur[k] = []
            _, _, lw = wrapper.encode(texts[s:s + CHUNK], return_raw_hidden_states=True, pooling_method="mean",
                                      batch_size=batch_size, verbose=False)
            lw = lw if lw.ndim == 3 else lw[:, None]
            layerwise.append(np.transpose(lw, (1, 0, 2)))
            nl = len(layers)
            # hooks fire layer-by-layer per batch: regroup to (N, nl, d)
            t = finish({k: torch.cat([torch.stack(cur[k][i:i + nl], 1) for i in range(0, len(cur[k]), nl)]) for k in raw}, S)
            for key in keys:
                feats[key].append(t[key].numpy().astype(np.float16))
    finally:
        for h in hooks:
            h.remove()
        for r in restore:
            r()
    return {k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(layerwise)


def weights(model):
    layers, oproj, _, anorm = parts(model)
    (H, hd), S = geom(model), scalars(model)
    W, b = [], []
    for l, s in zip(layers, S):
        w = oproj(l).weight.detach().float().cpu()
        n = anorm(l)
        W.append(s * pad(w if n is None else nscale(n).detach().float().cpu()[:, None] * w, H, hd))
        bias = oproj(l).bias
        b.append(s * (torch.zeros(w.shape[0]) if bias is None else bias.detach().float().cpu()))
    return torch.stack(W), torch.stack(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model_family", default="Pythia")
    ap.add_argument("--model_size", default="410m")
    ap.add_argument("--batch_size", type=int, default=256)
    a = ap.parse_args()
    wrapper = TextLayerwiseAutoModelWrapper(TextModelSpecifications(a.model_family, a.model_size, "main", ignore_checks=True),
                                            device_map="auto", evaluation_layer_idx=-1, use_memory_efficient_hooks=True)
    tdir = os.path.join(a.emb_dir, a.task)
    assert os.path.exists(os.path.join(tdir, "train.h5")), f"precompute produced no h5 for {a.task}"
    W, b = weights(wrapper.model)
    nheads, S = geom(wrapper.model)[0], scalars(wrapper.model)
    for split in ("train", "validation", "test"):
        path = os.path.join(tdir, f"{split}.h5")
        if not os.path.exists(path):
            continue
        emb, _, texts, _ = load_embeddings_from_h5(path, load_texts=True, load_metadata=True)
        texts = [t.decode() if isinstance(t, bytes) else str(t) for t in texts]
        f, lw = run(wrapper, texts, a.batch_size)
        z, mlp = f["z"], f["mlp"]
        # alignment check: our recomputed residual stream must reproduce the stored h5 rows
        err = np.abs(lw - emb).max() / (np.abs(emb).max() + 1e-6)
        # decomposition check: r_{l+1} - r_l == sum_h head_h + mlp + b_O on blocks before the last
        # (the last stored layer is post final-norm). Validates the hooks and the Gemma norm / PLE / layer_scalar folds.
        n, nl = min(64, len(z)), z.shape[1]
        rec = torch.einsum("nlk,ldk->nld", torch.tensor(z[:n]).float(), W) + torch.tensor(mlp[:n]).float() + b
        off = emb.shape[1] - 1 - nl
        delta = torch.tensor(lw[:n, off + 1:off + nl]) - S[:nl - 1, None] * torch.tensor(lw[:n, off:off + nl - 1])
        derr = ((rec[:, :-1] - delta).norm(dim=-1) / (delta.norm(dim=-1) + 1e-6)).median().item()
        print(f"{a.task}/{split}: N={len(texts)} " + " ".join(f"{k}={v.shape}" for k, v in f.items()) +
              f" h5={emb.shape} rel_maxerr={err:.2e} decomp_relerr_median={derr:.2e}", flush=True)
        # bf16 max-abs over N*L*d (Qwen3.6-35B hits 5.0e-2); a row shift gives O(1)
        assert lw.shape == emb.shape and err < 1e-1, "component features are not aligned with the h5 rows"
        # ponytail: loose bound - delta of two bf16-rounded residuals is noisy; a missing norm fold gives O(1) error
        assert derr < 0.15, "heads + MLP do not reconstruct the residual updates"
        if "gate" in f:  # each token's routed gates sum to 1 (norm_topk_prob), so their token mean does too
            assert np.abs(f["gate"].astype(np.float32).sum(-1) - 1).max() < 1e-2, "router gates do not sum to 1"
            blk = f["mlp"].astype(np.float32)
            serr = np.abs(f["shared"].astype(np.float32) + f.pop("routed") - blk).max() / (np.abs(blk).max() + 1e-6)
            assert serr < 1e-2, f"shared + routed experts != MoE block output ({serr:.2e})"
            f["mlp"] = f.pop("shared")  # no aggregate node: the MLP node is the shared branch, experts are their own nodes
        for k, v in f.items():
            np.save(os.path.join(tdir, f"{split}_{k}.npy"), v)
    np.save(os.path.join(tdir, "W_O.npy"), W.numpy().astype(np.float16))
    np.save(os.path.join(tdir, "b_O.npy"), b.numpy())
    with open(os.path.join(tdir, "meta.json"), "w") as f:
        json.dump({"nheads": nheads, "k": K,
                   "nexperts": getattr(wrapper.model.config, "text_config", wrapper.model.config).num_experts
                   if is_moe(wrapper.model) else 0}, f)


if __name__ == "__main__":
    main()
