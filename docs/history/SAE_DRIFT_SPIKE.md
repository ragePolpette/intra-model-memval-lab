# SAE drift spike: verification notes

Status: historical. Moved out of the docstring of `scripts/spikes/sae_drift_spike.py`, which is
a throwaway spike not wired into the consolidation pipeline. The spike dependencies are now
declared in the `[spike]` extra (`pip install -e ".[spike]"`) instead of being installed by hand.

The original notes follow (punctuation normalized to ASCII).

```text
Throwaway spike: do pretrained Gemma Scope 2 SAEs actually work on Gemma 3 270M IT here?

This is NOT production code and is not wired into consolidation/ or any promote/rollback
gate. It answers one question only: can we load a real Gemma Scope 2 SAE, run it against
real Gemma 3 270M IT activations in this environment, and get back sane shapes / sparsity /
reconstruction numbers. No drift metric, no pre/post comparison, no TraceArtifact wiring -
that is the next step, once this sanity check is green.

Extra, local-only dependencies for this spike (NOT added to pyproject.toml):

    pip install -e ".[train]"   # torch, transformers, peft, accelerate (already in the repo)
    pip install sae-lens        # pulls in transformer_lens, einops, etc.

Verified against sae-lens 6.49.1 / transformer-lens 3.8.0 installed fresh for this spike
(2026-08-26) by installing the package and reading its source directly, not from memory:

- `sae_lens.SAE.from_pretrained(release, sae_id, device=..., dtype=...)` is the current
  entry point (`sae_lens/saes/sae.py`); the pre-6.x `SAE.from_pretrained` that also returned
  `(cfg_dict, sparsity)` is now `SAE.from_pretrained_with_cfg_and_sparsity`.
- The exact release/sae_id names come straight from the packaged
  `sae_lens/pretrained_saes.yaml` (searched for "270m-it"), not guessed:
  release "gemma-scope-2-270m-it-res", repo_id "google/gemma-scope-2-270m-it",
  model "google/gemma-3-270m-it". `-res` releases only expose 4 layers: 5, 9, 12, 15.
- `sae_lens/loading/pretrained_sae_loaders.py::get_gemma_3_config_from_hf` maps a
  "resid_post" sae_id to the *HF-native* hook point "model.layers.{layer}.output" (this is
  the hook_name used by SAELens' own HuggingFace-hook conversion path, distinct from the
  TransformerLens-style "blocks.{layer}.hook_resid_post" name also stored in
  `sae.cfg.metadata.hook_name`). We are not using TransformerLens/HookedSAETransformer
  here (per the repo's existing HFTextGenerator pattern), so we hook the HF module directly.
- `transformers.models.gemma3.modeling_gemma3.Gemma3DecoderLayer.forward` returns a single
  `hidden_states` tensor (not a tuple), so a plain `register_forward_hook` on
  `model.model.layers[i]` captures exactly that resid_post activation.

Layer/site choice: residual stream ("-res" release), layer 9, width 16k, l0_medium.
- Residual stream over MLP/attention-output/transcoder sites: it is the cumulative
  representation state after the whole layer, which is what a downstream drift check
  (pre vs. post LoRA consolidation) would want to compare against fixed probe prompts --
  matches the "residual stream of an intermediate layer" starting point named in the task.
- Layer 9 is the middle option among the four layers this release actually ships (5, 9, 12,
  15) -- not the earliest (still close to token/lexical features) and not the latest
  (closest to the output head).
- Width 16k is the smallest/fastest-to-download width available for this release (vs. 65k /
  262k), appropriate for a same-day sanity spike, not a width chosen for research-grade
  feature resolution.
- l0_medium targets a mid-range sparsity (~60 active features/token per the release's own
  metadata for layer 9 width 16k), avoiding the very sparse "small" (~20) and very dense
  "big" (~150) variants.

Usage:
    python scripts/spikes/sae_drift_spike.py [--device cpu|cuda] [--layer 9]
                                              [--sae-release ...] [--sae-id ...]
```
