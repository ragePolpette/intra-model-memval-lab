"""Throwaway spike: do pretrained Gemma Scope 2 SAEs actually work on Gemma 3 270M IT here?

Not production code and not wired into consolidation/ or any promote/rollback gate. It loads
a Gemma Scope 2 SAE, runs it on real Gemma 3 270M IT activations and prints shapes, sparsity
and reconstruction error. Verification notes and the layer/site choice are recorded in
docs/history/SAE_DRIFT_SPIKE.md.

    pip install -e ".[spike]"
    python scripts/spikes/sae_drift_spike.py [--device cpu|cuda] [--layer 9]
                                              [--sae-release ...] [--sae-id ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intra_model_memval.consolidation.training import DEFAULT_BASE_MODEL  # noqa: E402

SAE_RELEASE = "gemma-scope-2-270m-it-res"
SAE_ID = "layer_9_width_16k_l0_medium"
LAYER_INDEX = 9

PROMPTS = [
    "What is the capital of France?",
    "Explain in one short sentence why the sky is blue.",
    "Write a one-line greeting for a customer support chat.",
]


def load_model_and_tokenizer(base_model_id: str, device: str | None):
    """Same loading pattern as HFTextGenerator in consolidation/training.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )
    model = model.to(device).eval()
    return model, tokenizer, device


def load_sae(release: str, sae_id: str, device: str):
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae.eval()


def capture_layer_output(model, layer_index: int):
    """Hook model.model.layers[layer_index] and return (handle, captured).

    captured[0] holds the tensor from the most recent forward pass. See the
    module docstring for why this HF-side hook corresponds to the SAE's
    expected "resid_post" input.
    """
    captured: list = [None]

    def _hook(_module, _inputs, output):
        captured[0] = output.detach()

    handle = model.model.layers[layer_index].register_forward_hook(_hook)
    return handle, captured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sae-release", default=SAE_RELEASE)
    parser.add_argument("--sae-id", default=SAE_ID)
    parser.add_argument("--layer", type=int, default=LAYER_INDEX)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import torch

    print(f"[load] base model: {args.base_model}")
    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.device)
    print(f"[load] device: {device}")

    print(f"[load] SAE release={args.sae_release} id={args.sae_id}")
    sae = load_sae(args.sae_release, args.sae_id, device)
    print(f"[sae] hook_name (TransformerLens convention): {sae.cfg.metadata.hook_name}")
    print(f"[sae] d_in={sae.cfg.d_in} d_sae={sae.cfg.d_sae}")

    handle, captured = capture_layer_output(model, args.layer)
    try:
        for prompt in PROMPTS:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                model(input_ids)

            activations = captured[0]
            if activations is None:
                raise RuntimeError("hook did not capture any activation")
            activations = activations.to(sae.dtype)

            with torch.no_grad():
                feature_acts = sae.encode(activations)
                reconstruction = sae.decode(feature_acts)

            l0_per_token = (feature_acts > 0).sum(dim=-1).float()
            squared_error = (activations - reconstruction).pow(2).sum(dim=-1)
            input_energy = activations.pow(2).sum(dim=-1).clamp_min(1e-8)
            normalized_error = squared_error / input_energy

            print(f"\n[prompt] {prompt!r}")
            print(f"  activations shape:  {tuple(activations.shape)}")
            print(f"  feature_acts shape: {tuple(feature_acts.shape)}")
            print(f"  reconstruction shape: {tuple(reconstruction.shape)}")
            print(
                f"  L0 (active features/token, of {sae.cfg.d_sae}): "
                f"mean={l0_per_token.mean().item():.1f} "
                f"min={l0_per_token.min().item():.0f} "
                f"max={l0_per_token.max().item():.0f}"
            )
            print(f"  reconstruction MSE (raw, mean over tokens): {squared_error.mean().item():.6f}")
            print(
                "  reconstruction error (fraction of input energy, mean over tokens): "
                f"{normalized_error.mean().item():.4f}"
            )
    finally:
        handle.remove()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
