"""Real LoRA training and generation on Gemma 3 270M IT (requires the `train` extra).

torch/transformers/peft are imported lazily inside the classes so the core
package keeps working without ML dependencies. The data-preparation helpers are
pure and tested without them.

POC simplifications, on purpose:
- full-sequence LM loss on the chat-formatted text (no completion-only masking)
- single adapter per run; stacking on a previous adapter is not handled yet
"""

from __future__ import annotations

import gc
import io
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..domain import ArtifactRef, ConsolidationCandidate, SyntheticExample
from ..persistence import ExperimentStore

DEFAULT_BASE_MODEL = "google/gemma-3-270m-it"
DEFAULT_MAX_NEW_TOKENS = 256

TRAIN_DEFAULTS: dict[str, Any] = {
    "rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "epochs": 3,
    "batch_size": 8,
    "max_length": 1024,
    "seed": 42,
}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def buffer_item_to_messages(item: Any) -> list[dict[str, str]]:
    """Map a replay-buffer item (SyntheticExample or {"input","target"} dict) to chat messages."""
    if isinstance(item, SyntheticExample):
        user, assistant = item.input, item.target
    elif isinstance(item, Mapping) and "input" in item and "target" in item:
        user, assistant = str(item["input"]), str(item["target"])
    else:
        raise TypeError(f"unsupported buffer item type: {type(item).__name__}")
    if not user.strip() or not assistant.strip():
        raise ValueError("buffer item input and target must not be empty")
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def resolve_train_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(TRAIN_DEFAULTS)
    resolved.update({key: value for key, value in train_config.items() if value is not None})
    return resolved


def zip_directory_bytes(directory: Path) -> bytes:
    buffer = io.BytesIO()
    base = Path(directory)
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(base))
    return buffer.getvalue()


def release_accelerator_memory() -> None:
    """Free cached CUDA memory after a model was dropped; a no-op without torch or CUDA."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GemmaLoraTrainer:
    """LoraTrainerAdapter implementation: trains an adapter and registers it as artifact.

    The returned ArtifactRef points to a zip of the adapter directory; the
    unzipped directory path is kept in `metadata["adapter_dir"]` for immediate
    post-training evaluation on the same machine.
    """

    def __init__(
        self,
        store: ExperimentStore,
        *,
        base_model_id: str = DEFAULT_BASE_MODEL,
        output_dir: Path | str = "adapters",
        device: str | None = None,
        revision: str | None = None,
    ):
        self.store = store
        self.base_model_id = base_model_id
        self.output_dir = Path(output_dir)
        self.device = device
        self.revision = revision

    def train(self, candidate: ConsolidationCandidate, buffer: Sequence[Any]) -> ArtifactRef:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        if not buffer:
            raise ValueError("replay buffer must not be empty")
        config = resolve_train_config(candidate.train_config)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        torch.manual_seed(int(config["seed"]))

        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, revision=self.revision)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            revision=self.revision,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        resolved_revision = getattr(model.config, "_commit_hash", None)
        lora_config = LoraConfig(
            r=int(config["rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        texts = [
            tokenizer.apply_chat_template(
                buffer_item_to_messages(item), tokenize=False, add_generation_prompt=False
            )
            for item in buffer
        ]
        encodings = tokenizer(texts, truncation=True, max_length=int(config["max_length"]))
        dataset = [
            {"input_ids": input_ids, "attention_mask": attention_mask}
            for input_ids, attention_mask in zip(
                encodings["input_ids"], encodings["attention_mask"]
            )
        ]

        adapter_dir = self.output_dir / candidate.consolidation_candidate_id
        adapter_dir.mkdir(parents=True, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=str(adapter_dir / "trainer-state"),
            num_train_epochs=float(config["epochs"]),
            per_device_train_batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            seed=int(config["seed"]),
            bf16=use_bf16,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        result = trainer.train()
        model.save_pretrained(str(adapter_dir))

        ref = self.store.register_artifact_bytes(
            zip_directory_bytes(adapter_dir), media_type="application/zip", suffix=".zip"
        )
        return ref.model_copy(
            update={
                "role": "lora-adapter",
                "metadata": {
                    "adapter_dir": str(adapter_dir),
                    "base_model_id": self.base_model_id,
                    "base_model_revision": resolved_revision,
                    "train_config": config,
                    "train_loss": float(result.training_loss),
                    "buffer_size": len(buffer),
                    "device": device,
                },
            }
        )


class HFTextGenerator:
    """TextGenerator backed by transformers, with an optional LoRA adapter on top."""

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL,
        *,
        adapter_dir: Path | str | None = None,
        device: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        revision: str | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=revision,
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_dir))
        self.model = model.to(self.device).eval()

    def generate(self, prompt: str) -> str:
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
