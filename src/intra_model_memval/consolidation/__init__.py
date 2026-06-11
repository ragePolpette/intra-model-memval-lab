"""Consolidation pipeline: lesson -> synthetic dataset -> LoRA with replay -> verdict."""

from .config import ConsolidationConfig
from .dataset import (
    DATASET_MEDIA_TYPE,
    DatasetCompositionReport,
    VerifierGateResult,
    apply_verifier_gate,
    check_dataset_composition,
    dataset_to_jsonl_bytes,
    examples_from_jsonl_bytes,
    register_dataset,
)
from .gating import GateThresholds, PromotionDecision, decide_promotion
from .interfaces import (
    ConflictChecker,
    LessonExtractor,
    LoraTrainerAdapter,
    VariationGenerator,
    VariationVerifier,
)
from .pipeline import ConsolidationPlanner
from .replay import ReplayBuffer, build_replay_buffer
from .training import (
    DEFAULT_BASE_MODEL,
    GemmaLoraTrainer,
    HFTextGenerator,
    buffer_item_to_messages,
    resolve_train_config,
)

__all__ = [
    "DATASET_MEDIA_TYPE",
    "DEFAULT_BASE_MODEL",
    "ConflictChecker",
    "ConsolidationConfig",
    "ConsolidationPlanner",
    "DatasetCompositionReport",
    "GateThresholds",
    "GemmaLoraTrainer",
    "HFTextGenerator",
    "LessonExtractor",
    "LoraTrainerAdapter",
    "PromotionDecision",
    "ReplayBuffer",
    "VariationGenerator",
    "VariationVerifier",
    "VerifierGateResult",
    "apply_verifier_gate",
    "buffer_item_to_messages",
    "build_replay_buffer",
    "check_dataset_composition",
    "dataset_to_jsonl_bytes",
    "decide_promotion",
    "examples_from_jsonl_bytes",
    "register_dataset",
    "resolve_train_config",
]
