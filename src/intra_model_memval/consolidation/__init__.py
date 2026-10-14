"""Consolidation pipeline: lesson -> synthetic dataset -> LoRA with replay -> verdict."""

from .config import ConsolidationConfig
from .dataset import (
    DATASET_MEDIA_TYPE,
    DatasetCompositionReport,
    HoldoutSplit,
    VerifierGateResult,
    apply_verifier_gate,
    check_dataset_composition,
    dataset_to_jsonl_bytes,
    examples_from_jsonl_bytes,
    register_dataset,
    split_holdout,
)
from .gating import GateThresholds, PromotionDecision, decide_promotion
from .interfaces import (
    ConflictChecker,
    LessonExtractor,
    LoraTrainerAdapter,
    VariationGenerator,
    VariationVerifier,
)
from .pipeline import PLANNER_MODE, TRAINING_MODE, ConsolidationPlanner
from .replay import ReplayBuffer, build_replay_buffer
from .training import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    GemmaLoraTrainer,
    HFTextGenerator,
    buffer_item_to_messages,
    release_accelerator_memory,
    resolve_train_config,
)

__all__ = [
    "DATASET_MEDIA_TYPE",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "PLANNER_MODE",
    "TRAINING_MODE",
    "ConflictChecker",
    "ConsolidationConfig",
    "ConsolidationPlanner",
    "DatasetCompositionReport",
    "GateThresholds",
    "GemmaLoraTrainer",
    "HFTextGenerator",
    "HoldoutSplit",
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
    "release_accelerator_memory",
    "resolve_train_config",
    "split_holdout",
]
