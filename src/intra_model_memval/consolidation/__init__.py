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

__all__ = [
    "DATASET_MEDIA_TYPE",
    "ConflictChecker",
    "ConsolidationConfig",
    "ConsolidationPlanner",
    "DatasetCompositionReport",
    "GateThresholds",
    "LessonExtractor",
    "LoraTrainerAdapter",
    "PromotionDecision",
    "ReplayBuffer",
    "VariationGenerator",
    "VariationVerifier",
    "VerifierGateResult",
    "apply_verifier_gate",
    "build_replay_buffer",
    "check_dataset_composition",
    "dataset_to_jsonl_bytes",
    "decide_promotion",
    "examples_from_jsonl_bytes",
    "register_dataset",
]
