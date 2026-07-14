"""Fully synthetic search-and-rescue acquisition substrate.

Every person, place, utterance, and score in this file is fictional. The corpus
exists only to demonstrate how a linguistic evidence stream binds to AST.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import hashlib

import numpy as np

from substrate_binding import SubstrateBinding

HIKER_DIMENSIONS = (
    "location",
    "time",
    "description",
    "equipment",
    "medical",
    "weather",
    "companions",
    "intentions",
)

HIKER_IMPORTANCE = {
    "location": 0.9,
    "time": 0.8,
    "description": 0.7,
    "equipment": 0.6,
    "medical": 0.8,
    "weather": 0.5,
    "companions": 0.6,
    "intentions": 0.7,
}

HIKER_DEPENDENCIES = {
    "intentions": ("location",),
    "companions": ("description",),
}


@dataclass(frozen=True)
class HikerEvidence:
    evidence_id: str
    target: str
    question: str
    answer: str
    scores: Mapping[str, float]
    condition: str
    stall_gt: bool = False

    @property
    def text(self) -> str:
        return f"Question: {self.question}\nAnswer: {self.answer}"


def _e(
    evidence_id: str,
    target: str,
    question: str,
    answer: str,
    scores: Mapping[str, float],
    condition: str,
    stall_gt: bool = False,
) -> HikerEvidence:
    return HikerEvidence(
        evidence_id=evidence_id,
        target=target,
        question=question,
        answer=answer,
        scores=dict(scores),
        condition=condition,
        stall_gt=stall_gt,
    )


_COMMON_PREFIX = [
    _e(
        "h01_free_recall",
        "location",
        "Tell me everything you remember about the missing hiker.",
        "Alex planned a day hike on Cedar Loop yesterday. They wore an orange shell and carried a small grey pack.",
        {"location": 0.35, "time": 0.25, "description": 0.35, "equipment": 0.20, "intentions": 0.30},
        "shared",
    ),
    _e(
        "h02_location",
        "location",
        "Where was Alex last seen?",
        "At the east Cedar Loop trailhead, beside the map board near the lower car park.",
        {"location": 0.78},
        "shared",
    ),
    _e(
        "h03_time",
        "time",
        "When was the last confirmed contact?",
        "A message arrived at 14:18 yesterday, followed by a trail-map photo at 14:21.",
        {"time": 0.88},
        "shared",
    ),
    _e(
        "h04_description",
        "description",
        "What was Alex wearing?",
        "An orange waterproof shell, charcoal trousers, a navy cap, and rectangular glasses.",
        {"description": 0.84},
        "shared",
    ),
]

EFFICIENT = _COMMON_PREFIX + [
    _e(
        "h05_equipment",
        "equipment",
        "What equipment did Alex carry?",
        "A grey day pack, water, a head torch, a paper map, and two trekking poles.",
        {"equipment": 0.86},
        "efficient",
    ),
    _e(
        "h06_medical",
        "medical",
        "Is there any relevant health information?",
        "No known medical condition was reported, and no regular medication was expected.",
        {"medical": 0.82},
        "efficient",
    ),
    _e(
        "h07_weather",
        "weather",
        "What conditions were expected on the route?",
        "Dry at departure, with low cloud and light rain forecast for the ridge after 17:00.",
        {"weather": 0.80},
        "efficient",
    ),
    _e(
        "h08_companions",
        "companions",
        "Was anyone hiking with Alex?",
        "No. The plan and the trailhead photograph both indicate a solo hike.",
        {"companions": 0.86},
        "efficient",
    ),
    _e(
        "h09_intentions",
        "intentions",
        "Which route did Alex intend to follow?",
        "The intended route was the clockwise Cedar Loop, taking the ridge fork and returning before dusk.",
        {"intentions": 0.90, "location": 0.88},
        "efficient",
    ),
]

_AGENT_REPEAT = _e(
    "h_stall_location_repeat",
    "location",
    "Where exactly was Alex last seen?",
    "At the east Cedar Loop trailhead, beside the map board near the lower car park.",
    {"location": 0.78},
    "agent_repetition",
    True,
)
AGENT_REPETITION = _COMMON_PREFIX + [_AGENT_REPEAT] * 5

INTERVIEWEE_DEGRADATION = _COMMON_PREFIX + [
    _e(
        "h_stall_equipment_1",
        "equipment",
        "What else was in the pack?",
        "I only remember that it was a small grey pack; I cannot add any reliable item.",
        {"equipment": 0.20},
        "interviewee_degradation",
        # NOT labelled a stall: this exchange resolves equipment from 0.00 to 0.20.
        # It produces gain, so by AST's own definition it is not a no-gain stall.
        # The stall begins at the first exchange that adds nothing.
        False,
    ),
    _e(
        "h_stall_equipment_2",
        "equipment",
        "Could you think again about the equipment?",
        "No, only the same small grey pack. I do not remember any additional equipment.",
        {"equipment": 0.20},
        "interviewee_degradation",
        True,
    ),
    _e(
        "h_stall_equipment_3",
        "equipment",
        "Was there any other useful equipment?",
        "Nothing else comes back to me beyond the grey day pack.",
        {"equipment": 0.20},
        "interviewee_degradation",
        True,
    ),
    _e(
        "h_stall_equipment_4",
        "equipment",
        "Can you provide one new equipment detail?",
        "I cannot provide a new detail. I only remember the grey pack.",
        {"equipment": 0.20},
        "interviewee_degradation",
        True,
    ),
]

SCENARIOS = {
    "efficient": EFFICIENT,
    "agent_repetition": AGENT_REPETITION,
    "interviewee_degradation": INTERVIEWEE_DEGRADATION,
}


def iter_all_evidence() -> Iterable[HikerEvidence]:
    seen: set[str] = set()
    for scenario in SCENARIOS.values():
        for item in scenario:
            if item.evidence_id not in seen:
                seen.add(item.evidence_id)
                yield item


def hash_text_embedding(text: str, dimension: int = 96) -> np.ndarray:
    """Deterministic model-free embedding used only by tests and smoke runs."""

    clean = " ".join(text.lower().replace("\n", " ").split())
    tokens = clean.split()
    features = tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]
    vector = np.zeros(dimension, dtype=np.float64)
    for feature in features:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimension
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        vector[0] = 1.0
        norm = 1.0
    return vector / norm


def hash_evidence_embedding(evidence: HikerEvidence, dimension: int = 96) -> np.ndarray:
    """Hash-based smoke-test vector; never the default experiment representation."""

    lexical = hash_text_embedding(evidence.text, dimension=dimension)
    anchor = np.zeros(dimension, dtype=np.float64)
    anchor[HIKER_DIMENSIONS.index(evidence.target)] = 1.0
    vector = 0.82 * anchor + 0.18 * lexical
    return vector / np.linalg.norm(vector)


class HikerEmbeddingTable:
    """Safe NPZ-backed lookup for precomputed Qwen3 evidence vectors."""

    DEFAULT_PATH = Path(__file__).parent / "data" / "hiker_embeddings_qwen3.npz"

    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path else self.DEFAULT_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Qwen3 embedding archive not found at {path}. "
                "Start Ollama, pull qwen3-embedding, and run "
                "`python precompute_embeddings.py`, or use "
                "`--hash-embeddings` only for a smoke test."
            )
        archive = np.load(path, allow_pickle=False)
        vectors = {
            key: np.asarray(archive[key], dtype=np.float64).reshape(-1)
            for key in archive.files
        }
        if not vectors:
            raise ValueError(f"embedding archive is empty: {path}")
        dimensions = {vector.size for vector in vectors.values()}
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent embedding dimensions in {path}: {sorted(dimensions)}")
        for key, vector in vectors.items():
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"non-finite embedding for {key!r}")
            norm = float(np.linalg.norm(vector))
            if norm <= 1e-12:
                raise ValueError(f"zero-norm embedding for {key!r}")
            vectors[key] = vector / norm
        self._vectors = vectors

    def __call__(self, evidence: HikerEvidence) -> np.ndarray:
        try:
            return self._vectors[evidence.evidence_id]
        except KeyError as exc:
            raise KeyError(f"missing precomputed embedding for {evidence.evidence_id!r}") from exc


def make_hiker_binding(
    embeddings_path: str | Path | None = None,
    *,
    use_hash_embeddings: bool = False,
) -> SubstrateBinding:
    embedder = (
        (lambda evidence: hash_evidence_embedding(evidence))
        if use_hash_embeddings
        else HikerEmbeddingTable(embeddings_path)
    )
    return SubstrateBinding(
        name=("synthetic-sar-hiker-hash-smoke" if use_hash_embeddings else "synthetic-sar-hiker-qwen3"),
        schema=HIKER_DIMENSIONS,
        importance=HIKER_IMPORTANCE,
        dependencies=HIKER_DEPENDENCIES,
        scorer=lambda evidence: evidence.scores,
        embedder=embedder,
        target_selector=lambda evidence: evidence.target,
        evidence_ref=lambda evidence: {
            "id": evidence.evidence_id,
            "question": evidence.question,
            "answer": evidence.answer,
        },
        ground_truth=lambda evidence: {
            "condition": evidence.condition,
            "stall": evidence.stall_gt,
        },
    )
