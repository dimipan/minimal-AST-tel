"""Substrate binding and shared trajectory contract for AST.

A substrate binding is B = (M, Y, S, e):

* M: finite task schema
* Y: evidence space
* S: evidence -> bounded per-schema completeness scores
* e: evidence -> unit-normalised representation

Only the binding is modality-specific. State integration, PE^B, SI^perp, and
calibration are shared by every substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from itertools import chain
import json

import numpy as np

from telemetry_tools_geometric import GeometricTelemetryTool

_MANIFEST_PATH = Path(__file__).with_name("calibration_manifest.json")
_MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))

ScoreMap = Mapping[str, float]
Scorer = Callable[[Any], ScoreMap]
Embedder = Callable[[Any], np.ndarray]
TargetSelector = Callable[[Any], str]
EvidenceRef = Callable[[Any], Any]
GroundTruth = Callable[[Any], Mapping[str, Any]]


@dataclass(frozen=True)
class SubstrateBinding:
    """Declared modality-specific boundary consumed by the frozen monitor."""

    name: str
    schema: tuple[str, ...]
    scorer: Scorer
    embedder: Embedder
    target_selector: TargetSelector
    importance: Mapping[str, float] = field(default_factory=dict)
    dependencies: Mapping[str, Sequence[str]] = field(default_factory=dict)
    evidence_ref: EvidenceRef = lambda evidence: str(evidence)
    ground_truth: GroundTruth = lambda evidence: {}

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("binding name must be non-empty")
        if not self.schema or len(set(self.schema)) != len(self.schema):
            raise ValueError("schema must contain unique dimensions")
        unknown_weights = set(self.importance) - set(self.schema)
        unknown_dependencies = set(self.dependencies) - set(self.schema)
        if unknown_weights or unknown_dependencies:
            raise ValueError("importance/dependencies contain dimensions outside schema")
        for dim, deps in self.dependencies.items():
            bad = set(deps) - set(self.schema)
            if bad:
                raise ValueError(f"dependencies for {dim!r} contain unknown dimensions: {bad}")

    def score(self, evidence: Any) -> dict[str, float]:
        raw = dict(self.scorer(evidence))
        unknown = set(raw) - set(self.schema)
        if unknown:
            raise ValueError(f"scorer emitted dimensions outside schema: {sorted(unknown)}")
        scores = {dim: float(raw.get(dim, 0.0)) for dim in self.schema}
        for dim, value in scores.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"score for {dim!r} must be finite and in [0, 1]")
        return scores

    def embed(self, evidence: Any) -> np.ndarray:
        vector = np.asarray(self.embedder(evidence), dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.all(np.isfinite(vector)):
            raise ValueError("embedding must be a non-empty finite vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("embedding norm must be positive")
        return vector / norm

    def target(self, evidence: Any) -> str:
        target = str(self.target_selector(evidence))
        if target not in self.schema:
            raise ValueError(f"target dimension {target!r} is not in schema")
        return target


class SubstrateKnowledgeState:
    """Monotone acquisition state shared by all AST bindings."""

    def __init__(self, dimensions: Sequence[str], embedding_dim: int):
        self.dimensions = list(dimensions)
        self.embedding_dim = int(embedding_dim)
        self.values = {d: 0.0 for d in self.dimensions}
        self.confidence = {d: 0.0 for d in self.dimensions}
        self.dimension_mentions = {d: 0 for d in self.dimensions}
        self.dimension_successes = {d: 0 for d in self.dimensions}
        self.success_threshold = float(
            _MANIFEST["state_integration"]["success_threshold"]
        )
        self.embeddings = {
            d: np.zeros(self.embedding_dim, dtype=np.float64) for d in self.dimensions
        }
        self.embedding_history = {d: [] for d in self.dimensions}
        self.max_norm_seen = 0.1
        self.response_log = {d: [] for d in self.dimensions}
        self.last_total_gain = 0.0

    def update(
        self,
        evidence_embedding: np.ndarray,
        new_values: Mapping[str, float],
        target_dimension: str,
    ) -> dict[str, float]:
        vector = np.asarray(evidence_embedding, dtype=np.float64).reshape(-1)
        if vector.size != self.embedding_dim:
            raise ValueError(
                f"embedding dimension changed from {self.embedding_dim} to {vector.size}"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            raise ValueError("evidence embedding must have positive norm")
        vector = vector / norm

        old = self.values.copy()
        for dim in self.dimensions:
            proposed = float(new_values.get(dim, 0.0))
            if proposed > self.values[dim]:
                self.values[dim] = proposed
                self.confidence[dim] = min(0.8 + proposed * 0.2, 1.0)

        gains = {dim: max(0.0, self.values[dim] - old[dim]) for dim in self.dimensions}
        self.last_total_gain = float(sum(gains.values()))

        self.dimension_mentions[target_dimension] += 1
        if gains[target_dimension] > self.success_threshold:
            self.dimension_successes[target_dimension] += 1

        self.response_log[target_dimension].append(
            {
                "embedding": vector.copy(),
                "gain": float(gains[target_dimension]),
                "upsilon": float(self.values[target_dimension]),
            }
        )

        for dim, gain in gains.items():
            if gain <= 0.0:
                continue
            self.embedding_history[dim].append(
                {
                    "embedding": self.embeddings[dim].copy(),
                    "norm": float(np.linalg.norm(self.embeddings[dim])),
                    "knowledge_value": old[dim],
                }
            )
            self.embeddings[dim] += vector * gain
            self.max_norm_seen = max(
                self.max_norm_seen, float(np.linalg.norm(self.embeddings[dim]))
            )

        return gains


class TelemetryRunner:
    """Apply the frozen monitor to one binding, exchange by exchange."""

    def __init__(
        self,
        schema: Sequence[str],
        embedding_dim: int,
        importance: Mapping[str, float] | None = None,
        dependencies: Mapping[str, Sequence[str]] | None = None,
    ):
        self.state = SubstrateKnowledgeState(schema, embedding_dim)
        self.importance = dict(importance or {d: 1.0 for d in schema})
        self.dependencies = dict(dependencies or {})
        self.history: list[dict[str, Any]] = []
        si_cfg = _MANIFEST["si_geometric_subspace"]
        # The frozen core reads state.last_total_gain when present, which makes the
        # dampening term aggregate rather than target-only. SubstrateKnowledgeState
        # always sets it, so pin the declared mode and fail loudly if it drifts.
        if si_cfg.get("dampening_gain") != "aggregate_delta_upsilon":
            raise ValueError(
                "calibration_manifest declares dampening_gain="
                f"{si_cfg.get('dampening_gain')!r}, but SubstrateKnowledgeState sets "
                "last_total_gain, which forces aggregate_delta_upsilon."
            )
        self._si_kw = {
            "lambda_damp": float(si_cfg["lambda_damp"]),
            "eta_min": float(si_cfg["eta_min"]),
            "membership_threshold": float(si_cfg["membership_threshold"]),
            "window": si_cfg["window"],
        }

    def step(
        self,
        evidence_embedding: np.ndarray,
        scorer_values: Mapping[str, float],
        target_dim: str,
    ) -> tuple[dict[str, float], float, dict[str, float], dict[str, Any]]:
        """Return (PE, SI, gains, si_diagnostic).

        si_diagnostic carries the per-dimension orthogonal novelty (eta), the
        geometric membership (rho), and the dampening scalar D. Phase-space
        analysis consumes eta (novelty) against delta_upsilon (yield) without
        re-instrumenting the frozen core.
        """
        gains = self.state.update(evidence_embedding, scorer_values, target_dim)
        self.history.append({"target_dimension": target_dim})
        pe = GeometricTelemetryTool.compute_bayesian_PE(
            self.state, self.importance, dependencies=self.dependencies
        )
        diagnostic = GeometricTelemetryTool.compute_geometric_subspace_SI_diagnostic(
            self.state, self.history, **self._si_kw
        )
        return pe, float(diagnostic["si"]), gains, diagnostic


class TrajectoryWriter:
    """Write a phase-space-ready JSONL trajectory shared across substrates."""

    SCHEMA_VERSION = "ast-tau-v1.1"

    def __init__(
        self,
        path: str | Path,
        *,
        substrate: str,
        session_id: str,
        schema: Sequence[str],
        meta: Mapping[str, Any] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._file.write(
            json.dumps(
                {
                    "record": "header",
                    "schema_version": self.SCHEMA_VERSION,
                    "substrate": substrate,
                    "session_id": session_id,
                    "task_schema_M": list(schema),
                    "calibration": _MANIFEST["manifest_version"],
                    "meta": dict(meta or {}),
                },
                sort_keys=True,
            )
            + "\n"
        )

    def write_exchange(
        self,
        *,
        t: int,
        evidence_ref: Any,
        target_dim: str,
        upsilon: Mapping[str, float],
        gains: Mapping[str, float],
        pe: Mapping[str, float],
        si: float,
        eta: Mapping[str, float] | None = None,
        rho: Mapping[str, float] | None = None,
        dampening: float | None = None,
        ground_truth: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "record": "exchange",
            "t": int(t),
            "y_ref": evidence_ref,
            "i_t": target_dim,
            "upsilon": {k: round(float(v), 8) for k, v in upsilon.items()},
            "delta_upsilon": {
                k: round(float(v), 8) for k, v in gains.items() if v > 0.0
            },
            "PE": {k: round(float(v), 8) for k, v in pe.items()},
            "PE_total": round(float(sum(pe.values())), 8),
            "knowledge_mean": round(float(np.mean(list(upsilon.values()))), 8),
            "gain_total": round(float(sum(gains.values())), 8),
            "SI": round(float(si), 8),
            "eta": {k: round(float(v), 8) for k, v in (eta or {}).items()},
            "rho": {k: round(float(v), 8) for k, v in (rho or {}).items()},
            "dampening": None if dampening is None else round(float(dampening), 8),
            "gt": dict(ground_truth or {}),
        }
        self._file.write(json.dumps(record, sort_keys=True) + "\n")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def run_binding(
    binding: SubstrateBinding,
    evidence_stream: Iterable[Any],
    *,
    output_path: str | Path | None = None,
    session_id: str = "session",
    meta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one evidence stream and optionally persist the common JSONL format."""

    iterator: Iterator[Any] = iter(evidence_stream)
    try:
        first = next(iterator)
    except StopIteration:
        return []

    first_embedding = binding.embed(first)
    runner = TelemetryRunner(
        binding.schema,
        first_embedding.size,
        importance=binding.importance,
        dependencies=binding.dependencies,
    )
    writer = (
        TrajectoryWriter(
            output_path,
            substrate=binding.name,
            session_id=session_id,
            schema=binding.schema,
            meta=meta,
        )
        if output_path is not None
        else None
    )

    records: list[dict[str, Any]] = []
    try:
        for t, evidence in enumerate(chain((first,), iterator), start=1):
            embedding = first_embedding if t == 1 else binding.embed(evidence)
            scores = binding.score(evidence)
            target = binding.target(evidence)
            pe, si, gains, diagnostic = runner.step(embedding, scores, target)
            record = {
                "t": t,
                "target": target,
                "upsilon": runner.state.values.copy(),
                "gains": gains,
                "pe": pe,
                "pe_total": float(sum(pe.values())),
                "si": float(si),
                "eta": dict(diagnostic["eta"]),
                "rho": dict(diagnostic["rho"]),
                "dampening": float(diagnostic["dampening"]),
                "knowledge_mean": float(np.mean(list(runner.state.values.values()))),
                "ground_truth": dict(binding.ground_truth(evidence)),
                "evidence_ref": binding.evidence_ref(evidence),
            }
            records.append(record)
            if writer is not None:
                writer.write_exchange(
                    t=t,
                    evidence_ref=record["evidence_ref"],
                    target_dim=target,
                    upsilon=record["upsilon"],
                    gains=gains,
                    pe=pe,
                    si=si,
                    eta=diagnostic["eta"],
                    rho=diagnostic["rho"],
                    dampening=diagnostic["dampening"],
                    ground_truth=record["ground_truth"],
                )
    finally:
        if writer is not None:
            writer.close()
    return records


def load_trajectory(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load an ``ast-tau-v1`` JSONL file."""

    lines = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("record") != "header":
        raise ValueError("trajectory does not begin with an AST header")
    return lines[0], [line for line in lines[1:] if line.get("record") == "exchange"]
