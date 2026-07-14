"""Small dependency-free evaluation helpers for synthetic AST demonstrations."""

from __future__ import annotations

from typing import Iterable, Mapping, Any
import math


def binary_auc(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    pos = list(positive)
    neg = list(negative)
    if not pos or not neg:
        return None
    wins = sum(p > n for p in pos for n in neg)
    ties = sum(p == n for p in pos for n in neg)
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def detector_metrics(
    records: list[Mapping[str, Any]], threshold: float = 0.20
) -> dict[str, float | int | None]:
    labels = [bool(r["ground_truth"].get("stall", False)) for r in records]
    scores = [float(r["si"]) for r in records]
    predictions = [score >= threshold for score in scores]

    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum((not y) and p for y, p in zip(labels, predictions))
    fn = sum(y and (not p) for y, p in zip(labels, predictions))
    tn = sum((not y) and (not p) for y, p in zip(labels, predictions))

    # None means "not applicable" (no positives, or no predictions), not "scored 0".
    # A stall-free control session has no positives; reporting recall=0.0 there would
    # read as a detector failure when it is simply undefined.
    has_pos = any(labels)
    precision = tp / (tp + fp) if tp + fp else (0.0 if has_pos else None)
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision and recall
        else (None if not has_pos else 0.0)
    )
    false_positive_rate = fp / (fp + tn) if fp + tn else None
    pos_scores = [s for s, y in zip(scores, labels) if y]
    neg_scores = [s for s, y in zip(scores, labels) if not y]

    first_stall = next((i for i, label in enumerate(labels) if label), None)
    first_detection = None
    if first_stall is not None:
        first_detection = next(
            (i for i in range(first_stall, len(predictions)) if predictions[i]), None
        )
    latency = (
        first_detection - first_stall
        if first_stall is not None and first_detection is not None
        else None
    )

    return {
        "n_exchanges": len(records),
        "n_stall": sum(labels),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "auc": binary_auc(pos_scores, neg_scores),
        "mean_si_stall": sum(pos_scores) / len(pos_scores) if pos_scores else None,
        "mean_si_clean": sum(neg_scores) / len(neg_scores) if neg_scores else None,
        "detection_latency": latency,
        "final_knowledge_mean": float(records[-1]["knowledge_mean"]) if records else math.nan,
        "final_pe_total": float(records[-1]["pe_total"]) if records else math.nan,
    }


# ---------------------------------------------------------------------------
# Threshold-free reporting
# ---------------------------------------------------------------------------
def cliffs_delta(positive: Iterable[float], negative: Iterable[float]) -> float | None:
    """Non-parametric effect size in [-1, 1]. 2*AUC - 1."""
    auc = binary_auc(positive, negative)
    return None if auc is None else 2.0 * auc - 1.0


def separation_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Threshold-free summary. This is the headline for any binding whose theta
    has not been fitted on a control set large enough to resolve the target FPR.

    AUC and mean-SI separation are invariant to the monotone rescaling that a
    change of embedding map induces; a fixed threshold is not. See
    calibration_manifest.json -> per_binding_calibration.
    """
    labels = [bool(r["ground_truth"].get("stall", False)) for r in records]
    scores = [float(r["si"]) for r in records]
    stall = [s for s, y in zip(scores, labels) if y]
    clean = [s for s, y in zip(scores, labels) if not y]

    # A within-condition AUC is VACUOUS when every negative exchange is a
    # structural zero. SI is identically 0 on the first exchange of any session
    # (no prior subspace exists to be recurrent against), so a condition whose
    # only clean exchanges are warm-up exchanges scores AUC 1.000 for free. That
    # is not a result, and reporting it would invite someone to quote it.
    # Pool across conditions instead: real negatives, real margin.
    vacuous = bool(stall) and bool(clean) and all(score == 0.0 for score in clean)
    auc = None if vacuous else binary_auc(stall, clean)

    return {
        "n_exchanges": len(records),
        "n_stall": sum(labels),
        "auc": auc,
        "auc_vacuous": vacuous,
        "auc_note": (
            "not reported: every clean exchange in this condition is a warm-up "
            "exchange with SI = 0 by construction; see the pooled row"
            if vacuous else None
        ),
        "cliffs_delta": None if vacuous else cliffs_delta(stall, clean),
        "mean_si_stall": sum(stall) / len(stall) if stall else None,
        "mean_si_clean": sum(clean) / len(clean) if clean else None,
        "si_separation": (
            sum(stall) / len(stall) - sum(clean) / len(clean)
            if stall and clean else None
        ),
        "max_si_clean": max(clean) if clean else None,
        "min_si_stall": min(stall) if stall else None,
        "perfect_rank_separation": (
            None if vacuous else (bool(min(stall) > max(clean)) if stall and clean else None)
        ),
        "final_knowledge_mean": float(records[-1]["knowledge_mean"]) if records else math.nan,
        "final_pe_total": float(records[-1]["pe_total"]) if records else math.nan,
    }


def threshold_sweep(
    runs: Mapping[str, list[Mapping[str, Any]]],
    thresholds: Iterable[float] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
) -> list[dict[str, Any]]:
    """Pooled operating characteristic across all conditions of one substrate.

    Reported INSTEAD OF choosing an operating point, so a reader can see where
    theta would have to sit for this binding without the author having picked it.
    """
    labels, scores = [], []
    for records in runs.values():
        for record in records:
            labels.append(bool(record["ground_truth"].get("stall", False)))
            scores.append(float(record["si"]))

    rows = []
    for threshold in thresholds:
        predictions = [s >= threshold for s in scores]
        tp = sum(y and p for y, p in zip(labels, predictions))
        fp = sum((not y) and p for y, p in zip(labels, predictions))
        fn = sum(y and (not p) for y, p in zip(labels, predictions))
        tn = sum((not y) and (not p) for y, p in zip(labels, predictions))
        rows.append({
            "threshold": threshold,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
        })
    return rows


def format_sweep(rows: list[dict[str, Any]]) -> str:
    def cell(value):
        return "  --  " if value is None else f"{value:6.3f}"
    lines = ["  theta   recall  precis     fpr    tp  fp  fn  tn",
             "  " + "-" * 46]
    for row in rows:
        lines.append(
            f"  {row['threshold']:.2f}  {cell(row['recall'])}  {cell(row['precision'])}  "
            f"{cell(row['fpr'])}  {row['tp']:3d} {row['fp']:3d} {row['fn']:3d} {row['tn']:3d}"
        )
    return "\n".join(lines)
