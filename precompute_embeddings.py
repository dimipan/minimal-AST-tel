"""Precompute Qwen3 embeddings for the synthetic SAR hiker evidence stream.

This script calls a local Ollama server. The corpus is synthetic, but the
representations are produced by the real ``qwen3-embedding`` model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import requests

from corpus_hiker import iter_all_evidence

DEFAULT_MODEL = "qwen3-embedding"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/embeddings"
DEFAULT_OUTPUT = Path("data/hiker_embeddings_qwen3.npz")


def _normalise(vector: Any) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Ollama returned an empty or non-finite embedding")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Ollama returned a zero-norm embedding")
    return arr / norm


def _request_embedding(
    session: requests.Session,
    *,
    text: str,
    model: str,
    url: str,
    timeout: float,
) -> np.ndarray:
    """Request one embedding, supporting Ollama's legacy and current payloads."""
    response = session.post(
        url,
        json={"model": model, "prompt": text},
        timeout=timeout,
    )

    # Some Ollama versions expose /api/embed rather than /api/embeddings.
    if response.status_code == 404 and url.rstrip("/").endswith("/api/embeddings"):
        embed_url = url.rstrip("/")[: -len("/api/embeddings")] + "/api/embed"
        response = session.post(
            embed_url,
            json={"model": model, "input": text},
            timeout=timeout,
        )

    response.raise_for_status()
    payload = response.json()
    if "embedding" in payload:
        return _normalise(payload["embedding"])
    if "embeddings" in payload and payload["embeddings"]:
        return _normalise(payload["embeddings"][0])
    raise KeyError("Ollama response contained neither 'embedding' nor 'embeddings'")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute qwen3-embedding vectors for the synthetic hiker corpus"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing embedding archive",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    items = list(iter_all_evidence())
    vectors: dict[str, np.ndarray] = {}
    with requests.Session() as session:
        for index, item in enumerate(items, start=1):
            print(f"[{index:02d}/{len(items):02d}] {item.evidence_id}")
            vectors[item.evidence_id] = _request_embedding(
                session,
                text=item.text,
                model=args.model,
                url=args.ollama_url,
                timeout=args.timeout,
            )

    dimensions = {vector.size for vector in vectors.values()}
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions changed during generation: {sorted(dimensions)}")

    np.savez_compressed(output, **vectors)
    text_digest = hashlib.sha256(
        "\n".join(f"{item.evidence_id}\t{item.text}" for item in items).encode("utf-8")
    ).hexdigest()
    archive_digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "format": "npz",
        "allow_pickle": False,
        "model": args.model,
        "provider": "local Ollama",
        "endpoint": args.ollama_url,
        "input": "Question and answer concatenated as one evidence item",
        "dimension": int(next(iter(dimensions))),
        "normalization": "l2",
        "corpus": "fully synthetic SAR hiker",
        "evidence_count": len(items),
        "texts_sha256": text_digest,
        "archive_sha256": archive_digest,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(items)} qwen embeddings to {output}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
