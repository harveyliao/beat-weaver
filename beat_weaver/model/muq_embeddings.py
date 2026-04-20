"""MuQ embedding extraction and export helpers."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from beat_weaver.model.audio import _find_audio_in_folder, load_audio

_AUDIO_EXTENSIONS = (".wav", ".ogg", ".egg", ".mp3", ".flac")


@dataclass
class MuQEmbeddingStats:
    """Summary statistics for a single embedding export."""

    audio_path: str
    embedding_path: str
    sample_rate: int
    audio_seconds: float
    embedding_shape: list[int]
    embedding_dtype: str
    embedding_bytes: int
    load_audio_seconds: float
    inference_seconds: float
    save_seconds: float
    total_seconds: float
    mean_abs: float
    std: float
    min_value: float
    max_value: float
    contains_nan: bool


class MuQEmbedder:
    """Thin wrapper around a pretrained MuQ model."""

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str | None = None,
    ) -> None:
        from muq import MuQ

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.model_name = model_name
        self.model = MuQ.from_pretrained(model_name).to(self.device).eval()

    def extract_file(self, audio_path: Path, sample_rate: int = 24000) -> tuple[np.ndarray, dict[str, float]]:
        """Load an audio file and extract MuQ last_hidden_state embeddings."""
        audio_path = Path(audio_path)

        load_t0 = time.perf_counter()
        audio, sr = load_audio(audio_path, sr=sample_rate)
        load_audio_seconds = time.perf_counter() - load_t0

        infer_t0 = time.perf_counter()
        with torch.no_grad():
            wav = torch.from_numpy(audio).unsqueeze(0).float().to(self.device)
            output = self.model(wav)
            embedding = output.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
        inference_seconds = time.perf_counter() - infer_t0

        return embedding, {
            "sample_rate": float(sr),
            "audio_seconds": float(len(audio) / sr),
            "load_audio_seconds": load_audio_seconds,
            "inference_seconds": inference_seconds,
        }


def summarize_embedding(
    embedding: np.ndarray,
    *,
    audio_path: Path,
    embedding_path: Path,
    sample_rate: int,
    audio_seconds: float,
    load_audio_seconds: float,
    inference_seconds: float,
    save_seconds: float,
) -> MuQEmbeddingStats:
    """Build a serializable summary for one embedding."""
    return MuQEmbeddingStats(
        audio_path=str(audio_path),
        embedding_path=str(embedding_path),
        sample_rate=sample_rate,
        audio_seconds=audio_seconds,
        embedding_shape=list(embedding.shape),
        embedding_dtype=str(embedding.dtype),
        embedding_bytes=int(embedding.nbytes),
        load_audio_seconds=load_audio_seconds,
        inference_seconds=inference_seconds,
        save_seconds=save_seconds,
        total_seconds=load_audio_seconds + inference_seconds + save_seconds,
        mean_abs=float(np.abs(embedding).mean()),
        std=float(embedding.std()),
        min_value=float(embedding.min()),
        max_value=float(embedding.max()),
        contains_nan=bool(np.isnan(embedding).any()),
    )


def find_audio_files_in_subfolders(
    root: Path,
    *,
    limit: int | None = None,
) -> list[Path]:
    """Find one audio file in each first-level subfolder, sorted by folder name."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    audio_files: list[Path] = []
    subfolders = sorted(path for path in root.iterdir() if path.is_dir())

    for folder in subfolders:
        info_dat = folder / "Info.dat"
        info_dat_lower = folder / "info.dat"
        audio_path = None
        if info_dat.exists():
            audio_path = _find_audio_in_folder(folder, info_dat)
        elif info_dat_lower.exists():
            audio_path = _find_audio_in_folder(folder, info_dat_lower)

        if audio_path is None:
            for ext in _AUDIO_EXTENSIONS:
                matches = sorted(folder.glob(f"*{ext}"))
                if matches:
                    audio_path = matches[0]
                    break

        if audio_path is None:
            continue

        audio_files.append(audio_path)
        if limit is not None and len(audio_files) >= limit:
            break

    return audio_files


def export_embeddings(
    audio_paths: list[Path],
    output_dir: Path,
    *,
    model_name: str = "OpenMuQ/MuQ-large-msd-iter",
    device: str | None = None,
) -> dict[str, object]:
    """Extract and save MuQ embeddings for a list of audio paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedder = MuQEmbedder(model_name=model_name, device=device)
    items: list[MuQEmbeddingStats] = []
    started_at = time.perf_counter()

    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        output_name = f"{audio_path.parent.name}_{audio_path.stem}.npy"
        embedding_path = output_dir / output_name
        stats_path = output_dir / f"{audio_path.parent.name}_{audio_path.stem}.json"

        embedding, timing = embedder.extract_file(audio_path)

        save_t0 = time.perf_counter()
        np.save(embedding_path, embedding)
        save_seconds = time.perf_counter() - save_t0

        stats = summarize_embedding(
            embedding,
            audio_path=audio_path,
            embedding_path=embedding_path,
            sample_rate=int(timing["sample_rate"]),
            audio_seconds=timing["audio_seconds"],
            load_audio_seconds=timing["load_audio_seconds"],
            inference_seconds=timing["inference_seconds"],
            save_seconds=save_seconds,
        )
        stats_path.write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
        items.append(stats)

    total_seconds = time.perf_counter() - started_at
    summary = {
        "model_name": model_name,
        "device": str(embedder.device),
        "num_files": len(items),
        "total_wall_seconds": total_seconds,
        "items": [asdict(item) for item in items],
        "aggregate": _aggregate_stats(items),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _aggregate_stats(items: list[MuQEmbeddingStats]) -> dict[str, float | int | list[int]]:
    """Aggregate output size and timing statistics across exports."""
    if not items:
        return {
            "total_embedding_bytes": 0,
            "mean_embedding_bytes": 0.0,
            "mean_audio_seconds": 0.0,
            "mean_inference_seconds": 0.0,
            "max_inference_seconds": 0.0,
            "mean_total_seconds": 0.0,
            "shape_set": [],
        }

    return {
        "total_embedding_bytes": int(sum(item.embedding_bytes for item in items)),
        "mean_embedding_bytes": float(np.mean([item.embedding_bytes for item in items])),
        "mean_audio_seconds": float(np.mean([item.audio_seconds for item in items])),
        "mean_inference_seconds": float(np.mean([item.inference_seconds for item in items])),
        "max_inference_seconds": float(max(item.inference_seconds for item in items)),
        "mean_total_seconds": float(np.mean([item.total_seconds for item in items])),
        "shape_set": sorted({tuple(item.embedding_shape) for item in items}),
    }
