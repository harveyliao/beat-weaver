"""Tests for MuQ embedding export helpers."""

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from beat_weaver.model.muq_embeddings import (
    find_audio_files_in_subfolders,
    summarize_embedding,
)


def test_find_audio_files_in_subfolders_respects_sorted_limit(tmp_path):
    root = tmp_path / "official"
    for name in ("c_track", "a_track", "b_track"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "song.wav").write_bytes(b"wav")

    found = find_audio_files_in_subfolders(root, limit=2)

    assert [path.parent.name for path in found] == ["a_track", "b_track"]
    assert all(path.name == "song.wav" for path in found)


def test_summarize_embedding_reports_shape_bytes_and_nan_flag(tmp_path):
    embedding = np.array([[0.0, 1.0], [-1.0, 2.0]], dtype=np.float32)
    stats = summarize_embedding(
        embedding,
        audio_path=tmp_path / "song.wav",
        embedding_path=tmp_path / "song.npy",
        sample_rate=24000,
        audio_seconds=1.5,
        load_audio_seconds=0.1,
        inference_seconds=0.2,
        save_seconds=0.05,
    )

    assert stats.embedding_shape == [2, 2]
    assert stats.embedding_bytes == embedding.nbytes
    assert stats.contains_nan is False
    assert stats.total_seconds == pytest.approx(0.35)
    assert stats.mean_abs == pytest.approx(1.0)
