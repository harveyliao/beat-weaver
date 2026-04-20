"""Audio preprocessing — mel spectrograms and beat-aligned framing.

Depends on librosa and soundfile (optional ML dependencies).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.interpolate import interp1d
from tqdm import tqdm

logger = logging.getLogger(__name__)
_MUQ_LABEL_RATE = 25.0
_MUQ_DEFAULT_OVERLAP_SECONDS = 5.0


def load_audio(path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file and resample to target sample rate.

    Returns (audio_mono, sample_rate) where audio_mono is float32 1-D.
    """
    path = Path(path)
    audio, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    # Mix to mono
    audio = audio.mean(axis=1)
    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log-mel spectrogram.

    Returns float32 array of shape (n_mels, T).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, window="hann",
    )
    # Convert to dB scale (log-magnitude), ref=max
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def beat_align_spectrogram(
    mel: np.ndarray,
    sr: int,
    hop_length: int,
    bpm: float,
    subdivisions_per_beat: int = 16,
) -> np.ndarray:
    """Resample spectrogram frames to align with beat subdivisions.

    Each output frame corresponds to one 1/16th note position.

    Args:
        mel: Log-mel spectrogram of shape (n_mels, T_frames).
        sr: Audio sample rate.
        hop_length: STFT hop length used for mel.
        bpm: Beats per minute.
        subdivisions_per_beat: Number of subdivisions per beat (default 16).

    Returns:
        Float32 array of shape (n_mels, T_beats) where T_beats is the total
        number of beat subdivisions covered by the audio.
    """
    n_mels, n_frames = mel.shape

    # Time of each spectrogram frame
    frame_times = librosa.frames_to_time(
        np.arange(n_frames), sr=sr, hop_length=hop_length,
    )

    # Duration of the audio
    duration = frame_times[-1] if n_frames > 0 else 0.0

    # Total beat subdivisions
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil(duration * subs_per_second))

    if total_subs == 0:
        return np.zeros((n_mels, 0), dtype=np.float32)

    # Time of each subdivision
    sub_times = np.arange(total_subs) / subs_per_second

    # Interpolate: for each sub_time, find the nearest frame
    # Use linear interpolation across the time axis
    frame_indices = np.interp(sub_times, frame_times, np.arange(n_frames))

    # Vectorized interpolation across all mel bins at once
    x_coords = np.arange(n_frames, dtype=np.float64)
    f = interp1d(x_coords, mel, axis=1, kind="linear",
                 fill_value="extrapolate", assume_sorted=True)
    aligned = f(frame_indices).astype(np.float32)

    return aligned


def detect_bpm(
    audio: np.ndarray, sr: int = 22050, default: float = 120.0,
) -> float:
    """Estimate the BPM of an audio signal using librosa beat tracking.

    Returns the estimated tempo as a float.  Falls back to *default*
    (120 BPM) when beat tracking cannot determine a tempo (e.g. for very
    short or non-rhythmic audio).
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    # librosa may return an array; extract scalar
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else default
    tempo = float(tempo)
    if tempo <= 0:
        return default
    return tempo


def compute_onset_envelope(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute onset strength envelope.

    Returns float32 array of shape (1, T).
    """
    onset = librosa.onset.onset_strength(
        y=audio, sr=sr, hop_length=hop_length,
    )
    return onset.astype(np.float32).reshape(1, -1)


def compute_mel_with_onset(
    audio: np.ndarray,
    sr: int = 22050,
    n_mels: int = 80,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute log-mel spectrogram with onset strength as extra channel.

    Returns float32 array of shape (n_mels + 1, T).
    """
    mel = compute_mel_spectrogram(audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                  hop_length=hop_length)
    onset = compute_onset_envelope(audio, sr=sr, hop_length=hop_length)
    # Align lengths (onset may differ by 1 frame from mel)
    min_len = min(mel.shape[1], onset.shape[1])
    return np.vstack([mel[:, :min_len], onset[:, :min_len]])


# ── Audio manifest ──────────────────────────────────────────────────────────

_AUDIO_EXTENSIONS = {".ogg", ".egg", ".wav", ".mp3", ".flac"}


def _hash_folder(folder: Path) -> str:
    """Compute a content hash for a map folder.

    Uses the same algorithm as ``beat_weaver.pipeline.processor.compute_map_hash``
    so the audio manifest keys match the song hashes in the Parquet data.
    """
    from beat_weaver.pipeline.processor import compute_map_hash

    return compute_map_hash(folder)


def build_audio_manifest(raw_dirs: list[Path]) -> dict[str, str]:
    """Scan raw map folders and build a hash → audio file path mapping.

    Looks for Info.dat files and their referenced audio filenames.
    Falls back to scanning for common audio extensions.
    """
    manifest: dict[str, str] = {}
    skipped = 0

    for raw_dir in raw_dirs:
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            logger.warning("Raw directory not found: %s", raw_dir)
            continue

        info_files = list(raw_dir.rglob("Info.dat"))
        info_files.extend(
            info_file
            for info_file in raw_dir.rglob("info.dat")
            if info_file.name != "Info.dat"
        )

        with tqdm(total=len(info_files), desc="Building manifest") as pbar:
            for info_file in info_files:
                folder = info_file.parent
                song_hash = _hash_folder(folder)

                if song_hash in manifest:
                    pbar.update(1)
                    pbar.set_postfix(found=len(manifest), skipped=skipped)
                    continue

                audio_path = _find_audio_in_folder(folder, info_file)
                if audio_path:
                    manifest[song_hash] = str(audio_path)
                else:
                    skipped += 1

                pbar.update(1)
                pbar.set_postfix(found=len(manifest), skipped=skipped)

    logger.info("Built audio manifest: %d entries", len(manifest))
    return manifest


def _find_audio_in_folder(folder: Path, info_file: Path) -> Path | None:
    """Find the audio file in a map folder."""
    # Try parsing Info.dat for the audio filename
    try:
        import json as _json
        info = _json.loads(info_file.read_text(encoding="utf-8-sig"))
        # v2/v3: _songFilename, v4: audio.songFilename or song.songFilename
        audio_name = (
            info.get("_songFilename")
            or info.get("audio", {}).get("songFilename")
            or info.get("song", {}).get("songFilename")
        )
        if audio_name:
            audio_path = folder / audio_name
            if audio_path.exists():
                return audio_path
    except Exception:
        pass

    # Fallback: scan for common audio files
    for ext in _AUDIO_EXTENSIONS:
        for f in folder.glob(f"*{ext}"):
            return f
    return None


def interpolate_muq_to_beat_grid(
    muq_features: np.ndarray,
    bpm: float,
    muq_hz: float = 25.0,
    subdivisions_per_beat: int = 16,
) -> np.ndarray:
    """Resample MuQ features (25 Hz) onto the 1/16th note beat grid.

    Args:
        muq_features: (T_muq, 1024) — MuQ hidden states at 25 Hz.
        bpm: Beats per minute.
        muq_hz: MuQ output frame rate (default 25 Hz).
        subdivisions_per_beat: Beat subdivisions (default 16 = 1/16th note).

    Returns:
        Float32 array of shape (1024, T_beats), transposed to match mel convention.
    """
    n_muq_frames, feat_dim = muq_features.shape
    if n_muq_frames == 0:
        return np.zeros((feat_dim, 0), dtype=np.float32)

    duration = (n_muq_frames - 1) / muq_hz  # seconds

    # Beat grid times
    beats_per_second = bpm / 60.0
    subs_per_second = beats_per_second * subdivisions_per_beat
    total_subs = int(np.ceil(duration * subs_per_second))
    if total_subs == 0:
        return np.zeros((feat_dim, 0), dtype=np.float32)

    sub_times = np.arange(total_subs) / subs_per_second  # seconds

    # MuQ frame indices (fractional) for each beat position
    muq_indices = sub_times * muq_hz  # float indices into MuQ frames
    # Clip to valid range
    muq_indices = np.clip(muq_indices, 0, n_muq_frames - 1)

    # Linear interpolation in feature space
    lo = np.floor(muq_indices).astype(int)
    hi = np.minimum(lo + 1, n_muq_frames - 1)
    frac = (muq_indices - lo).astype(np.float32)

    aligned = (
        muq_features[lo] * (1 - frac[:, None])
        + muq_features[hi] * frac[:, None]
    )

    # Transpose to (feat_dim, T_beats) to match mel convention (n_mels, T)
    return aligned.T.astype(np.float32)


def _plan_muq_windows(
    audio_seconds: float,
    max_chunk_seconds: float,
    overlap_seconds: float = _MUQ_DEFAULT_OVERLAP_SECONDS,
) -> list[tuple[float, float, float]]:
    """Plan overlapping MuQ windows as (start, end, trim_left) in seconds."""
    if audio_seconds <= 0:
        return []
    if max_chunk_seconds <= 0 or audio_seconds <= max_chunk_seconds:
        return [(0.0, audio_seconds, 0.0)]

    overlap_seconds = max(0.0, min(overlap_seconds, max_chunk_seconds / 4.0))
    stride_seconds = max_chunk_seconds - overlap_seconds
    if stride_seconds <= 0:
        return [(0.0, audio_seconds, 0.0)]

    windows: list[tuple[float, float, float]] = []
    start = 0.0
    while start < audio_seconds:
        end = min(start + max_chunk_seconds, audio_seconds)
        trim_left = 0.0 if not windows else overlap_seconds
        windows.append((start, end, trim_left))
        if end >= audio_seconds:
            break
        start += stride_seconds
    return windows


def _extract_muq_features(
    audio: np.ndarray,
    model,
    device,
    *,
    sample_rate: int = 24000,
    max_chunk_seconds: float = 0.0,
    overlap_seconds: float = _MUQ_DEFAULT_OVERLAP_SECONDS,
) -> np.ndarray:
    """Extract raw MuQ features, chunking long songs when requested."""
    import torch

    audio_seconds = len(audio) / sample_rate if sample_rate > 0 else 0.0
    windows = _plan_muq_windows(audio_seconds, max_chunk_seconds, overlap_seconds)
    if not windows:
        return np.zeros((0, 1024), dtype=np.float32)

    chunks: list[np.ndarray] = []
    for start_sec, end_sec, trim_left_sec in windows:
        start_sample = int(round(start_sec * sample_rate))
        end_sample = int(round(end_sec * sample_rate))
        chunk_audio = audio[start_sample:end_sample]

        with torch.no_grad():
            wav_tensor = torch.from_numpy(chunk_audio).unsqueeze(0).float().to(device)
            output = model(wav_tensor)
            features = output.last_hidden_state[0].cpu().numpy().astype(np.float32)
            del wav_tensor, output

        trim_left_frames = int(round(trim_left_sec * _MUQ_LABEL_RATE))
        if trim_left_frames > 0:
            features = features[trim_left_frames:]
        chunks.append(features)

        if getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()

    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _compute_one_muq(
    audio_path: str,
    song_hash: str,
    cache_path: str,
    muq_model_name: str,
    max_audio_duration: float = 0.0,
) -> str | None:
    """Compute and cache raw MuQ features at 25 Hz.

    Returns song_hash on success, None on error.
    Saves shape (T_muq, 1024) — beat-grid interpolation is deferred to
    dataset __getitem__ time.

    NOTE: Unlike mel caching, MuQ caching is NOT parallelized with
    ProcessPoolExecutor because MuQ requires GPU and loading the model
    per-worker is wasteful. This function is called sequentially from
    warm_muq_cache() which holds a single MuQ model instance.
    """
    try:
        audio, _ = load_audio(Path(audio_path), sr=24000)
        import torch
        from muq import MuQ

        # Load MuQ model (caller should pre-load; this is fallback)
        model = MuQ.from_pretrained(muq_model_name)
        model.eval()

        features = _extract_muq_features(
            audio,
            model,
            torch.device("cpu"),
            sample_rate=24000,
            max_chunk_seconds=max_audio_duration,
        )

        np.save(cache_path, features)
        return song_hash
    except Exception as e:
        logger.warning("Failed to compute MuQ features for %s: %s", song_hash, e)
        return None


def warm_muq_cache(
    processed_dir: Path,
    audio_manifest_path: Path,
    config: "ModelConfig",
    max_songs: int | None = None,
) -> int:
    """Pre-compute raw MuQ features (25 Hz) for all songs in the manifest.

    Runs a frozen MuQ model in fp32 on GPU (or CPU), extracts
    last_hidden_state at 25 Hz, and saves raw features as .npy files
    in ``muq_cache/``.  Beat-grid interpolation is deferred to dataset
    ``__getitem__`` time (cheap linear interp).

    Cache files are keyed by song hash only (no BPM), since raw MuQ
    features are BPM-independent.

    Returns the number of newly computed feature files.
    """
    import torch

    cache_dir = processed_dir / "muq_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Version check — invalidate if model name or cache format changes
    version_file = cache_dir / "VERSION"
    expected_version = f"raw_v2:{config.muq_model_name}"
    needs_clear = False
    if version_file.exists():
        current_version = version_file.read_text(encoding="utf-8").strip()
        if current_version != expected_version:
            needs_clear = True
    elif any(cache_dir.glob("*.npy")):
        needs_clear = True
    if needs_clear:
        n_stale = sum(1 for _ in cache_dir.glob("*.npy"))
        logger.warning(
            "MuQ cache version mismatch (need %s). Clearing %d stale files.",
            expected_version, n_stale,
        )
        for npy_file in cache_dir.glob("*.npy"):
            npy_file.unlink()
    version_file.write_text(expected_version, encoding="utf-8")

    manifest = load_manifest(audio_manifest_path)

    # Find songs that need MuQ computation (no BPM needed for raw features)
    todo: list[tuple[str, str, str]] = []
    for song_hash, audio_path in manifest.items():
        cache_path = str(cache_dir / f"{song_hash}.npy")
        if not Path(cache_path).exists():
            todo.append((str(audio_path), song_hash, cache_path))

    if not todo:
        logger.info("MuQ cache is warm: all %d songs already cached", len(manifest))
        return 0

    if max_songs is not None and len(todo) > max_songs:
        logger.info(
            "Limiting MuQ cache warming to %d of %d songs", max_songs, len(todo),
        )
        todo = todo[:max_songs]

    logger.info(
        "Warming MuQ cache: %d songs to compute (%d already cached)",
        len(todo), len(manifest) - len(todo),
    )

    # Load MuQ model once (fp32, GPU if available)
    from muq import MuQ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    muq_model = MuQ.from_pretrained(config.muq_model_name)
    muq_model.eval()
    muq_model.to(device)
    logger.info("MuQ model loaded on %s", device)

    computed = 0
    max_dur = config.max_audio_duration
    total = len(todo)

    for i, (audio_path, song_hash, cache_path) in enumerate(todo, 1):
        try:
            audio, _ = load_audio(Path(audio_path), sr=24000)
            features = _extract_muq_features(
                audio,
                muq_model,
                device,
                sample_rate=24000,
                max_chunk_seconds=max_dur,
            )

            np.save(cache_path, features)
            computed += 1
        except Exception as e:
            logger.warning("Failed to compute MuQ for %s: %s", song_hash, e)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if i % 100 == 0 or i == total:
            logger.info("MuQ cache progress: %d/%d computed", i, total)

    logger.info("MuQ cache warm: %d newly computed", computed)
    return computed


def save_manifest(manifest: dict[str, str], path: Path) -> None:
    """Save audio manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, str]:
    """Load audio manifest from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
