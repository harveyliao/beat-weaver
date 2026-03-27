"""Write normalized Beat Saber beatmap data to Parquet files and JSON metadata."""

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from beat_weaver.schemas.normalized import NormalizedBeatmap

logger = logging.getLogger(__name__)

# Maximum Parquet file size in bytes before starting a new file.
MAX_FILE_BYTES: int = 1_000_000_000  # 1 GB
_PROCESSED_PATTERNS = (
    "notes_*.parquet",
    "bombs_*.parquet",
    "obstacles_*.parquet",
    "metadata.json",
)

# --- Arrow schemas -----------------------------------------------------------

NOTES_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
        pa.field("color", pa.int8()),
        pa.field("cut_direction", pa.int8()),
        pa.field("angle_offset", pa.int16()),
    ]
)

BOMBS_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
    ]
)

OBSTACLES_SCHEMA = pa.schema(
    [
        pa.field("song_hash", pa.string()),
        pa.field("source", pa.string()),
        pa.field("difficulty", pa.string()),
        pa.field("characteristic", pa.string()),
        pa.field("bpm", pa.float32()),
        pa.field("beat", pa.float32()),
        pa.field("time_seconds", pa.float32()),
        pa.field("duration_beats", pa.float32()),
        pa.field("x", pa.int8()),
        pa.field("y", pa.int8()),
        pa.field("width", pa.int8()),
        pa.field("height", pa.int8()),
    ]
)

_SCHEMAS = {
    "notes": NOTES_SCHEMA,
    "bombs": BOMBS_SCHEMA,
    "obstacles": OBSTACLES_SCHEMA,
}


def has_processed_output(output_dir: Path) -> bool:
    """Return True when *output_dir* already contains processed output files."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False
    return any(any(output_dir.glob(pattern)) for pattern in _PROCESSED_PATTERNS)


def _empty_columns(schema: pa.Schema) -> dict[str, list]:
    return {name: [] for name in schema.names}


def _clamp8(values: list[int]) -> list[int]:
    return [max(-128, min(127, v)) for v in values]


def _beatmaps_to_tables(
    beatmaps: list[NormalizedBeatmap],
) -> tuple[dict[str, dict[str, pa.Table]], dict[str, dict]]:
    """Convert beatmaps into Arrow tables grouped by song_hash and metadata rows."""
    notes_by_hash: dict[str, dict[str, list]] = {}
    bombs_by_hash: dict[str, dict[str, list]] = {}
    obstacles_by_hash: dict[str, dict[str, list]] = {}
    metadata_by_hash: dict[str, dict] = {}

    for bm in beatmaps:
        meta = bm.metadata
        diff = bm.difficulty_info

        song_hash = meta.hash
        source = meta.source
        difficulty = diff.difficulty
        characteristic = diff.characteristic
        bpm = meta.bpm

        if song_hash not in notes_by_hash:
            notes_by_hash[song_hash] = _empty_columns(NOTES_SCHEMA)
        cols = notes_by_hash[song_hash]
        for note in bm.notes:
            cols["song_hash"].append(song_hash)
            cols["source"].append(source)
            cols["difficulty"].append(difficulty)
            cols["characteristic"].append(characteristic)
            cols["bpm"].append(bpm)
            cols["beat"].append(note.beat)
            cols["time_seconds"].append(note.time_seconds)
            cols["x"].append(note.x)
            cols["y"].append(note.y)
            cols["color"].append(note.color)
            cols["cut_direction"].append(note.cut_direction)
            cols["angle_offset"].append(note.angle_offset)

        if song_hash not in bombs_by_hash:
            bombs_by_hash[song_hash] = _empty_columns(BOMBS_SCHEMA)
        bcols = bombs_by_hash[song_hash]
        for bomb in bm.bombs:
            bcols["song_hash"].append(song_hash)
            bcols["source"].append(source)
            bcols["difficulty"].append(difficulty)
            bcols["characteristic"].append(characteristic)
            bcols["bpm"].append(bpm)
            bcols["beat"].append(bomb.beat)
            bcols["time_seconds"].append(bomb.time_seconds)
            bcols["x"].append(bomb.x)
            bcols["y"].append(bomb.y)

        if song_hash not in obstacles_by_hash:
            obstacles_by_hash[song_hash] = _empty_columns(OBSTACLES_SCHEMA)
        ocols = obstacles_by_hash[song_hash]
        for obs in bm.obstacles:
            ocols["song_hash"].append(song_hash)
            ocols["source"].append(source)
            ocols["difficulty"].append(difficulty)
            ocols["characteristic"].append(characteristic)
            ocols["bpm"].append(bpm)
            ocols["beat"].append(obs.beat)
            ocols["time_seconds"].append(obs.time_seconds)
            ocols["duration_beats"].append(obs.duration_beats)
            ocols["x"].append(obs.x)
            ocols["y"].append(obs.y)
            ocols["width"].append(obs.width)
            ocols["height"].append(obs.height)

        if song_hash not in metadata_by_hash:
            metadata_by_hash[song_hash] = {
                "hash": song_hash,
                "source": source,
                "source_id": meta.source_id,
                "song_name": meta.song_name,
                "song_author": meta.song_author,
                "mapper_name": meta.mapper_name,
                "bpm": bpm,
                "score": meta.score,
                "difficulties": [],
            }

        metadata_by_hash[song_hash]["difficulties"].append(
            {
                "characteristic": characteristic,
                "difficulty": difficulty,
                "note_count": diff.note_count,
                "nps": diff.nps,
            }
        )

    for cols in notes_by_hash.values():
        for key in ("x", "y", "color", "cut_direction"):
            cols[key] = _clamp8(cols[key])
    for cols in bombs_by_hash.values():
        for key in ("x", "y"):
            cols[key] = _clamp8(cols[key])
    for cols in obstacles_by_hash.values():
        for key in ("x", "y", "width", "height"):
            cols[key] = _clamp8(cols[key])

    tables_by_prefix = {
        "notes": {},
        "bombs": {},
        "obstacles": {},
    }
    valid_metadata: dict[str, dict] = {}

    all_hashes = sorted(
        set(notes_by_hash) | set(bombs_by_hash) | set(obstacles_by_hash) | set(metadata_by_hash)
    )
    for song_hash in all_hashes:
        try:
            tables_by_prefix["notes"][song_hash] = pa.table(
                notes_by_hash.get(song_hash, _empty_columns(NOTES_SCHEMA)),
                schema=NOTES_SCHEMA,
            )
            tables_by_prefix["bombs"][song_hash] = pa.table(
                bombs_by_hash.get(song_hash, _empty_columns(BOMBS_SCHEMA)),
                schema=BOMBS_SCHEMA,
            )
            tables_by_prefix["obstacles"][song_hash] = pa.table(
                obstacles_by_hash.get(song_hash, _empty_columns(OBSTACLES_SCHEMA)),
                schema=OBSTACLES_SCHEMA,
            )
            if song_hash in metadata_by_hash:
                valid_metadata[song_hash] = metadata_by_hash[song_hash]
        except (pa.ArrowInvalid, OverflowError, ValueError, TypeError) as exc:
            meta = metadata_by_hash.get(song_hash, {})
            logger.warning(
                "Skipping malformed song %s (%s): %s",
                song_hash,
                meta.get("source_id", "unknown"),
                exc,
            )

    return tables_by_prefix, valid_metadata


class ParquetWriteSession:
    """Append beatmaps to numbered Parquet files and finalize metadata once."""

    def __init__(
        self,
        output_dir: Path,
        max_file_bytes: int = MAX_FILE_BYTES,
        *,
        allow_existing: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_file_bytes = max_file_bytes
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not allow_existing and has_processed_output(self.output_dir):
            raise FileExistsError(
                f"Processed output already exists in {self.output_dir}. "
                "Use a new output directory or remove existing parquet/metadata files."
            )

        self._writers: dict[str, pq.ParquetWriter | None] = {
            prefix: None for prefix in _SCHEMAS
        }
        self._current_paths: dict[str, Path | None] = {
            prefix: None for prefix in _SCHEMAS
        }
        self._file_indices: dict[str, int] = {prefix: 0 for prefix in _SCHEMAS}
        self._written_files: dict[str, list[Path]] = {prefix: [] for prefix in _SCHEMAS}
        self._metadata_by_hash: dict[str, dict] = {}
        self._closed = False

    def _open_writer(self, prefix: str) -> tuple[pq.ParquetWriter, Path]:
        path = self.output_dir / f"{prefix}_{self._file_indices[prefix]:04d}.parquet"
        writer = pq.ParquetWriter(path, _SCHEMAS[prefix], compression="snappy")
        self._file_indices[prefix] += 1
        return writer, path

    def _close_writer(self, prefix: str) -> None:
        writer = self._writers[prefix]
        path = self._current_paths[prefix]
        if writer is None or path is None:
            return
        writer.close()
        self._written_files[prefix].append(path)
        self._writers[prefix] = None
        self._current_paths[prefix] = None

    def _append_tables(self, prefix: str, tables_by_hash: dict[str, pa.Table]) -> None:
        for _song_hash, table in sorted(tables_by_hash.items()):
            if table.num_rows == 0:
                continue
            if self._writers[prefix] is None:
                self._writers[prefix], self._current_paths[prefix] = self._open_writer(prefix)

            writer = self._writers[prefix]
            current_path = self._current_paths[prefix]
            assert writer is not None and current_path is not None
            writer.write_table(table)

            current_size = current_path.stat().st_size
            if current_size >= self.max_file_bytes:
                self._close_writer(prefix)

    def _merge_metadata(self, metadata_by_hash: dict[str, dict]) -> None:
        for song_hash, meta in metadata_by_hash.items():
            if song_hash not in self._metadata_by_hash:
                self._metadata_by_hash[song_hash] = meta
                continue
            self._metadata_by_hash[song_hash]["difficulties"].extend(meta["difficulties"])
            if self._metadata_by_hash[song_hash].get("score") is None and meta.get("score") is not None:
                self._metadata_by_hash[song_hash]["score"] = meta["score"]

    def append(self, beatmaps: list[NormalizedBeatmap]) -> None:
        if self._closed:
            raise RuntimeError("Cannot append after close()")
        if not beatmaps:
            return

        tables_by_prefix, metadata_by_hash = _beatmaps_to_tables(beatmaps)
        for prefix in ("notes", "bombs", "obstacles"):
            self._append_tables(prefix, tables_by_prefix[prefix])
        self._merge_metadata(metadata_by_hash)

    def close(self) -> None:
        if self._closed:
            return
        for prefix in _SCHEMAS:
            self._close_writer(prefix)

        metadata_list = list(self._metadata_by_hash.values())
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=2)

        logger.info(
            "Wrote %d notes files, %d bombs files, %d obstacles files to %s",
            len(self._written_files["notes"]),
            len(self._written_files["bombs"]),
            len(self._written_files["obstacles"]),
            self.output_dir,
        )
        self._closed = True

    def __enter__(self) -> "ParquetWriteSession":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if exc_type is None:
            self.close()
        else:
            for prefix in _SCHEMAS:
                self._close_writer(prefix)
            self._closed = True


def write_parquet(
    beatmaps: list[NormalizedBeatmap],
    output_dir: Path,
    max_file_bytes: int = MAX_FILE_BYTES,
) -> None:
    """Write a list of normalized beatmaps to Parquet files and JSON metadata."""
    with ParquetWriteSession(output_dir, max_file_bytes=max_file_bytes) as session:
        session.append(beatmaps)


def read_notes_parquet(path: Path) -> pa.Table:
    """Read notes Parquet file(s) and return a single Arrow table.

    Accepts either a single ``.parquet`` file or a directory containing
    ``notes_*.parquet`` files.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("notes_*.parquet"))
        if not files:
            single = path / "notes.parquet"
            if single.exists():
                return pq.read_table(single)
            raise FileNotFoundError(f"No notes Parquet files in {path}")
        tables = [pq.read_table(f) for f in files]
        return pa.concat_tables(tables)
    return pq.read_table(path)
