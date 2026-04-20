# MuQ Repo Migration Checklist

Target repo name: `muq-beat-weaver`

Goal: create a separate repo for rapid pipeline-component experiments, starting with swapping the feature extraction / encoder path to pretrained MuQ while preserving Beat Weaver's tokenizer, decoder-side logic, and domain semantics.

## Stage 1: Scaffold the new repo

- [x] Create `../muq-beat-weaver`
- [x] Add `pyproject.toml`
- [x] Add `README.md`
- [x] Add package root `muq_beat_weaver/`
- [x] Add `configs/`
- [x] Add `scripts/`
- [x] Add `tests/`

## Stage 2: Copy reusable modules

- [x] Copy `beat_weaver/model/tokenizer.py` -> `muq_beat_weaver/model/tokenizer.py`
- [x] Copy `beat_weaver/model/inference.py` -> `muq_beat_weaver/model/inference.py`
- [x] Copy `beat_weaver/model/training.py` -> `muq_beat_weaver/model/training.py`
- [x] Copy `beat_weaver/model/exporter.py` -> `muq_beat_weaver/model/exporter.py`
- [x] Copy `beat_weaver/model/evaluate.py` -> `muq_beat_weaver/model/evaluate.py`
- [x] Copy `beat_weaver/parsers/` -> `muq_beat_weaver/parsers/`
- [x] Copy `beat_weaver/schemas/` -> `muq_beat_weaver/schemas/`
- [x] Copy `tests/fixtures/` -> `tests/fixtures/`

## Stage 3: Rebuild the model boundary around MuQ

- [x] Create `muq_beat_weaver/model/config.py` with MuQ-first config fields
- [x] Create `muq_beat_weaver/model/audio.py` for waveform loading, mono conversion, resampling, padding, and masks
- [x] Create `muq_beat_weaver/model/encoder.py` for MuQ loading and encoder masking
- [x] Create `muq_beat_weaver/model/adapter.py` for projection from MuQ hidden size to decoder dim
- [x] Create `muq_beat_weaver/model/decoder.py` by extracting decoder-side logic from `beat_weaver/model/transformer.py`
- [x] Create `muq_beat_weaver/model/model.py` to compose encoder, adapter, and decoder
- [ ] Port useful parts of `beat_weaver/model/dataset.py` into a MuQ-compatible `muq_beat_weaver/model/dataset.py`

## Stage 4: Add repo entry points

- [x] Create `muq_beat_weaver/cli.py` with experiment-focused commands only
- [x] Add `scripts/train.py`
- [x] Add `scripts/generate.py`
- [x] Add `scripts/evaluate.py`
- [x] Add `scripts/embed_muq.py`

## Stage 5: Port tests

- [x] Port tokenizer tests
- [x] Port inference tests
- [x] Port exporter tests
- [x] Port evaluate tests
- [x] Port parser tests
- [x] Port schema tests
- [ ] Split and port MuQ embedding tests into encoder-focused tests

## Stage 6: Exclude Beat Weaver baseline-only concerns

- [x] Do not copy `beat_weaver/sources/`
- [x] Do not copy `beat_weaver/pipeline/`
- [x] Do not copy `beat_weaver/storage/`
- [x] Do not copy the current monolithic `beat_weaver/cli.py`
- [x] Do not copy the current encoder implementation as-is
- [x] Do not copy current conformer config JSON files as active configs

## Stage 7: First validation target

- [ ] `uv sync` succeeds in the new repo
- [x] Imports work for the new package
- [ ] One config can build the model
- [ ] MuQ wrapper returns `memory` and `memory_mask`
- [ ] A decoder forward pass runs with projected encoder memory
- [ ] Basic tests pass

## Stage 8: Follow-up work after scaffold

- [ ] Hook training to waveform / MuQ inputs end-to-end
- [ ] Add generation path from checkpoint
- [ ] Add evaluation path
- [ ] Add frozen-encoder baseline config
- [ ] Add partial-unfreeze config
- [ ] Decide whether to consume prepared parquet directly or define a cleaner interchange format
