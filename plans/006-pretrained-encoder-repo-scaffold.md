## Pretrained Encoder Repo Scaffold

Goal: stand up a clean repo for a Beat Weaver style decoder conditioned on a pretrained audio encoder.

### Repo Shape

Suggested layout:

```text
pretrained-beat-weaver/
  pyproject.toml
  README.md
  configs/
    baseline.json
    frozen_encoder.json
    partial_finetune.json
  scripts/
    train.py
    generate.py
    evaluate.py
  pretrained_beat_weaver/
    __init__.py
    cli.py
    model/
      __init__.py
      config.py
      encoder.py
      decoder.py
      model.py
      tokenizer.py
      audio.py
      dataset.py
      training.py
      inference.py
      exporter.py
      evaluate.py
    parsers/
    schemas/
    sources/
  tests/
```

### Copy First

Copy these modules nearly unchanged from this repo:

- `beat_weaver/model/tokenizer.py`
- `beat_weaver/model/inference.py`
- `beat_weaver/model/training.py`
- `beat_weaver/model/exporter.py`
- `beat_weaver/model/evaluate.py`
- `beat_weaver/model/dataset.py`
- `beat_weaver/parsers/`
- `beat_weaver/schemas/`

These give you the domain logic and output format without locking you into the current encoder.

### Split The Current Model File

Do not carry over `transformer.py` as one large file.

Instead split it into:

- `model/encoder.py`
  - pretrained encoder wrapper
  - optional projection layer
  - freeze/unfreeze utilities

- `model/decoder.py`
  - current `TokenDecoder`
  - RoPE helpers if you still want them on the decoder side

- `model/model.py`
  - top-level model class
  - glue between encoder and decoder

This keeps the pretrained integration isolated.

### Encoder Contract

Design the encoder wrapper around one stable interface:

```text
forward(audio, audio_mask=None) -> memory, memory_mask
```

Rules:

- `audio` is whatever the pretrained model expects
- `memory` is `(batch, time, hidden)`
- `memory_mask` is aligned to the encoder output time axis

Avoid tying the rest of the repo to mel-only assumptions.

### Config Shape

Start with these config fields:

- `encoder_type`
- `pretrained_model_name`
- `encoder_input_type`
- `encoder_output_dim`
- `freeze_encoder`
- `unfreeze_last_n_layers`
- `decoder_dim`
- `decoder_layers`
- `decoder_heads`
- `decoder_ff_dim`
- `project_encoder_to_decoder_dim`
- `sample_rate`
- `max_audio_len_seconds`
- `batch_size`
- `learning_rate`
- `encoder_learning_rate`
- `warmup_steps`
- `gradient_accumulation_steps`

Keep separate LR support for encoder and decoder. You will probably need it.

### Audio Frontend

Make `model/audio.py` encoder-aware, not globally fixed.

Support one of these modes:

- raw waveform input
- mel input for AST-like models
- model-native feature extractor path

Do not hardwire one frontend until you commit to a specific pretrained family.

### Recommended First Milestone

Milestone 1:

- frozen pretrained encoder
- single linear projection to decoder dim
- existing autoregressive decoder
- existing token grammar constraints

This is the lowest-risk baseline and gives you a meaningful result quickly.

### Suggested Implementation Order

1. Copy tokenizer, parsing, schema, and export pieces.
2. Split decoder logic out of the current model file.
3. Implement a dummy encoder wrapper that returns random or projected features.
4. Make training and inference run end-to-end with the dummy encoder.
5. Replace dummy encoder with the real pretrained encoder wrapper.
6. Add freezing and partial unfreezing controls.
7. Tune batching, truncation, and memory usage.

The dummy wrapper step matters because it validates the repo shape before adding external model complexity.

### Test Plan

Start with these tests:

- encoder wrapper returns expected `memory` and `memory_mask` shapes
- decoder accepts projected encoder memory
- full model forward pass works on one small batch
- training step produces gradients in decoder and adapter
- frozen encoder does not receive gradients
- partially unfrozen encoder only updates allowed layers
- inference runs one short generation pass

### Good Boundary Decisions

Make these boundaries explicit early:

- encoder owns feature extraction requirements
- decoder never knows which encoder produced the memory
- adapter owns hidden-size conversion
- training loop owns optimizer groups and freeze policy

If you keep those boundaries clean, trying different pretrained encoders becomes cheap.

### What Not To Do Initially

Avoid these early complications:

- supporting multiple unrelated pretrained encoders at once
- keeping backward compatibility with the current config format
- merging custom Conformer and pretrained encoder paths into one codepath
- optimizing long-song chunking before the short-song path is stable

### Exit Criteria For The Prototype

The new repo is structurally sound when:

- one config can train end-to-end,
- one checkpoint can generate a valid map,
- encoder freezing works as intended,
- and swapping the encoder wrapper does not require decoder changes.
