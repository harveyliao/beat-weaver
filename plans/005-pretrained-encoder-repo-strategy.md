## Pretrained Encoder Repo Strategy

Goal: build a separate repo that reuses the strong parts of Beat Weaver while replacing the in-house Conformer/Transformer audio encoder with a pretrained audio model.

### Recommendation

Start in a separate repo, not in-place.

Reason:
- The current codebase is organized around a custom `AudioEncoder` that consumes mel features and returns decoder memory.
- A pretrained encoder usually changes preprocessing, sequence length, hidden size, freezing strategy, and dependency footprint.
- Keeping those experiments isolated makes comparison, debugging, and cleanup much easier.

Move the approach back in-place only after the pretrained variant proves materially better.

### Reuse As-Is

These parts are good candidates to copy with minimal change:

- Token vocabulary and tokenization logic
  - `beat_weaver/model/tokenizer.py`
  - The compound token design and grammar constraints are specific and useful.

- Decoder-side architecture
  - `beat_weaver/model/transformer.py`
  - Reuse `TokenDecoder`, decoder masking, and autoregressive generation flow.

- Inference loop and constrained decoding
  - `beat_weaver/model/inference.py`
  - The grammar mask and sampling pipeline should transfer cleanly.

- Training loop structure
  - `beat_weaver/model/training.py`
  - Reuse the trainer shape, mixed precision, checkpointing, logging, and loss plumbing.

- Evaluation/export pipeline
  - `beat_weaver/model/evaluate.py`
  - `beat_weaver/model/exporter.py`

- Dataset semantics and map parsing
  - `beat_weaver/model/dataset.py`
  - `beat_weaver/parsers/`
  - `beat_weaver/schemas/`

### Reuse With Edits

- `beat_weaver/model/config.py`
  - Keep the config pattern.
  - Replace Conformer-specific fields with encoder-provider settings such as:
    - `encoder_type`
    - `pretrained_model_name`
    - `freeze_encoder`
    - `encoder_output_dim`
    - `encoder_downsample_factor`
    - `project_encoder_to_decoder_dim`

- `beat_weaver/model/transformer.py`
  - Keep `BeatWeaverModel` shape and decoder concepts.
  - Replace `AudioEncoder` with a provider wrapper around the pretrained model.

- `beat_weaver/model/audio.py`
  - Keep only pieces that still match the pretrained model’s expected input format.
  - Redesign if the new encoder wants raw waveform, different sample rate, or different spectrogram parameters.

### Redesign

These areas should be treated as new design work:

- Encoder implementation
  - Current code assumes mel input projected by a linear layer into `encoder_dim`.
  - A pretrained encoder should instead expose a wrapper with a stable contract:
    - input: waveform or model-specific features
    - output: `(batch, time, hidden)`
    - optional mask aligned with the output time axis

- Encoder-decoder bridge
  - If pretrained hidden size differs from decoder size, add a projection layer.
  - If time resolution is compressed, decide whether to:
    - accept the lower resolution,
    - upsample features,
    - or redesign token timing assumptions.

- Freezing and finetuning policy
  - Support:
    - frozen encoder
    - partially unfrozen top layers
    - full finetuning

- Batching and memory strategy
  - Pretrained encoders often have different VRAM behavior than the current Conformer.
  - Revisit truncation, chunking, and accumulation rules.

### Suggested Architecture

Keep the top-level contract simple:

1. Audio frontend
   - Produces the input representation expected by the pretrained model.

2. Pretrained encoder wrapper
   - Loads the external model.
   - Returns contextualized audio features and an output mask.

3. Adapter / projection
   - Maps encoder hidden size to decoder hidden size when needed.

4. Token decoder
   - Reuse the current decoder and generation logic.

### Minimal Interface To Preserve

If you preserve this interface, most of the decoder and training code stays portable:

- `encode(audio, audio_mask) -> memory, memory_mask`
- `decode(tokens, memory, token_mask, memory_mask) -> logits`

The current repo effectively already follows this pattern, except the encoder is tied to mel input.

### Good First Prototype

For the first version:

- Keep the existing tokenization and decoder.
- Use a frozen pretrained encoder.
- Add a single projection layer into decoder dimension.
- Train only the projection + decoder first.

This gives you the cleanest signal about whether the pretrained features help.

### When To Merge Back

Merge the approach into the current repo only if:

- the pretrained encoder clearly improves validation quality or generation quality,
- the dependency burden is acceptable,
- and you want to maintain both encoder paths long-term.

Otherwise keep it separate and let this repo remain the custom-encoder baseline.
