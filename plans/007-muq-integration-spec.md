## MuQ-First Integration Spec

Goal: build the separate pretrained-encoder repo around MuQ as the first encoder family.

Reference implementation:
- Local clone: `E:\github_repos\MuQ`
- Main wrapper: `E:\github_repos\MuQ\src\muq\muq\muq.py`
- Core model internals: `E:\github_repos\MuQ\src\muq\muq\models\muq_model.py`

### Why MuQ Fits

MuQ is a music-native SSL encoder, not a speech encoder.

From the local clone:
- `MuQ.from_pretrained(...)` is the intended loading path.
- `forward(...)` returns `BaseModelOutput`.
- `last_hidden_state` is `(batch, sequence_length, hidden_size)`.
- `hidden_states` is optionally available.
- The repo explicitly requires `24 kHz` input audio.
- The repo recommends `fp32` during inference to avoid NaNs.

This is a good match for a sequence-to-sequence decoder conditioned on music features.

### Actual MuQ Interface

The local wrapper in `src/muq/muq/muq.py` exposes:

- `MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")`
- `muq(wavs, attention_mask=None, output_hidden_states=True)`

Returned fields:

- `output.last_hidden_state`
- `output.hidden_states`

Relevant observed config defaults from the local code:

- `hop_length = 240`
- `n_mels = 128`
- `encoder_dim = 1024`
- `encoder_depth = 12`
- `label_rate = 25`

### High-Level Repo Design

Use this top-level model structure:

1. waveform loader + resampler
2. MuQ encoder wrapper
3. adapter / projection layer
4. current Beat Weaver style autoregressive token decoder

Keep the encoder contract explicit:

- `encode(waveform, waveform_mask=None) -> memory, memory_mask`

Where:

- `waveform` is `(batch, samples)` at `24,000 Hz`
- `waveform_mask` is sample-level validity if batching variable-length audio
- `memory` is `(batch, time, hidden)`
- `memory_mask` is aligned to MuQ output time

### New Repo Module Plan

Suggested files:

- `model/audio.py`
  - audio loading
  - mono conversion
  - resampling to 24 kHz
  - waveform padding and sample masks

- `model/encoder.py`
  - `MuQEncoderWrapper`
  - MuQ loading
  - hidden-state selection
  - output mask derivation
  - freeze/unfreeze policy

- `model/adapter.py`
  - linear projection from MuQ hidden size to decoder size
  - optional layer norm

- `model/decoder.py`
  - copied/adapted current token decoder

- `model/model.py`
  - composition of encoder + adapter + decoder

### MuQ Wrapper Behavior

The wrapper should:

- load MuQ once using `MuQ.from_pretrained(...)`
- run MuQ in encoder mode only
- expose either:
  - final hidden state only, or
  - a chosen intermediate layer, or
  - a learned weighted combination later

For the first version, use:

- `last_hidden_state`
- no weighted layer mixing
- no hidden-state pooling

Reason:
- simplest path
- preserves time structure
- easiest baseline to debug

### Input Pipeline

Use raw waveform, not mel spectrograms, for the new repo.

Rules:

- resample everything to `24,000 Hz`
- convert to mono
- normalize consistently
- pad waveforms in batch
- carry a sample-level `waveform_mask`

Do not preserve the current Beat Weaver mel-only encoder API internally. That would work against MuQ’s intended usage.

### Output Time Axis And Masks

This is one of the main integration risks.

MuQ internally performs preprocessing and subsampling before its Conformer stack. The wrapper must therefore derive a `memory_mask` aligned to MuQ output length.

First implementation strategy:

- run MuQ on padded waveform batch
- inspect `last_hidden_state.shape[1]` as the output sequence length
- derive output lengths from input valid sample counts using the same batch element ratios
- build a boolean `memory_mask`

Do not assume sample mask length equals MuQ feature length.

This part should be tested carefully because decoder cross-attention masking depends on it.

### Adapter Design

MuQ hidden size in the local code is `1024`.

Your decoder does not need to use that size directly.

First adapter:

- `Linear(1024 -> decoder_dim)`
- optional `LayerNorm`
- optional dropout

Recommended first decoder dim:

- `256` or `512`

That keeps the trainable part manageable while the encoder stays frozen.

### Freezing Policy

First milestone:

- freeze all MuQ parameters
- train only:
  - adapter
  - decoder
  - output head

Second milestone:

- unfreeze only the top MuQ layers or final encoder block(s)
- use a smaller LR for MuQ than for the decoder

Do not start with full MuQ finetuning.

### Numerical Stability Policy

The MuQ README recommends `fp32` for inference due to possible NaNs.

Practical policy:

- keep MuQ forward in `fp32`
- allow adapter + decoder to use mixed precision later if stable
- if AMP is used globally, carve out MuQ forward to run without autocast initially

That is the safest baseline.

### Training Config Additions

Add these config fields:

- `encoder_type = "muq"`
- `muq_model_name`
- `muq_hidden_state_strategy`
- `muq_layer_index`
- `freeze_encoder`
- `unfreeze_last_n_layers`
- `encoder_output_dim`
- `decoder_dim`
- `encoder_lr`
- `decoder_lr`
- `input_sample_rate`
- `target_sample_rate = 24000`
- `max_audio_seconds`

Suggested defaults for v1:

- `muq_model_name = "OpenMuQ/MuQ-large-msd-iter"`
- `muq_hidden_state_strategy = "last"`
- `freeze_encoder = true`
- `unfreeze_last_n_layers = 0`
- `decoder_dim = 256`

### What To Reuse From Beat Weaver

Reuse directly:

- tokenizer design
- token decoder logic
- grammar-constrained inference
- training loop structure
- exporter/evaluator
- dataset and map parsing semantics

Do not reuse directly:

- mel-based `AudioEncoder`
- mel-specific config assumptions
- onset-feature path as part of the encoder input

MuQ already owns the audio feature extraction side.

### Initial Milestones

Milestone 1:

- batch raw waveforms
- load MuQ
- produce `memory` and `memory_mask`
- run decoder forward pass successfully

Milestone 2:

- train adapter + decoder with frozen MuQ
- validate end-to-end token prediction

Milestone 3:

- generate valid maps from checkpoint
- evaluate timing quality and note density behavior

Milestone 4:

- experiment with:
  - different MuQ layers
  - top-layer partial unfreezing
  - decoder size changes

### Primary Risks

- MuQ sequence resolution may still be too coarse for precise Beat Saber placement.
- Long songs may be expensive in VRAM with a 300M encoder.
- The non-commercial weight license may limit downstream use.
- Sample-mask to memory-mask conversion may be easy to get subtly wrong.

### Test Requirements

Minimum tests for the MuQ wrapper:

- resampling path outputs 24 kHz waveform
- wrapper returns `(memory, memory_mask)` with aligned time dimension
- frozen MuQ receives no gradients
- adapter output matches decoder hidden size
- full model forward pass works on a short example
- one-step generation runs without shape or mask errors

### Final Recommendation

For the first MuQ-based repo:

- use raw 24 kHz mono waveform input
- use frozen `MuQ-large-msd-iter`
- use `last_hidden_state`
- add a simple `1024 -> decoder_dim` projection
- keep the existing Beat Weaver decoder and grammar logic

That is the cleanest, lowest-risk way to test whether MuQ features improve map generation.
