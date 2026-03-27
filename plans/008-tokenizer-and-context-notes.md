# Tokenizer And Context Notes

This note summarizes the current tokenizer design, timing representation, handling of simultaneous notes, and the practical limits around sequence length and audio windowing.

## Tokenizer Structure

The tokenizer in `beat_weaver/model/tokenizer.py` uses a fixed 291-token vocabulary:

- `0`: `PAD`
- `1`: `START`
- `2`: `END`
- `3-7`: difficulty tokens
- `8`: `BAR`
- `9-72`: `POS_0 .. POS_63`
- `73`: `LEFT_EMPTY`
- `74-181`: left-hand compound note tokens
- `182`: `RIGHT_EMPTY`
- `183-290`: right-hand compound note tokens

The emitted sequence shape is:

`START -> DIFF_* -> BAR -> POS_* -> LEFT_*|LEFT_EMPTY -> RIGHT_*|RIGHT_EMPTY -> ... -> END`

The tokenizer groups notes by quantized beat location and emits one left slot and one right slot for each occupied position.

## How Notes Are Represented

Each note token stores:

- column `x` in `[0..3]`
- row `y` in `[0..2]`
- cut direction in `[0..8]`

The compound token formula is:

`base + x * 27 + y * 9 + direction`

Where:

- left-hand notes use `LEFT_BASE = 74`
- right-hand notes use `RIGHT_BASE = 183`

This gives `4 * 3 * 9 = 108` note tokens per hand.

## How Beats Are Represented

Beat times are not stored as raw floats. They are quantized to a 1/16-beat grid:

`total_subdivisions = round(beat * 16)`

Then split into:

- `bar_index = total_subdivisions // 64`
- `sub_in_bar = total_subdivisions % 64`

Because each bar is treated as 4 beats and each beat has 16 subdivisions, each bar has 64 `POS_*` slots.

Important consequence:

- `BAR` is not an absolute time token by itself.
- `BAR` only advances the current bar counter.
- The absolute beat is reconstructed from the pair `(current_bar, POS_*)`.

During decode:

`beat = (current_bar * 64 + current_sub) / 16`

Examples:

- first `BAR`, then `POS_0` means absolute beat `0.0`
- first `BAR`, then `POS_16` means absolute beat `1.0`
- second `BAR`, then `POS_16` means absolute beat `5.0`

## How Doubles Are Handled

A "double" in the common Beat Saber sense means one red note and one blue note at the same beat.

The tokenizer supports this directly.

For a single quantized beat position, it emits:

- one `POS_*` token
- one left-hand token or `LEFT_EMPTY`
- one right-hand token or `RIGHT_EMPTY`

So a red+blue same-beat pattern becomes one `POS_*` followed by both hand slots.

Example at beat `0.0`:

- red `(x=1, y=0, dir=1)`
- blue `(x=2, y=0, dir=1)`

Token sequence:

`[START, DIFF_EXPERT, BAR, POS_0, LEFT(1,0,d=1), RIGHT(2,0,d=1), END]`

Actual token ids:

`[1, 6, 8, 9, 102, 238, 2]`

## Same-Hand Collisions

The tokenizer only supports at most one left note and one right note per quantized beat slot.

If two red notes or two blue notes land on the same quantized position:

- the first one is kept
- the later duplicate is dropped
- a debug log message is emitted

This is a representational limit of the current token format.

## What Lets The Model Learn Repeating Patterns

The model can learn repeated map motifs because:

- the decoder is autoregressive
- each next token is predicted from the prior token prefix
- audio is also available through cross-attention
- the token stream has strong regular structure through `BAR` and `POS_*`

So short and medium-range repeated structures can be learned statistically from ground-truth maps.

However, there is no explicit copy mechanism or phrase-level repetition token. Repetition is learned implicitly.

## What Limits `max_seq_len`

`max_seq_len` is a training and inference budget constraint.

In the dataset path:

- tokenized sequences longer than `config.max_seq_len` are truncated
- shorter sequences are padded

So the model is only trained on prefixes up to that limit.

At inference time:

- generation also stops at `config.max_seq_len`

This limit is therefore set mostly by:

- GPU memory
- transformer attention cost
- acceptable training speed
- acceptable inference speed

The shipped configs currently use:

- medium conformer: `max_seq_len = 2048`
- large conformer: `max_seq_len = 4096`

## What Limits `max_audio_len`

`max_audio_len` is a separate cap for the audio side.

After beat alignment, one audio frame corresponds to one 1/16-beat subdivision. In the dataset:

- if the mel spectrogram is longer than `config.max_audio_len`, it is truncated

The code comment states this is done to fit in VRAM.

This limit is driven by:

- encoder memory/time cost
- decoder cross-attention cost against encoder memory
- overall training throughput

The shipped configs currently use:

- medium conformer: `max_audio_len = 4096`
- large conformer: `max_audio_len = 8192`

These correspond to:

- `4096` frames = `256` beats = `64` bars
- `8192` frames = `512` beats = `128` bars

## Why Full-Song Generation Uses Windows

Full songs are not generated as one giant pass because the model is configured around bounded audio context.

If the beat-aligned audio length exceeds `max_audio_len`, `generate_full_song()`:

- splits audio into overlapping windows
- generates tokens independently per window
- decodes notes per window
- shifts note beats by window offset
- merges windows using midpoint ownership in overlap regions

This makes arbitrary song lengths tractable without requiring a giant encoder memory or one extremely long decode pass.

The main reasons for this design are:

- bounded VRAM use
- better practical inference time
- consistency with the training setup, which also uses bounded contexts

The tradeoff is weaker long-range consistency across distant sections of a song.

## Practical Guidance For Replacing The Audio Encoder

If the current Conformer/Transformer audio frontend is replaced with a more efficient encoder, `max_audio_len` can likely be increased.

But the encoder alone is not the whole story. Increasing `max_audio_len` also increases the decoder's cross-attention cost against the encoder output.

So the best improvements usually come from one of these:

- a cheaper encoder at the same temporal resolution
- an encoder that downsamples time before exposing memory to the decoder
- a compressed latent representation for long audio
- a hierarchical or chunked encoder design

In practice, the right `max_audio_len` should be tuned empirically:

1. keep the rest of the pipeline fixed
2. swap in the new encoder
3. measure VRAM, throughput, and generation quality
4. increase `max_audio_len` until the tradeoff becomes poor
5. only then consider increasing `max_seq_len` if longer token context is also needed

## Rough Token-To-Bar Estimates

A rough token count model is:

`tokens ~= 3 + bars * (1 + 3 * occupied_positions_per_bar)`

Where:

- `3` is `START + DIFF + END`
- each bar contributes one `BAR`
- each occupied position contributes `POS + LEFT-slot + RIGHT-slot`

Examples:

- `2048` tokens with about 8 occupied positions per bar gives about 81 bars
- `4096` tokens with about 8 occupied positions per bar gives about 163 bars

The actual bottleneck is often `max_audio_len` rather than `max_seq_len`, especially on sparser maps.
