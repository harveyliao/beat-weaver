# MuQ Frame Alignment Experiments

## Problem

MuQ outputs features at a fixed 25Hz (one frame every 40ms). Our current pipeline produces one feature vector per 1/16th note — a variable-rate representation that depends on BPM.

At typical Beat Saber BPMs, there is less than one MuQ frame per 1/16th note:

| BPM | 1/16th note duration | MuQ frames per 1/16th note |
|-----|----------------------|----------------------------|
| 100 | 37.5ms               | 0.94                       |
| 120 | 31.25ms              | 0.78                       |
| 150 | 25.0ms               | 0.625                      |
| 180 | 20.8ms               | 0.52                       |

The decoder, tokenizer, and training loop all assume beat-aligned audio frames. The alignment strategy we choose determines how much of the existing architecture we preserve and how much information we retain from MuQ.

## Chosen approach: Option A first, then B if needed

### Option A: Interpolate MuQ features onto the beat grid (first experiment)

Resample the 25Hz MuQ feature sequence to the 1/16th note grid using linear interpolation in feature space.

For each beat-grid position at time `t_beat`:
1. Compute `t_beat = (bar * 64 + sub) / 16 * 60.0 / BPM` (seconds)
2. Find the two surrounding MuQ frames: `idx = t_beat * 25.0`, `lo = floor(idx)`, `hi = lo + 1`
3. Lerp: `feature = muq[lo] * (1 - frac) + muq[hi] * frac`

**Why start here:**
- Preserves the entire decoder architecture unchanged — cross-attention, masking, RoPE, batching all stay the same.
- Isolates the experimental variable to "better audio features" only. If performance changes, we know it's MuQ, not the alignment change.
- Can be done entirely in the pre-caching step. Cache beat-aligned MuQ features the same way we cache mel spectrograms now. Training code barely changes.
- MuQ features are smooth and slowly-varying at 25Hz — neighboring frames are highly correlated, so interpolation is well-behaved.

**Implementation sketch:**
- Modify `warm_mel_cache()` (or add a parallel `warm_muq_cache()`) that:
  1. Loads audio at 24kHz (MuQ requirement, vs current 22.05kHz)
  2. Runs frozen MuQ forward pass in fp32
  3. Extracts `last_hidden_state` → `(1, T_muq, 1024)`
  4. Reads BPM from the dataset metadata
  5. Computes the 1/16th note time grid
  6. Interpolates MuQ features onto that grid → `(T_beats, 1024)`
  7. Saves as `.npy` in a separate cache directory (e.g. `data/processed/muq_cache/`)
- The dataset loader reads from `muq_cache/` instead of `mel_cache/`
- The adapter/projection layer maps 1024 → decoder_dim

**What to measure:**
- Val loss and token accuracy vs. current best (59.4% on medium conformer)
- Training speed (should be comparable since features are pre-cached)
- Generation quality: onset F1, NPS accuracy, parity violations
- Sanity check: listen to generated maps in-game

**Known tradeoffs:**
- At high BPM, we're upsampling — duplicating information, not creating it
- At low BPM, we're downsampling — averaging over multiple MuQ frames per beat slot
- Feature-space interpolation assumes linearity in the representation, which is approximate but reasonable for smooth SSL features

### Option B: Feed native 25Hz with beat-position embeddings (second experiment)

If Option A works well and we want to push further, try feeding MuQ features at their native 25Hz rate with an added positional signal.

**Design:**
- Pre-cache raw MuQ features at 25Hz without resampling
- For each MuQ frame at time `t` seconds, compute its beat position: `beat_pos = t * BPM / 60.0`
- Encode `beat_pos` as a sinusoidal or learned positional embedding
- Add the beat-position embedding to each MuQ feature vector before the adapter
- The decoder cross-attends to these 25Hz features (no forced grid alignment)

**Why this could be better than A:**
- Zero information loss — decoder sees exactly what MuQ computed
- The beat-position embedding gives cross-attention a musical-time signal without forcing a rigid grid
- More natural for the attention mechanism to learn soft alignment

**Why it's riskier:**
- Sequence lengths scale with wall-clock duration, not musical structure. A 4-min song at 25Hz = 6000 frames. Cross-attention cost grows.
- The decoder currently assumes beat-aligned input for batching — variable-length 25Hz sequences need padding/masking changes
- More moving parts to debug if something doesn't work
- Need to rethink `max_audio_len` semantics (currently in beat frames, would become wall-clock frames)

**What changes vs. Option A:**
- Cache format: raw 25Hz features, no interpolation
- Dataset loader: returns variable-length sequences + mask
- Adapter: includes beat-position embedding addition
- Config: `max_audio_len` reinterpreted as 25Hz frames (e.g. 6000 = 4 min)
- Batching: need proper padding and memory masking for variable lengths

### Option C: Pool MuQ frames into beat-aligned windows (backup)

If Option A's interpolation artifacts are a problem, try weighted average pooling instead.

For each 1/16th note interval `[t_start, t_end)`:
- Gather all MuQ frames whose centers fall within the interval
- If none fall within (high BPM), take the nearest frame
- If multiple fall within (low BPM), weighted average by overlap fraction

This is more principled than lerp (proper downsampling with anti-aliasing) but produces the same output shape as Option A, so the rest of the pipeline stays identical.

Only pursue this if Option A shows artifacts attributable to interpolation (unlikely given the smoothness of SSL features).

## Experiment order

1. **Option A** — interpolate to beat grid, frozen MuQ, projection + decoder training
2. If A matches or beats current 59.4%: try unfreezing top 2-4 MuQ layers with low LR
3. If A plateaus and we suspect alignment is the bottleneck: **Option B** — native 25Hz + beat embeddings
4. **Option C** only if interpolation artifacts are specifically identified

## Shared prerequisites (all options)

- Audio loading path that resamples to 24kHz mono (MuQ requirement)
- MuQ installed as dependency (`pip install muq` or local editable install from `../MuQ`)
- Pre-caching infrastructure for MuQ features (parallel, versioned, like existing mel cache)
- Projection layer: `Linear(1024, decoder_dim)` + optional LayerNorm
- MuQ runs in fp32 during caching (recommended by MuQ authors to avoid NaNs)
- Config additions: `encoder_type`, `muq_model_name`, `freeze_encoder`, etc. (see plan 007)
