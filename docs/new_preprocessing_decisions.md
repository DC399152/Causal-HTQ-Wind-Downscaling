# New Preprocessing Decisions

This document is the final preprocessing design baseline for the new repository.
`docs/old_repo_reference_needed.md` is only a reference for raw data clues and historical implementation details. If old logic conflicts with this document, this document wins.

Target dataset:

```text
ds_paris_1h_to_10min_6h_causal_start_v1
```

## Non-Negotiable Task Semantics

- Timestamp convention: `timestamp = interval start`.
- Task: hour-end reconstruction, not future forecasting.
- Hourly input `X_T` is the 3600 s average over `[T, T + 1h)`.
- 600 s targets are the six 10 min averages inside the same hour:

```text
Y_T, Y_{T+10}, Y_{T+20}, Y_{T+30}, Y_{T+40}, Y_{T+50}
```

- Input context is causal-last:

```text
X_{T-5h}, X_{T-4h}, X_{T-3h}, X_{T-2h}, X_{T-1h}, X_T
```

- Core dataset arrays use semantic shapes:

```text
X: [N, L, H, C]
Y: [N, T_out, H, C]
```

where `L = 6` and `T_out = 6`.

- Mask convention:

```text
True  = valid
False = invalid
```

- There is no `center_time` concept in the new preprocessing. Each sample is identified by `hour_start`, which is the timestamp of the current hourly input `X_T`. Downscaling is defined only from the last context time `X_T`.

## Raw Data Reading Rules

Raw 3600 s and 600 s NetCDF files are read as paired sources. The expected historical naming pattern is:

```text
*_3600s.nc
*_600s.nc
```

The initial pairing rule is:

1. Scan the configured raw 3600 s directory for `*_3600s.nc`.
2. Scan the configured raw 600 s directory for `*_600s.nc`.
3. Strip the suffix `_3600s.nc` or `_600s.nc`.
4. Pair files only when the stripped prefix is present in both sets.
5. Do not generate samples from unpaired 3600 s or 600 s files.

The reader should load only configured variables. Candidate variable names from old raw files are:

```text
u
v
time
station
altitude
station_lat
station_lon
station_altitude
station_height
flag_suspect_retrieval_warn
flag_suspect_retrieval_removed
flag_low_signal_warn
```

The actual variable names, dimensions, units, and dtypes must be verified with `scripts/inspect_raw_nc.py` before implementation is finalized. The preprocessing config remains the source of truth for variable names after inspection.

Expected raw wind component layout may be:

```text
[station, time, altitude]
```

but this must be verified. If the raw dimension order differs, preprocessing must explicitly transpose into the new semantic order before sample construction.

## 600s / 3600s Alignment Rules

Preprocessing must first merge all paired raw files into station-specific continuous time indexes, then construct samples from that global time index. File boundaries must not define sample boundaries.

For every valid hour start timestamp `T`, construct exactly one current-hour target block:

```text
X_T -> [Y_T, Y_{T+10}, Y_{T+20}, Y_{T+30}, Y_{T+40}, Y_{T+50}]
```

Target offsets from `T` are:

```text
[0, 10, 20, 30, 40, 50] minutes
```

The input context for that same sample is:

```text
[X_{T-5h}, X_{T-4h}, X_{T-3h}, X_{T-2h}, X_{T-1h}, X_T]
```

Validation requirements:

- All six hourly context timestamps must exist.
- All six 600 s target timestamps must exist.
- Context and target timestamps may cross raw file and calendar-day boundaries.
- Hourly context timestamps must be exactly 3600 s apart.
- Target timestamps must be exactly 600 s apart.
- `target_times[0] == hour_start`.
- `context_times[-1] == hour_start`.
- No `X_{T+1h}` or later hourly input may be included.

The old end-aligned target offsets must not be used:

```text
[-50, -40, -30, -20, -10, 0] minutes
```

Implementation decision:

- Do not build samples independently inside each daily raw pair.
- Build a global station series from all paired files first.
- Then slide over every available hourly `T`.
- Example: for `T = 2023-12-30T00:00`, context may include `2023-12-29T19:00` through `2023-12-29T23:00` from the previous file and `X_T` from the current file.

## Height Layer Selection Rules

The new preprocessing uses an explicit configured list of target AGL heights. The Paris v1 candidate from old references is:

```yaml
selected_heights_agl: [250, 275, 300, 325, 350, 375]
height_reference: agl_rounded_station_altitude
max_height_diff: 0.1
```

After inspecting the Paris raw files, the raw `altitude` coordinate is a 25 m grid. Therefore the implemented Paris v1 preprocessing config uses:

```yaml
max_height_diff: 12.5
```

This keeps the nearest-layer rule while allowing up to half a vertical grid step. No vertical interpolation is introduced.

Decision:

1. Treat configured selected heights as AGL target heights unless the config explicitly states otherwise.
2. Convert AGL targets to ASL targets using the station altitude reference:

```text
target_height_asl = station_altitude_reference + target_height_agl
```

3. Select the nearest raw `altitude` layer for each target ASL height.
4. Reject the station/file if any nearest-layer absolute difference exceeds configured `max_height_diff`.
5. Do not vertically interpolate in v1.

Dataset metadata must preserve:

```text
selected_heights_agl
height_indices
target_heights_asl
actual_heights_asl
actual_heights_agl
height_reference
max_height_diff
```

The exact station altitude variable and whether it needs rounding must be verified with `inspect_raw_nc.py`.

## NaN / Flag / Mask Handling Rules

The dataset must carry validity masks separately from values.

Base numeric validity:

```text
valid = isfinite(value) and value != missing_value
```

Default missing sentinel:

```text
missing_value = -999.0
```

Decision:

- Masks use `True = valid`, `False = invalid`.
- `NaN`, `inf`, `-inf`, and `missing_value` are invalid.
- Values may be saved with invalid entries filled by `missing_value`, but the mask is authoritative.
- Missing values must not contribute to normalization statistics, loss masks, or quality summaries.

Output masks:

```text
x_valid_mask: [N, L, H, C]
y_valid_mask: [N, T_out, H, C]
```

QC flag policy for v1:

- QC flags must be inspected before they are used.
- If flags are present and interpretable, preprocessing should convert them into additional invalid positions in the same `True = valid` convention.
- If a flag is percentage-based over an aggregation interval, the config must define the rejection threshold before using it.
- If flag semantics are uncertain, do not silently use or ignore them. Record the uncertainty in metadata and keep the base numeric mask only.

Candidate flags requiring inspection:

```text
flag_suspect_retrieval_warn
flag_suspect_retrieval_removed
flag_low_signal_warn
```

Open questions for flags:

- Are they present in both 3600 s and 600 s files?
- Are values binary, categorical, or percentage occurrence?
- Are dimensions compatible with `[station, time, altitude]`?
- Should warn-level flags invalidate data or only be recorded?
- Should removed-level flags always invalidate data?

## Sample Filtering Rules

The first implementation should expose thresholds in config instead of hard-coding them.

Recommended initial thresholds:

```yaml
min_valid_ratio_x: 0.8
min_valid_ratio_x_per_hour: 0.8
min_valid_ratio_x_current_hour: 1.0
min_valid_ratio_y: 0.8
```

Filtering decisions:

- Drop a sample if total `X` valid ratio is below `min_valid_ratio_x`.
- Drop a sample if any context hour has valid ratio below `min_valid_ratio_x_per_hour`.
- Drop a sample if current hour `X_T` has valid ratio below `min_valid_ratio_x_current_hour`.
- Drop a sample if total `Y` valid ratio is below `min_valid_ratio_y`.
- Filter per sample, not by deleting an entire station unless station-level failure is explicitly configured.

## Dataset Output Keys and Shapes

The final dataset should use these primary keys:

```text
x_context
y_target
x_valid_mask
y_valid_mask
hour_start
context_times
target_times
station_id
station_index
height_indices
selected_heights_agl
target_heights_asl
actual_heights_asl
actual_heights_agl
station_lat
station_lon
station_altitude
station_height
source_3600s_file
source_600s_file
channel_names
timestamp_semantics
context_alignment
target_time_offsets_minutes
missing_value
qc_policy
```

Required shapes:

```text
x_context:    [N, L, H, C]
y_target:     [N, T_out, H, C]
x_valid_mask: [N, L, H, C]
y_valid_mask: [N, T_out, H, C]
```

Required scalar/list semantics:

```text
L = 6
T_out = 6
target_time_offsets_minutes = [0, 10, 20, 30, 40, 50]
timestamp_semantics = "start"
context_alignment = "causal_last"
```

Recommended channels for v1:

```text
channel_names = ["u", "v"]
```

If `ws` and `wd` are used instead of `u` and `v`, that must be an explicit config decision after raw inspection. Do not mix channel definitions silently.

## Train / Val / Test Split Rules

Splits must be chronological and must avoid leakage across the same reconstruction hour.

Decision:

1. Split by unique `hour_start`, not by random sample index.
2. Sort unique `hour_start` values ascending.
3. Apply configured ratios in chronological order.
4. Insert a temporal embargo gap between train/val and val/test.
5. Assign all stations/samples with the same `hour_start` to the same split.
6. Save split files as sample IDs or row indices plus split metadata.

Default ratios:

```yaml
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
split_gap_hours: 24
split_by_unique_time: true
split_time_key: hour_start
```

Do not use `center_time`, because the new dataset has no center-time concept.

Embargo rule:

```text
train | 24h gap | val | 24h gap | test
```

Gap samples remain in `dataset.npz` with `split = "gap"` for auditability, but they must not be used for model training, validation, or testing. This prevents adjacent highly-overlapping samples from crossing split boundaries. The gap must be at least the input context horizon; Paris v1 uses 24 hours as a conservative default.

Normalization statistics must be computed from the train split only and must ignore invalid values according to masks.

## Old Logic Not Migrated

Do not migrate these old preprocessing/model-era assumptions:

- End-aligned timestamp logic.
- Target offsets `[-50, -40, -30, -20, -10, 0]`.
- `center_time` keys, grouping, or naming.
- Centered context windows.
- Any context that includes future hourly inputs after `X_T`.
- Core tensor shape `[C, H, T]` or batched image-style `[B, C, H, T]`.
- Dataset keys `baseline_repeat` and `baseline_linear`.
- Model-specific mask-channel concatenation logic.
- MLP, U-Net, GAN, generator, discriminator, or CNN residual artifacts.
- Silent ignoring of QC flags without documenting whether they exist and what they mean.

Old logic that may be reused only as reference:

- Pairing `*_3600s.nc` and `*_600s.nc` by prefix.
- Candidate raw variables such as `u`, `v`, `time`, `station`, and `altitude`.
- Station metadata variable candidates.
- Nearest-layer height selection idea.
- Mask convention `True = valid`.
- Chronological train/val/test split by unique time.
- Computing normalization statistics from train only.

## Items Requiring `inspect_raw_nc.py` Verification

Before writing the full processing code, verify:

- Actual raw file naming and whether `*_3600s.nc` / `*_600s.nc` pairing is sufficient.
- Exact variable names for wind channels, time, station, height, station coordinates, and station altitude.
- Raw dimension order for wind variables.
- Time variable dtype, units, calendar, timezone assumptions, and whether timestamps are already interval starts.
- Whether 3600 s and 600 s files share the same station and height dimensions.
- Whether altitude is ASL, AGL, or another convention.
- Whether station altitude needs rounding for the Paris height reference.
- Missing value encoding, including `_FillValue`, `missing_value`, NaN, and `-999.0`.
- Existence and semantics of `flag_suspect_retrieval_warn`, `flag_suspect_retrieval_removed`, and `flag_low_signal_warn`.
- Whether QC flags are binary, categorical, or percentage occurrence.
- Shapes of QC flags and whether they align with wind variables.
- Whether `u/v` or `ws/wd` should be the canonical v1 channel pair.

After inspection, update the preprocessing YAML and this document if a raw-data fact contradicts a candidate assumption. The task semantics in this document should not be changed unless the project definition itself changes.
