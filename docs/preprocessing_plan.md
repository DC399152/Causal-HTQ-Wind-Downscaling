# Preprocessing Plan

## Goal

Create:

```text
ds_paris_1h_to_10min_6h_causal_start_v1
```

from raw NetCDF wind profile files.

## Step 1: Inspect Raw Files

Use:

```bash
python scripts/inspect_raw_nc.py data/raw/example.nc
```

The script prints dimensions, variables, time range, variable shapes, and missing-value summaries.

Raw variable names are currently unknown. Fill these TODO fields in:

```text
configs/preprocessing/paris_1h_to_10min_6h_causal_start_v1.yaml
```

TODO variables:

- time coordinate name
- height coordinate name
- hourly wind speed variable
- hourly wind direction variable
- 10 min wind speed variable
- 10 min wind direction variable

## Step 2: Standardize Coordinates

Convert raw files into a consistent internal representation:

```text
time_hourly
time_10min
height
hourly_profile: [time_hourly, height, channel]
target_profile: [time_10min, height, channel]
```

All timestamps must be interpreted as start timestamps.

## Step 3: Build Aligned Samples

First merge all paired raw files into station-specific continuous time indexes. Then, for each valid hour start `T`:

1. Select causal hourly context `X_{T-L+1:T}`.
2. Select targets `Y_T, Y_{T+10}, ..., Y_{T+50}`.
3. Check that all context and target timestamps exist.
4. Check missing values according to config.
5. Save sample arrays and metadata.

The context is allowed to cross raw file and calendar-day boundaries. The first hours of a day should be kept when the previous day's hourly context exists.

Output semantic shapes:

```text
X_context: [N, L, H, C]
Y_target:  [N, 6, H, C]
```

## Step 4: Save Dataset Metadata

Metadata should include:

- dataset name
- creation timestamp
- raw file list
- variable mapping
- timestamp semantics
- context length
- target step count
- height levels
- channel names
- split definitions

## Step 5: Chronological Splits With Embargo

Split by unique `target_time_start` in chronological order:

```text
train | gap | val | gap | test
```

Paris v1 uses:

```yaml
split_gap_hours: 24
```

Gap samples are saved with `split = "gap"` and listed in `splits/gap.txt`, but they are excluded from train/val/test use.

## Deferred Work

Phase 0 creates the framework only. Final raw parsing, QC policy, splits, normalization, and artifact format will be implemented after raw `.nc` inspection.
