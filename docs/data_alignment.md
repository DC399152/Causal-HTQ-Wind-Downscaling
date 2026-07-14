# Data Alignment

## Core Rule

For every hour start timestamp `T`, the hourly input `X_T` corresponds to exactly these targets:

```text
X_T -> Y_T, Y_{T+10}, Y_{T+20}, Y_{T+30}, Y_{T+40}, Y_{T+50}
```

All timestamps are start timestamps.

## Interval Meaning

```text
X_T      covers [T, T + 60min)
Y_T      covers [T, T + 10min)
Y_T+10   covers [T + 10min, T + 20min)
Y_T+20   covers [T + 20min, T + 30min)
Y_T+30   covers [T + 30min, T + 40min)
Y_T+40   covers [T + 40min, T + 50min)
Y_T+50   covers [T + 50min, T + 60min)
```

## Causal Context

For context length `L`, the input sample for target hour `T` is:

```text
X_{T-L+1:T}
```

With `L = 6`, this is:

```text
X_{T-5h}, X_{T-4h}, X_{T-3h}, X_{T-2h}, X_{T-1h}, X_T
```

This includes the current hourly average `X_T`, because the project setting is hour-end reconstruction. It excludes all hourly profiles after `T`.

## Shape Semantics

Core tensors use time-major semantic axes after batch:

```text
input_context: [B, L, H, C]
target_10min:  [B, T_out, H, C]
```

The project should not use image-style `[B, C, H, T]` as the core representation.

## Validation Checks

The data builder verifies:

- Every target hour has all six 10 min target intervals.
- The six target intervals are exactly 600 s apart.
- The hourly timestamp matches the first target timestamp.
- Context hours are consecutive 3600 s intervals.
- No future hourly timestamp is included in a sample.
- Hourly and target stations are matched by `station_id`, not by array position.
- Paris NC hourly and target height indices are selected independently from their own height coordinates.
- Hourly and target actual heights must be within the configured tolerance.
- A station cannot silently mix incompatible height schemas across files.
- `station_id + target_time_start` must be unique.

## Audit Fields

The core fields remain unchanged, but newer datasets also store audit fields:

```text
context_times_hourly: [N, L]
hourly_height_values: [N, H]
target_height_values: [N, H]
hourly_source_files: [N, L]
target_source_files: [N, T_out]
```

`height_values` is the representative height used by downstream code:

```text
height_values = 0.5 * (hourly_height_values + target_height_values)
```

This is only valid after the hourly/target height difference passes the configured tolerance check.

`source_file` is kept for backward compatibility and is a summary set of the hourly and target source files used by the sample.
