# Project Brief

## Background

This project studies temporal downscaling for wind profile data. The goal is to reconstruct higher-frequency 10 min vertical wind profiles from lower-frequency hourly wind profiles, while preserving a causal operational setting.

The repository is a clean new implementation for the Causal HTQ-Transformer route. It does not migrate older MLP, U-Net, GAN, generator/discriminator, or CNN residual attempts.

## Task

The target task is **hour-end wind profile reconstruction**.

For an hour indexed by start timestamp `T`:

- `X_T` is the 3600 s hourly average wind profile for `[T, T + 1h)`.
- `Y_T, Y_{T+10}, Y_{T+20}, Y_{T+30}, Y_{T+40}, Y_{T+50}` are the six 600 s averages inside the same hour.

The model input is causal hourly context:

```text
X_{T-L+1:T}
```

where `L` is the number of context hours. For the first target dataset version, `L = 6`.

The model output is:

```text
Y_hat_T, Y_hat_{T+10}, ..., Y_hat_{T+50}
```

with semantic shape:

```text
[B, T_out, H, C]
```

where `T_out = 6`, `H` is the number of height levels, and `C` is the number of profile channels.

## Timestamp Semantics

All timestamps use **start-time semantics**.

This means a timestamp names the beginning of the averaging interval, not the end:

- `X_T` covers `[T, T + 1h)`.
- `Y_{T+20}` covers `[T + 20min, T + 30min)`.

This convention must be preserved in preprocessing, dataset metadata, model documentation, and evaluation.

## Hour-End Reconstruction Setting

Although `X_T` is labeled by the start of the hour, it is only available after `[T, T + 1h)` has completed. The operational setting is therefore:

1. The current hour ends.
2. `X_T` becomes available.
3. The model uses causal hourly context through `X_T`.
4. The model reconstructs the six 10 min profiles that occurred inside `[T, T + 1h)`.

This is reconstruction of sub-hourly structure after observing the completed hourly average, not forecasting future 10 min profiles before the hour occurs.

## Dataset

Target dataset name:

```text
ds_paris_1h_to_10min_6h_causal_start_v1
```

Name components:

- `paris`: site or data source label.
- `1h_to_10min`: temporal downscaling direction.
- `6h`: six-hour causal hourly context.
- `causal`: only current and past hourly profiles are used.
- `start`: timestamps denote interval starts.
- `v1`: first clean dataset specification.
