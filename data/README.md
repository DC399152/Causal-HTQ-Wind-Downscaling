# Data Directory

This repository does not commit raw NetCDF files or generated dataset artifacts.

Expected layout:

```text
data/
  raw/        Original .nc files, ignored by git
  processed/ Intermediate standardized files, ignored by git if large
  datasets/  Generated datasets such as ds_paris_1h_to_10min_6h_causal_start_v1
```

The Phase 0 pipeline assumes timestamp=start semantics:

- `X_T` is an hourly profile averaged over `[T, T + 1h)`.
- `Y_T, Y_{T+10}, ..., Y_{T+50}` are the six 10 min profiles inside that hour.

Run `scripts/inspect_raw_nc.py` on representative raw files before filling variable names in the preprocessing config.
