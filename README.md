# Causal HTQ Wind Downscaling

This is a new, clean research repository for **Causal HTQ-Transformer wind profile temporal downscaling**.

The target task is hour-end wind profile reconstruction:

- Input hourly profiles `X_T` are 3600 s averages over `[T, T + 1h)`.
- Target 10 min profiles are `Y_T, Y_{T+10}, ..., Y_{T+50}`, the six consecutive 600 s averages inside the same hour.
- Timestamps use **start-time semantics**.
- After the hour ends, `X_T` is available. The model uses causal hourly context `X_{T-L+1:T}` to reconstruct the six 10 min profiles inside the current hour.

Target dataset name:

```text
ds_paris_1h_to_10min_6h_causal_start_v1
```

## Clean-Room Scope

This repository intentionally does **not** migrate historical model attempts from older repositories.

Not included:

- MLP baselines
- U-Net models
- GAN, generator, or discriminator models
- CNN residual models
- Image-style core tensor logic such as `[B, C, H, T]`

The new implementation will focus on:

- A reproducible data generation pipeline
- Explicit timestamp alignment rules
- Semantic tensor shapes such as `[B, L, H, C]` and `[B, T_out, H, C]`
- A new Causal Height-Time Query Transformer architecture

## Repository Layout

```text
configs/        YAML configs for preprocessing and HTQ experiments
data/           Raw, processed, and dataset artifact locations
docs/           Project notes, task definition, alignment, and architecture docs
scripts/        CLI entry points for inspection, dataset building, training, and evaluation
src/            Source package
tests/          Lightweight tests for alignment, shapes, and model constraints
```

## Phase 0 Status

Phase 0 creates the project skeleton, documentation, and data pipeline framework only.
Model training and a complete Transformer implementation are intentionally deferred.
