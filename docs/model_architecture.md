# Model Architecture

## Causal HTQ-Transformer

Causal HTQ-Transformer means **Causal Height-Time Query Transformer**.

The intended route is:

1. Represent hourly wind profiles as height-time tokens.
2. Use a Transformer encoder to model relationships across height and causal hourly context.
3. Construct target 10 min-height queries.
4. Use decoder cross-attention from target queries to encoded hourly context.
5. Predict a zero-mean residual.
6. Return:

```text
pred = current_hourly + residual
```

## Tensor Semantics

Inputs:

```text
X_context: [B, L, H, C]
```

Outputs:

```text
pred:     [B, T_out, H, C]
residual: [B, T_out, H, C]
```

For this dataset, `L = 6` and `T_out = 6`.

## Zero-Mean Residual Constraint

The residual should be zero-mean over the six 10 min target steps:

```text
mean(residual, dim=T_out) = 0
```

This keeps the average of reconstructed 10 min profiles aligned with the current hourly profile.

## Phase 0 Boundary

This document describes the model route only. Phase 0 does not implement a complete Transformer, optimizer, trainer, or model training workflow.
