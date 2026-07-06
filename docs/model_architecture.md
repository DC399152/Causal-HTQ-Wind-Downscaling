# Model Architecture

## Causal HTQ-Transformer

Causal HTQ-Transformer means **Causal Height-Time Query Transformer**.

The intended route is:

1. Represent hourly wind profiles as height-time tokens.
2. Use a Transformer encoder to model relationships across height and causal hourly context.
3. Construct target 10 min-height queries.
4. Use decoder cross-attention from target queries to encoded hourly context.
5. Predict a residual correction.
6. Return:

```text
pred = current_hourly + residual
```

## Tensor Semantics

Inputs:

```text
X_context: [B, L, H, C]
X_mask:    [B, L, H, C]
```

Outputs:

```text
pred:     [B, T_out, H, C]
residual: [B, T_out, H, C]
```

For this dataset, `L = 6` and `T_out = 6`.

## Mask Semantics

Preprocessing and Dataset normalization use this convention:

```text
True  = valid
False = invalid
```

Invalid `X_context` values are filled with `0.0` after normalization. This is only a missing-value placeholder in normalized space; it is not a physical zero wind component. The model must receive `X_mask` and must not interpret masked placeholders as observed values.

The model forward interface is:

```text
forward(x_hourly, x_mask)
```

where:

```text
x_hourly: [B, L, H, C]
x_mask:   [B, L, H, C]
```

The tokenizer should construct:

```text
token_valid = any(x_mask, dim=channel)  # [B, L, H]
delta[:, 0] = 0
delta[:, t] = x_hourly[:, t] - x_hourly[:, t-1]
token_features = concat([x_hourly, delta, x_mask.float()], dim=channel)
```

This gives the Transformer normalized values, hourly changes, and explicit validity information.

Target masks are used in the loss:

```text
loss = lambda_l1 * masked_l1_loss(pred, y_10min, y_mask)
     + lambda_temporal * temporal_gradient_loss(pred, y_10min, y_mask)
     + lambda_vertical * vertical_gradient_loss(pred, y_10min, y_mask)
```

Invalid target values must not contribute to loss or metrics.

## Residual Constraint

The current implementation does not hard-constrain residuals to be zero-mean over the six 10 min target steps.

```text
pred = current_hourly + residual
```

This is intentionally less rigid than forcing `mean(residual, dim=T_out) = 0`, because real 600s and 3600s products may differ due to missing values, QC, or aggregation details. Validation and test metrics are still computed after denormalizing predictions and targets to physical m/s units.

## Phase 0 Boundary

This document describes the model route only. Phase 0 does not implement a complete Transformer, optimizer, trainer, or model training workflow.
