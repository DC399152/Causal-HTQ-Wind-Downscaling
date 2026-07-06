# Task Definition

## Inputs

Input tensor semantics:

```text
X_context: [B, L, H, C]
```

where:

- `B`: batch size
- `L`: causal context length in hours
- `H`: height levels
- `C`: input profile channels

For sample hour `T`, the context is:

```text
X_{T-L+1}, ..., X_{T-1}, X_T
```

No future hourly profiles after `T` are allowed.

## Targets

Target tensor semantics:

```text
Y_target: [B, T_out, H, C]
```

where `T_out = 6` for the six 10 min intervals in the current hour:

```text
Y_T, Y_{T+10}, Y_{T+20}, Y_{T+30}, Y_{T+40}, Y_{T+50}
```

## Prediction Form

The Causal HTQ-Transformer predicts a zero-mean residual around the current hourly profile:

```text
pred = current_hourly + residual
```

The residual should average to zero across the six 10 min target steps for each height and channel, so that the reconstructed 10 min profiles remain consistent with the current hourly profile.

## Deferred Items

Phase 0 does not implement training, full model internals, or final losses. These are documented as interfaces and TODOs only.
