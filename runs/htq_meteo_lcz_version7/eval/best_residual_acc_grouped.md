# Grouped Generalization Diagnostic

Checkpoint: `runs\htq_meteo_lcz_version7\best_residual_acc.pt` (epoch 15)

Strength bins use train-set sample-level RMS thresholds, so train/val/test are compared against the same scale.

## Overall

| split | n | speed mean | residual mean | gradient mean | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 33161 | 7.967 | 1.161 | 1.515 | 0.706 | 0.242 | 0.084 | 0.277 | 0.143 |
| val | 4142 | 10.529 | 1.170 | 1.501 | 0.716 | 0.224 | 0.064 | 0.287 | 0.151 |
| test | 3893 | 10.024 | 1.079 | 1.398 | 0.667 | 0.193 | 0.056 | 0.291 | 0.152 |

## By Station

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | PAARBO | 2033 | 0.780 | 0.197 | 0.059 | 0.257 | 0.127 |
| train | PACHEM | 2993 | 0.867 | 0.205 | 0.068 | 0.240 | 0.116 |
| train | PAJUSS | 8802 | 0.636 | 0.243 | 0.113 | 0.303 | 0.170 |
| train | PALUPD | 4011 | 1.002 | 0.187 | 0.064 | 0.223 | 0.107 |
| train | PAROIS | 5069 | 0.876 | 0.205 | 0.056 | 0.237 | 0.116 |
| train | PASIRT | 9158 | 0.461 | 0.307 | 0.090 | 0.399 | 0.239 |
| train | dufeng_site_a | 1095 | 0.880 | 0.236 | 0.103 | 0.266 | 0.154 |
| val | PAARBO | 254 | 0.672 | 0.229 | 0.048 | 0.283 | 0.145 |
| val | PACHEM | 374 | 0.819 | 0.138 | -0.009 | 0.239 | 0.119 |
| val | PAJUSS | 1100 | 0.626 | 0.228 | 0.089 | 0.365 | 0.217 |
| val | PALUPD | 501 | 1.177 | 0.102 | 0.036 | 0.185 | 0.091 |
| val | PAROIS | 633 | 0.871 | 0.205 | 0.050 | 0.247 | 0.119 |
| val | PASIRT | 1144 | 0.428 | 0.314 | 0.088 | 0.449 | 0.286 |
| val | dufeng_site_a | 136 | 1.252 | 0.206 | 0.058 | 0.244 | 0.150 |
| test | PAARBO | 238 | 0.938 | 0.088 | 0.032 | 0.206 | 0.100 |
| test | PACHEM | 350 | 0.840 | 0.166 | 0.028 | 0.272 | 0.135 |
| test | PAJUSS | 1053 | 0.561 | 0.215 | 0.093 | 0.345 | 0.198 |
| test | PALUPD | 454 | 0.829 | 0.161 | 0.037 | 0.242 | 0.117 |
| test | PAROIS | 588 | 0.849 | 0.177 | 0.032 | 0.263 | 0.133 |
| test | PASIRT | 1120 | 0.446 | 0.226 | 0.055 | 0.408 | 0.259 |
| test | dufeng_site_a | 90 | 1.291 | 0.160 | 0.073 | 0.237 | 0.133 |

## By Season

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DJF_winter | 4542 | 0.719 | 0.200 | 0.072 | 0.258 | 0.130 |
| train | JJA_summer | 12203 | 0.742 | 0.236 | 0.078 | 0.263 | 0.135 |
| train | MAM_spring | 6758 | 0.694 | 0.239 | 0.086 | 0.285 | 0.152 |
| train | SON_autumn | 9658 | 0.665 | 0.270 | 0.098 | 0.297 | 0.155 |
| val | DJF_winter | 1841 | 0.788 | 0.179 | 0.043 | 0.248 | 0.127 |
| val | JJA_summer | 136 | 1.252 | 0.206 | 0.058 | 0.244 | 0.150 |
| val | SON_autumn | 2165 | 0.622 | 0.264 | 0.082 | 0.338 | 0.181 |
| test | DJF_winter | 3089 | 0.616 | 0.192 | 0.061 | 0.300 | 0.156 |
| test | JJA_summer | 90 | 1.291 | 0.160 | 0.073 | 0.237 | 0.133 |
| test | MAM_spring | 177 | 0.701 | 0.243 | 0.036 | 0.350 | 0.205 |
| test | SON_autumn | 537 | 0.846 | 0.183 | 0.036 | 0.269 | 0.136 |

## By Calendar Month

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 01 | 1354 | 0.775 | 0.184 | 0.074 | 0.251 | 0.128 |
| train | 02 | 1182 | 0.519 | 0.239 | 0.070 | 0.323 | 0.173 |
| train | 03 | 2116 | 0.748 | 0.242 | 0.089 | 0.279 | 0.143 |
| train | 04 | 2200 | 0.681 | 0.231 | 0.081 | 0.294 | 0.159 |
| train | 05 | 2442 | 0.658 | 0.243 | 0.088 | 0.283 | 0.155 |
| train | 06 | 3173 | 0.738 | 0.238 | 0.086 | 0.268 | 0.142 |
| train | 07 | 4021 | 0.799 | 0.222 | 0.072 | 0.243 | 0.123 |
| train | 08 | 5009 | 0.698 | 0.246 | 0.077 | 0.279 | 0.141 |
| train | 09 | 4560 | 0.613 | 0.278 | 0.097 | 0.314 | 0.167 |
| train | 10 | 3028 | 0.651 | 0.289 | 0.117 | 0.302 | 0.161 |
| train | 11 | 2070 | 0.799 | 0.226 | 0.071 | 0.262 | 0.129 |
| train | 12 | 2006 | 0.799 | 0.187 | 0.071 | 0.242 | 0.120 |
| val | 01 | 722 | 0.873 | 0.169 | 0.025 | 0.240 | 0.119 |
| val | 02 | 144 | 0.875 | 0.083 | -0.037 | 0.201 | 0.098 |
| val | 06 | 136 | 1.252 | 0.206 | 0.058 | 0.244 | 0.150 |
| val | 10 | 1193 | 0.628 | 0.267 | 0.079 | 0.324 | 0.169 |
| val | 11 | 972 | 0.615 | 0.261 | 0.085 | 0.355 | 0.197 |
| val | 12 | 975 | 0.712 | 0.200 | 0.068 | 0.263 | 0.139 |
| test | 01 | 1157 | 0.581 | 0.217 | 0.077 | 0.324 | 0.169 |
| test | 02 | 1651 | 0.648 | 0.179 | 0.052 | 0.285 | 0.147 |
| test | 03 | 177 | 0.701 | 0.243 | 0.036 | 0.350 | 0.205 |
| test | 06 | 90 | 1.291 | 0.160 | 0.073 | 0.237 | 0.133 |
| test | 11 | 537 | 0.846 | 0.183 | 0.036 | 0.269 | 0.136 |
| test | 12 | 281 | 0.566 | 0.170 | 0.045 | 0.288 | 0.162 |

## By Residual Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.816 | 0.149 | 0.069 | 0.199 | 0.103 |
| train | low_le_q50 | 16581 | 0.360 | 0.292 | 0.098 | 0.501 | 0.288 |
| train | mid_q50_q90 | 13264 | 0.864 | 0.201 | 0.071 | 0.279 | 0.144 |
| val | high_gt_q90 | 421 | 1.840 | 0.154 | 0.058 | 0.215 | 0.113 |
| val | low_le_q50 | 2113 | 0.379 | 0.255 | 0.074 | 0.493 | 0.286 |
| val | mid_q50_q90 | 1608 | 0.868 | 0.202 | 0.052 | 0.293 | 0.154 |
| test | high_gt_q90 | 235 | 1.841 | 0.134 | 0.036 | 0.227 | 0.119 |
| test | low_le_q50 | 2066 | 0.390 | 0.211 | 0.062 | 0.465 | 0.262 |
| test | mid_q50_q90 | 1592 | 0.855 | 0.177 | 0.052 | 0.271 | 0.141 |

## By Temporal-Gradient Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.785 | 0.107 | 0.064 | 0.175 | 0.086 |
| train | low_le_q50 | 16581 | 0.364 | 0.327 | 0.102 | 0.502 | 0.329 |
| train | mid_q50_q90 | 13264 | 0.866 | 0.169 | 0.068 | 0.283 | 0.150 |
| val | high_gt_q90 | 395 | 1.833 | 0.098 | 0.038 | 0.178 | 0.090 |
| val | low_le_q50 | 2126 | 0.383 | 0.289 | 0.077 | 0.480 | 0.312 |
| val | mid_q50_q90 | 1621 | 0.883 | 0.170 | 0.053 | 0.310 | 0.169 |
| test | high_gt_q90 | 256 | 1.757 | 0.082 | 0.019 | 0.203 | 0.100 |
| test | low_le_q50 | 2041 | 0.394 | 0.239 | 0.065 | 0.461 | 0.290 |
| test | mid_q50_q90 | 1596 | 0.843 | 0.151 | 0.051 | 0.281 | 0.149 |

## Findings

- Overall train-to-val MAE gap is 0.010 m/s; residual ACC changes from 0.242 to 0.224.
- Validation mean wind speed shifts by +32.2% from train, while mean residual magnitude shifts by +0.8% and temporal-gradient magnitude by -1.0%.
- Validation seasonal composition is DJF_winter=1841, JJA_summer=136, SON_autumn=2165; not every season is represented.
- Highest validation MAE station is `dufeng_site_a` (1.252 m/s). Lowest validation residual ACC station is `PALUPD` (0.102).
- Validation high-residual group: MAE 1.840, residual ACC 0.154.
- Validation high-gradient group: gradient ACC 0.038, gradient RMS ratio 0.090.
- Even on train, residual std ratio is 0.277 and gradient RMS ratio is 0.143. The model is already strongly under-dispersed before considering validation shift.
- A ratio below 1 means predicted residual/gradient variability is too small; a ratio above 1 means it is over-amplified.

The JSON report also contains calendar-month and station-by-month tables for detailed auditing.
