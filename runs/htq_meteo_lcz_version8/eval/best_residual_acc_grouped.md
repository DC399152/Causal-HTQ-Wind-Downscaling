# Grouped Generalization Diagnostic

Checkpoint: `runs\htq_meteo_lcz_version8\best_residual_acc.pt` (epoch 14)

Strength bins use train-set sample-level RMS thresholds, so train/val/test are compared against the same scale.

## Overall

| split | n | speed mean | residual mean | gradient mean | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 33161 | 7.967 | 1.161 | 1.515 | 0.717 | 0.236 | 0.079 | 0.353 | 0.176 |
| val | 4142 | 10.529 | 1.170 | 1.501 | 0.729 | 0.225 | 0.058 | 0.379 | 0.193 |
| test | 3893 | 10.024 | 1.079 | 1.398 | 0.679 | 0.190 | 0.059 | 0.375 | 0.189 |

## By Station

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | PAARBO | 2033 | 0.791 | 0.189 | 0.054 | 0.331 | 0.158 |
| train | PACHEM | 2993 | 0.879 | 0.199 | 0.059 | 0.327 | 0.155 |
| train | PAJUSS | 8802 | 0.648 | 0.238 | 0.106 | 0.374 | 0.202 |
| train | PALUPD | 4011 | 1.014 | 0.181 | 0.058 | 0.294 | 0.137 |
| train | PAROIS | 5069 | 0.886 | 0.201 | 0.057 | 0.311 | 0.147 |
| train | PASIRT | 9158 | 0.470 | 0.301 | 0.085 | 0.495 | 0.284 |
| train | dufeng_site_a | 1095 | 0.892 | 0.228 | 0.082 | 0.331 | 0.184 |
| val | PAARBO | 254 | 0.683 | 0.207 | 0.027 | 0.371 | 0.186 |
| val | PACHEM | 374 | 0.831 | 0.142 | 0.003 | 0.316 | 0.151 |
| val | PAJUSS | 1100 | 0.639 | 0.232 | 0.080 | 0.457 | 0.262 |
| val | PALUPD | 501 | 1.193 | 0.097 | 0.025 | 0.270 | 0.127 |
| val | PAROIS | 633 | 0.888 | 0.196 | 0.031 | 0.354 | 0.167 |
| val | PASIRT | 1144 | 0.438 | 0.322 | 0.088 | 0.575 | 0.351 |
| val | dufeng_site_a | 136 | 1.254 | 0.215 | 0.095 | 0.310 | 0.180 |
| test | PAARBO | 238 | 0.959 | 0.077 | 0.039 | 0.312 | 0.143 |
| test | PACHEM | 350 | 0.848 | 0.161 | 0.017 | 0.342 | 0.163 |
| test | PAJUSS | 1053 | 0.573 | 0.216 | 0.104 | 0.443 | 0.246 |
| test | PALUPD | 454 | 0.838 | 0.155 | 0.031 | 0.315 | 0.148 |
| test | PAROIS | 588 | 0.858 | 0.179 | 0.032 | 0.335 | 0.164 |
| test | PASIRT | 1120 | 0.460 | 0.221 | 0.058 | 0.534 | 0.324 |
| test | dufeng_site_a | 90 | 1.316 | 0.182 | 0.062 | 0.265 | 0.141 |

## By Season

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DJF_winter | 4542 | 0.729 | 0.196 | 0.063 | 0.345 | 0.166 |
| train | JJA_summer | 12203 | 0.752 | 0.230 | 0.074 | 0.329 | 0.163 |
| train | MAM_spring | 6758 | 0.702 | 0.236 | 0.082 | 0.354 | 0.180 |
| train | SON_autumn | 9658 | 0.677 | 0.263 | 0.090 | 0.387 | 0.196 |
| val | DJF_winter | 1841 | 0.802 | 0.177 | 0.038 | 0.331 | 0.162 |
| val | JJA_summer | 136 | 1.254 | 0.215 | 0.095 | 0.310 | 0.180 |
| val | SON_autumn | 2165 | 0.634 | 0.266 | 0.074 | 0.446 | 0.231 |
| test | DJF_winter | 3089 | 0.629 | 0.188 | 0.064 | 0.399 | 0.200 |
| test | JJA_summer | 90 | 1.316 | 0.182 | 0.062 | 0.265 | 0.141 |
| test | MAM_spring | 177 | 0.697 | 0.249 | 0.043 | 0.390 | 0.215 |
| test | SON_autumn | 537 | 0.855 | 0.185 | 0.034 | 0.340 | 0.166 |

## By Calendar Month

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 01 | 1354 | 0.787 | 0.176 | 0.067 | 0.343 | 0.166 |
| train | 02 | 1182 | 0.522 | 0.244 | 0.065 | 0.384 | 0.195 |
| train | 03 | 2116 | 0.760 | 0.238 | 0.078 | 0.363 | 0.179 |
| train | 04 | 2200 | 0.688 | 0.229 | 0.075 | 0.346 | 0.177 |
| train | 05 | 2442 | 0.666 | 0.240 | 0.093 | 0.352 | 0.185 |
| train | 06 | 3173 | 0.748 | 0.229 | 0.078 | 0.334 | 0.171 |
| train | 07 | 4021 | 0.809 | 0.216 | 0.071 | 0.311 | 0.151 |
| train | 08 | 5009 | 0.708 | 0.243 | 0.074 | 0.343 | 0.170 |
| train | 09 | 4560 | 0.624 | 0.272 | 0.089 | 0.399 | 0.205 |
| train | 10 | 3028 | 0.667 | 0.280 | 0.107 | 0.399 | 0.208 |
| train | 11 | 2070 | 0.809 | 0.219 | 0.065 | 0.349 | 0.167 |
| train | 12 | 2006 | 0.811 | 0.181 | 0.060 | 0.334 | 0.159 |
| val | 01 | 722 | 0.886 | 0.157 | 0.017 | 0.321 | 0.154 |
| val | 02 | 144 | 0.888 | 0.096 | -0.021 | 0.290 | 0.132 |
| val | 06 | 136 | 1.254 | 0.215 | 0.095 | 0.310 | 0.180 |
| val | 10 | 1193 | 0.644 | 0.268 | 0.070 | 0.445 | 0.225 |
| val | 11 | 972 | 0.622 | 0.263 | 0.079 | 0.446 | 0.241 |
| val | 12 | 975 | 0.727 | 0.203 | 0.061 | 0.347 | 0.176 |
| test | 01 | 1157 | 0.592 | 0.210 | 0.079 | 0.413 | 0.211 |
| test | 02 | 1651 | 0.664 | 0.177 | 0.054 | 0.388 | 0.192 |
| test | 03 | 177 | 0.697 | 0.249 | 0.043 | 0.390 | 0.215 |
| test | 06 | 90 | 1.316 | 0.182 | 0.062 | 0.265 | 0.141 |
| test | 11 | 537 | 0.855 | 0.185 | 0.034 | 0.340 | 0.166 |
| test | 12 | 281 | 0.578 | 0.163 | 0.058 | 0.416 | 0.222 |

## By Residual Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.823 | 0.148 | 0.066 | 0.237 | 0.116 |
| train | low_le_q50 | 16581 | 0.374 | 0.287 | 0.091 | 0.640 | 0.355 |
| train | mid_q50_q90 | 13264 | 0.871 | 0.195 | 0.067 | 0.369 | 0.185 |
| val | high_gt_q90 | 421 | 1.848 | 0.152 | 0.055 | 0.268 | 0.133 |
| val | low_le_q50 | 2113 | 0.396 | 0.256 | 0.066 | 0.656 | 0.369 |
| val | mid_q50_q90 | 1608 | 0.875 | 0.202 | 0.050 | 0.398 | 0.203 |
| test | high_gt_q90 | 235 | 1.848 | 0.144 | 0.021 | 0.250 | 0.126 |
| test | low_le_q50 | 2066 | 0.408 | 0.210 | 0.068 | 0.617 | 0.335 |
| test | mid_q50_q90 | 1592 | 0.862 | 0.171 | 0.052 | 0.364 | 0.182 |

## By Temporal-Gradient Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.797 | 0.104 | 0.057 | 0.223 | 0.105 |
| train | low_le_q50 | 16581 | 0.375 | 0.322 | 0.095 | 0.633 | 0.401 |
| train | mid_q50_q90 | 13264 | 0.876 | 0.163 | 0.064 | 0.364 | 0.187 |
| val | high_gt_q90 | 395 | 1.848 | 0.099 | 0.043 | 0.248 | 0.117 |
| val | low_le_q50 | 2126 | 0.396 | 0.290 | 0.072 | 0.642 | 0.404 |
| val | mid_q50_q90 | 1621 | 0.894 | 0.170 | 0.044 | 0.398 | 0.210 |
| test | high_gt_q90 | 256 | 1.776 | 0.082 | 0.010 | 0.244 | 0.115 |
| test | low_le_q50 | 2041 | 0.409 | 0.236 | 0.067 | 0.611 | 0.369 |
| test | mid_q50_q90 | 1596 | 0.852 | 0.149 | 0.056 | 0.362 | 0.185 |

## Findings

- Overall train-to-val MAE gap is 0.012 m/s; residual ACC changes from 0.236 to 0.225.
- Validation mean wind speed shifts by +32.2% from train, while mean residual magnitude shifts by +0.8% and temporal-gradient magnitude by -1.0%.
- Validation seasonal composition is DJF_winter=1841, JJA_summer=136, SON_autumn=2165; not every season is represented.
- Highest validation MAE station is `dufeng_site_a` (1.254 m/s). Lowest validation residual ACC station is `PALUPD` (0.097).
- Validation high-residual group: MAE 1.848, residual ACC 0.152.
- Validation high-gradient group: gradient ACC 0.043, gradient RMS ratio 0.117.
- Even on train, residual std ratio is 0.353 and gradient RMS ratio is 0.176. The model is already strongly under-dispersed before considering validation shift.
- A ratio below 1 means predicted residual/gradient variability is too small; a ratio above 1 means it is over-amplified.

The JSON report also contains calendar-month and station-by-month tables for detailed auditing.
