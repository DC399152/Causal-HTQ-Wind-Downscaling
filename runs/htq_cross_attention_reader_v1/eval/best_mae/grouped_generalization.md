# Grouped Generalization Diagnostic

Checkpoint: `runs\htq_cross_attention_reader_v1\best_mae.pt` (epoch 7)

Strength bins use train-set sample-level RMS thresholds, so train/val/test are compared against the same scale.

## Overall

| split | n | speed mean | residual mean | gradient mean | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 32066 | 8.046 | 1.151 | 1.507 | 0.707 | 0.217 | 0.051 | 0.276 | 0.142 |
| val | 4006 | 10.727 | 1.141 | 1.471 | 0.701 | 0.213 | 0.047 | 0.290 | 0.150 |
| test | 3803 | 10.109 | 1.055 | 1.371 | 0.654 | 0.179 | 0.051 | 0.296 | 0.154 |

## By Station

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | PAARBO | 2033 | 0.787 | 0.172 | 0.026 | 0.255 | 0.126 |
| train | PACHEM | 2993 | 0.874 | 0.180 | 0.036 | 0.243 | 0.119 |
| train | PAJUSS | 8802 | 0.644 | 0.216 | 0.070 | 0.299 | 0.166 |
| train | PALUPD | 4011 | 1.011 | 0.163 | 0.032 | 0.226 | 0.108 |
| train | PAROIS | 5069 | 0.882 | 0.187 | 0.032 | 0.242 | 0.118 |
| train | PASIRT | 9158 | 0.467 | 0.282 | 0.064 | 0.391 | 0.234 |
| val | PAARBO | 254 | 0.676 | 0.197 | 0.020 | 0.285 | 0.147 |
| val | PACHEM | 374 | 0.820 | 0.140 | -0.003 | 0.235 | 0.117 |
| val | PAJUSS | 1100 | 0.629 | 0.217 | 0.079 | 0.351 | 0.207 |
| val | PALUPD | 501 | 1.184 | 0.092 | 0.016 | 0.203 | 0.098 |
| val | PAROIS | 633 | 0.875 | 0.189 | 0.028 | 0.254 | 0.123 |
| val | PASIRT | 1144 | 0.430 | 0.301 | 0.065 | 0.434 | 0.276 |
| test | PAARBO | 238 | 0.940 | 0.079 | 0.041 | 0.231 | 0.111 |
| test | PACHEM | 350 | 0.841 | 0.158 | 0.033 | 0.267 | 0.131 |
| test | PAJUSS | 1053 | 0.563 | 0.193 | 0.069 | 0.347 | 0.201 |
| test | PALUPD | 454 | 0.832 | 0.142 | 0.018 | 0.250 | 0.121 |
| test | PAROIS | 588 | 0.849 | 0.173 | 0.049 | 0.252 | 0.127 |
| test | PASIRT | 1120 | 0.447 | 0.212 | 0.056 | 0.398 | 0.256 |

## By Season

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DJF_winter | 4542 | 0.723 | 0.181 | 0.044 | 0.264 | 0.133 |
| train | JJA_summer | 11944 | 0.740 | 0.212 | 0.048 | 0.260 | 0.132 |
| train | MAM_spring | 5922 | 0.684 | 0.219 | 0.054 | 0.291 | 0.151 |
| train | SON_autumn | 9658 | 0.673 | 0.240 | 0.057 | 0.295 | 0.153 |
| val | DJF_winter | 1841 | 0.792 | 0.168 | 0.033 | 0.253 | 0.128 |
| val | SON_autumn | 2165 | 0.624 | 0.250 | 0.060 | 0.331 | 0.177 |
| test | DJF_winter | 3089 | 0.618 | 0.176 | 0.050 | 0.304 | 0.159 |
| test | MAM_spring | 177 | 0.698 | 0.232 | 0.055 | 0.327 | 0.188 |
| test | SON_autumn | 537 | 0.847 | 0.181 | 0.056 | 0.256 | 0.129 |

## By Calendar Month

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 01 | 1354 | 0.781 | 0.159 | 0.033 | 0.255 | 0.128 |
| train | 02 | 1182 | 0.522 | 0.223 | 0.051 | 0.333 | 0.178 |
| train | 03 | 2116 | 0.754 | 0.219 | 0.057 | 0.278 | 0.142 |
| train | 04 | 1934 | 0.674 | 0.216 | 0.054 | 0.293 | 0.152 |
| train | 05 | 1872 | 0.614 | 0.223 | 0.050 | 0.312 | 0.166 |
| train | 06 | 2914 | 0.709 | 0.215 | 0.057 | 0.273 | 0.141 |
| train | 07 | 4021 | 0.806 | 0.197 | 0.045 | 0.244 | 0.122 |
| train | 08 | 5009 | 0.706 | 0.222 | 0.045 | 0.268 | 0.137 |
| train | 09 | 4560 | 0.621 | 0.248 | 0.053 | 0.311 | 0.165 |
| train | 10 | 3028 | 0.661 | 0.256 | 0.073 | 0.299 | 0.160 |
| train | 11 | 2070 | 0.806 | 0.200 | 0.044 | 0.261 | 0.129 |
| train | 12 | 2006 | 0.804 | 0.172 | 0.047 | 0.249 | 0.123 |
| val | 01 | 722 | 0.876 | 0.157 | 0.011 | 0.245 | 0.121 |
| val | 02 | 144 | 0.876 | 0.081 | -0.018 | 0.211 | 0.102 |
| val | 10 | 1193 | 0.631 | 0.252 | 0.053 | 0.326 | 0.170 |
| val | 11 | 972 | 0.615 | 0.248 | 0.068 | 0.337 | 0.187 |
| val | 12 | 975 | 0.718 | 0.189 | 0.058 | 0.267 | 0.139 |
| test | 01 | 1157 | 0.583 | 0.201 | 0.054 | 0.325 | 0.170 |
| test | 02 | 1651 | 0.651 | 0.165 | 0.049 | 0.292 | 0.152 |
| test | 03 | 177 | 0.698 | 0.232 | 0.055 | 0.327 | 0.188 |
| test | 11 | 537 | 0.847 | 0.181 | 0.056 | 0.256 | 0.129 |
| test | 12 | 281 | 0.566 | 0.136 | 0.036 | 0.291 | 0.168 |

## By Residual Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3207 | 1.813 | 0.120 | 0.029 | 0.182 | 0.090 |
| train | low_le_q50 | 16033 | 0.361 | 0.270 | 0.066 | 0.517 | 0.302 |
| train | mid_q50_q90 | 12826 | 0.865 | 0.176 | 0.039 | 0.284 | 0.145 |
| val | high_gt_q90 | 392 | 1.774 | 0.139 | 0.022 | 0.197 | 0.096 |
| val | low_le_q50 | 2062 | 0.379 | 0.240 | 0.054 | 0.509 | 0.300 |
| val | mid_q50_q90 | 1552 | 0.860 | 0.195 | 0.045 | 0.296 | 0.155 |
| test | high_gt_q90 | 215 | 1.731 | 0.123 | 0.036 | 0.197 | 0.098 |
| test | low_le_q50 | 2030 | 0.391 | 0.197 | 0.053 | 0.479 | 0.274 |
| test | mid_q50_q90 | 1558 | 0.850 | 0.163 | 0.050 | 0.277 | 0.144 |

## By Temporal-Gradient Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3207 | 1.778 | 0.080 | 0.023 | 0.168 | 0.080 |
| train | low_le_q50 | 16033 | 0.365 | 0.304 | 0.068 | 0.510 | 0.338 |
| train | mid_q50_q90 | 12826 | 0.869 | 0.144 | 0.038 | 0.281 | 0.147 |
| val | high_gt_q90 | 363 | 1.764 | 0.085 | 0.017 | 0.177 | 0.083 |
| val | low_le_q50 | 2075 | 0.383 | 0.275 | 0.060 | 0.497 | 0.328 |
| val | mid_q50_q90 | 1568 | 0.877 | 0.159 | 0.037 | 0.296 | 0.159 |
| test | high_gt_q90 | 237 | 1.646 | 0.067 | 0.009 | 0.188 | 0.088 |
| test | low_le_q50 | 2013 | 0.395 | 0.225 | 0.060 | 0.471 | 0.301 |
| test | mid_q50_q90 | 1553 | 0.841 | 0.136 | 0.046 | 0.278 | 0.146 |

## Findings

- Overall train-to-val MAE gap is -0.006 m/s; residual ACC changes from 0.217 to 0.213.
- Validation mean wind speed shifts by +33.3% from train, while mean residual magnitude shifts by -0.8% and temporal-gradient magnitude by -2.4%.
- Validation seasonal composition is DJF_winter=1841, SON_autumn=2165; not every season is represented.
- Highest validation MAE station is `PALUPD` (1.184 m/s). Lowest validation residual ACC station is `PALUPD` (0.092).
- Validation high-residual group: MAE 1.774, residual ACC 0.139.
- Validation high-gradient group: gradient ACC 0.017, gradient RMS ratio 0.083.
- Even on train, residual std ratio is 0.276 and gradient RMS ratio is 0.142. The model is already strongly under-dispersed before considering validation shift.
- A ratio below 1 means predicted residual/gradient variability is too small; a ratio above 1 means it is over-amplified.

The JSON report also contains calendar-month and station-by-month tables for detailed auditing.
