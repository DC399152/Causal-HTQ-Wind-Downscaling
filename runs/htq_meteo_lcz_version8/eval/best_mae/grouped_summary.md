# Grouped Generalization Diagnostic

Checkpoint: `runs\htq_meteo_lcz_version8\best_mae.pt` (epoch 4)

Strength bins use train-set sample-level RMS thresholds, so train/val/test are compared against the same scale.

## Overall

| split | n | speed mean | residual mean | gradient mean | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 33161 | 7.967 | 1.161 | 1.515 | 0.714 | 0.220 | 0.057 | 0.214 | 0.111 |
| val | 4142 | 10.529 | 1.170 | 1.501 | 0.718 | 0.210 | 0.057 | 0.227 | 0.119 |
| test | 3893 | 10.024 | 1.079 | 1.398 | 0.668 | 0.184 | 0.054 | 0.231 | 0.121 |

## By Station

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | PAARBO | 2033 | 0.787 | 0.171 | 0.034 | 0.201 | 0.099 |
| train | PACHEM | 2993 | 0.875 | 0.181 | 0.037 | 0.195 | 0.096 |
| train | PAJUSS | 8802 | 0.643 | 0.219 | 0.085 | 0.228 | 0.127 |
| train | PALUPD | 4011 | 1.011 | 0.161 | 0.030 | 0.179 | 0.086 |
| train | PAROIS | 5069 | 0.882 | 0.189 | 0.035 | 0.189 | 0.093 |
| train | PASIRT | 9158 | 0.470 | 0.288 | 0.067 | 0.302 | 0.180 |
| train | dufeng_site_a | 1095 | 0.890 | 0.212 | 0.055 | 0.201 | 0.112 |
| val | PAARBO | 254 | 0.676 | 0.213 | 0.040 | 0.232 | 0.120 |
| val | PACHEM | 374 | 0.817 | 0.137 | -0.001 | 0.194 | 0.096 |
| val | PAJUSS | 1100 | 0.628 | 0.213 | 0.083 | 0.274 | 0.163 |
| val | PALUPD | 501 | 1.179 | 0.095 | 0.025 | 0.169 | 0.082 |
| val | PAROIS | 633 | 0.873 | 0.194 | 0.045 | 0.204 | 0.100 |
| val | PASIRT | 1144 | 0.432 | 0.294 | 0.073 | 0.346 | 0.220 |
| val | dufeng_site_a | 136 | 1.246 | 0.172 | 0.077 | 0.180 | 0.103 |
| test | PAARBO | 238 | 0.937 | 0.085 | 0.048 | 0.191 | 0.093 |
| test | PACHEM | 350 | 0.833 | 0.164 | 0.042 | 0.205 | 0.102 |
| test | PAJUSS | 1053 | 0.560 | 0.206 | 0.095 | 0.277 | 0.161 |
| test | PALUPD | 454 | 0.831 | 0.145 | 0.011 | 0.203 | 0.098 |
| test | PAROIS | 588 | 0.851 | 0.175 | 0.035 | 0.202 | 0.103 |
| test | PASIRT | 1120 | 0.447 | 0.213 | 0.045 | 0.325 | 0.208 |
| test | dufeng_site_a | 90 | 1.304 | 0.164 | 0.082 | 0.165 | 0.087 |

## By Season

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | DJF_winter | 4542 | 0.724 | 0.178 | 0.049 | 0.211 | 0.106 |
| train | JJA_summer | 12203 | 0.750 | 0.215 | 0.052 | 0.202 | 0.103 |
| train | MAM_spring | 6758 | 0.700 | 0.222 | 0.061 | 0.222 | 0.117 |
| train | SON_autumn | 9658 | 0.674 | 0.245 | 0.064 | 0.227 | 0.119 |
| val | DJF_winter | 1841 | 0.789 | 0.173 | 0.041 | 0.204 | 0.103 |
| val | JJA_summer | 136 | 1.246 | 0.172 | 0.077 | 0.180 | 0.103 |
| val | SON_autumn | 2165 | 0.625 | 0.244 | 0.069 | 0.264 | 0.143 |
| test | DJF_winter | 3089 | 0.616 | 0.183 | 0.057 | 0.246 | 0.129 |
| test | JJA_summer | 90 | 1.304 | 0.164 | 0.082 | 0.165 | 0.087 |
| test | MAM_spring | 177 | 0.695 | 0.237 | 0.040 | 0.243 | 0.143 |
| test | SON_autumn | 537 | 0.849 | 0.179 | 0.039 | 0.205 | 0.104 |

## By Calendar Month

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 01 | 1354 | 0.781 | 0.162 | 0.056 | 0.205 | 0.103 |
| train | 02 | 1182 | 0.522 | 0.222 | 0.053 | 0.246 | 0.130 |
| train | 03 | 2116 | 0.756 | 0.223 | 0.055 | 0.225 | 0.116 |
| train | 04 | 2200 | 0.687 | 0.220 | 0.062 | 0.221 | 0.117 |
| train | 05 | 2442 | 0.665 | 0.222 | 0.065 | 0.219 | 0.118 |
| train | 06 | 3173 | 0.746 | 0.214 | 0.059 | 0.206 | 0.107 |
| train | 07 | 4021 | 0.806 | 0.196 | 0.049 | 0.195 | 0.097 |
| train | 08 | 5009 | 0.708 | 0.231 | 0.052 | 0.208 | 0.106 |
| train | 09 | 4560 | 0.622 | 0.254 | 0.063 | 0.236 | 0.126 |
| train | 10 | 3028 | 0.663 | 0.264 | 0.083 | 0.229 | 0.124 |
| train | 11 | 2070 | 0.807 | 0.196 | 0.038 | 0.211 | 0.105 |
| train | 12 | 2006 | 0.805 | 0.163 | 0.042 | 0.203 | 0.101 |
| val | 01 | 722 | 0.873 | 0.162 | 0.017 | 0.200 | 0.099 |
| val | 02 | 144 | 0.872 | 0.064 | -0.013 | 0.176 | 0.084 |
| val | 06 | 136 | 1.246 | 0.172 | 0.077 | 0.180 | 0.103 |
| val | 10 | 1193 | 0.631 | 0.249 | 0.066 | 0.258 | 0.136 |
| val | 11 | 972 | 0.618 | 0.237 | 0.073 | 0.271 | 0.151 |
| val | 12 | 975 | 0.714 | 0.198 | 0.066 | 0.211 | 0.110 |
| test | 01 | 1157 | 0.583 | 0.203 | 0.059 | 0.261 | 0.137 |
| test | 02 | 1651 | 0.648 | 0.173 | 0.055 | 0.237 | 0.123 |
| test | 03 | 177 | 0.695 | 0.237 | 0.040 | 0.243 | 0.143 |
| test | 06 | 90 | 1.304 | 0.164 | 0.082 | 0.165 | 0.087 |
| test | 11 | 537 | 0.849 | 0.179 | 0.039 | 0.205 | 0.104 |
| test | 12 | 281 | 0.564 | 0.157 | 0.057 | 0.247 | 0.139 |

## By Residual Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.843 | 0.123 | 0.030 | 0.140 | 0.070 |
| train | low_le_q50 | 16581 | 0.363 | 0.274 | 0.077 | 0.396 | 0.231 |
| train | mid_q50_q90 | 13264 | 0.873 | 0.176 | 0.039 | 0.224 | 0.116 |
| val | high_gt_q90 | 421 | 1.852 | 0.138 | 0.045 | 0.155 | 0.078 |
| val | low_le_q50 | 2113 | 0.379 | 0.237 | 0.065 | 0.403 | 0.237 |
| val | mid_q50_q90 | 1608 | 0.870 | 0.193 | 0.049 | 0.239 | 0.126 |
| test | high_gt_q90 | 235 | 1.852 | 0.126 | 0.018 | 0.148 | 0.075 |
| test | low_le_q50 | 2066 | 0.389 | 0.203 | 0.068 | 0.380 | 0.218 |
| test | mid_q50_q90 | 1592 | 0.857 | 0.168 | 0.041 | 0.227 | 0.118 |

## By Temporal-Gradient Strength

| split | group | n | MAE | residual ACC | gradient ACC | residual std ratio | gradient RMS ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| train | high_gt_q90 | 3316 | 1.800 | 0.084 | 0.023 | 0.134 | 0.064 |
| train | low_le_q50 | 16581 | 0.370 | 0.308 | 0.080 | 0.389 | 0.258 |
| train | mid_q50_q90 | 13264 | 0.875 | 0.144 | 0.037 | 0.220 | 0.116 |
| val | high_gt_q90 | 395 | 1.837 | 0.090 | 0.035 | 0.146 | 0.071 |
| val | low_le_q50 | 2126 | 0.385 | 0.271 | 0.068 | 0.395 | 0.261 |
| val | mid_q50_q90 | 1621 | 0.884 | 0.159 | 0.048 | 0.236 | 0.128 |
| test | high_gt_q90 | 256 | 1.764 | 0.072 | 0.001 | 0.146 | 0.070 |
| test | low_le_q50 | 2041 | 0.394 | 0.229 | 0.065 | 0.375 | 0.240 |
| test | mid_q50_q90 | 1596 | 0.843 | 0.145 | 0.047 | 0.225 | 0.119 |

## Findings

- Overall train-to-val MAE gap is 0.004 m/s; residual ACC changes from 0.220 to 0.210.
- Validation mean wind speed shifts by +32.2% from train, while mean residual magnitude shifts by +0.8% and temporal-gradient magnitude by -1.0%.
- Validation seasonal composition is DJF_winter=1841, JJA_summer=136, SON_autumn=2165; not every season is represented.
- Highest validation MAE station is `dufeng_site_a` (1.246 m/s). Lowest validation residual ACC station is `PALUPD` (0.095).
- Validation high-residual group: MAE 1.852, residual ACC 0.138.
- Validation high-gradient group: gradient ACC 0.035, gradient RMS ratio 0.071.
- Even on train, residual std ratio is 0.214 and gradient RMS ratio is 0.111. The model is already strongly under-dispersed before considering validation shift.
- A ratio below 1 means predicted residual/gradient variability is too small; a ratio above 1 means it is over-amplified.

The JSON report also contains calendar-month and station-by-month tables for detailed auditing.
