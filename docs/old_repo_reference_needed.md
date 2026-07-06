# Old Repo Data Reference

## 1. 原始数据结构和文件命名规则

- 入口脚本：`scripts/preprocess_nc_pairs.py`
- 核心逻辑：`src/preprocessing_nc.py`
- 配置字段：
  - `paths.raw_3600s_dir`
  - `paths.raw_600s_dir`
- 文件命名：
  - `*_3600s.nc`
  - `*_600s.nc`
- 配对函数：`pair_nc_files()`
- 配对规则：去掉 `_3600s.nc` / `_600s.nc` 后，按相同 prefix 配对。
- 如果 `raw_3600s_dir` 和 `raw_600s_dir` 都找不到文件，旧代码会 fallback 到 `Path(raw_3600s_dir).parent` 下查找。

示例：

```text
paris_dwl_L3V1.42_202206100000_202206110000_3600s.nc
paris_dwl_L3V1.42_202206100000_202206110000_600s.nc
```

## 2. 原始 NetCDF 变量名

读取函数：`open_nc_arrays(path, variables)`

旧代码实际使用：

| 含义 | NetCDF 名 |
|---|---|
| u 风 | `u` |
| v 风 | `v` |
| time | `time` |
| station | `station` |
| height/layer | `altitude` |
| station latitude | `station_lat` |
| station longitude | `station_lon` |
| station altitude | `station_altitude` |
| station height | `station_height` |

原始文件中还存在但旧 preprocessing 没有用于 mask 的变量：

```text
flag_suspect_retrieval_warn
flag_suspect_retrieval_removed
ws
wd
system_id
```

`flag_low_signal_warn` 是否存在：不确定，需要新仓库 `inspect_raw_nc.py` 验证。

## 3. 高度层选择规则

相关函数：

- `station_height_selection()`
- `find_height_indices()`

Paris 配置中使用：

```yaml
selected_heights: [250, 275, 300, 325, 350, 375]
height_reference: "agl_rounded_station_altitude"
max_height_diff: 0.1
```

高度逻辑：

- 旧配置中的 `selected_heights` 是 AGL 目标高度。
- 使用 `station_altitude_ref` 转成 ASL：

```text
target_asl = station_altitude_ref + selected_heights
```

- 然后在 NetCDF `altitude` 层里找最近层。
- 如果最近层和目标高度差超过 `max_height_diff`，报错。
- 不做垂直插值。

因此旧代码是：

```text
AGL target heights -> ASL target heights -> nearest altitude layer index
```

保存 metadata：

```text
selected_heights
selected_heights_agl
height_indices
target_heights_asl
actual_heights_asl
actual_heights_agl_true
```

## 4. 600s 和 3600s 数据如何配对

函数：`pair_nc_files()`

逻辑：

1. 分别扫描 `*_3600s.nc` 和 `*_600s.nc`。
2. 去掉后缀得到 prefix。
3. 只保留两个集合交集中的 prefix。
4. 返回：

```python
(path_3600, path_600, prefix)
```

如果某天只有 600s 或只有 3600s，旧代码不会生成 pair，因此不会进入样本生成。

## 5. 时间对齐逻辑和滑动窗口构造

相关函数：

- `_time_values_to_minutes()`
- `_target_offsets()`
- `_target_slice_bounds()`
- `build_samples_from_sequence()`

旧代码曾默认按：

```text
timestamp = interval end
```

因此旧 target offsets 是：

```text
[-50, -40, -30, -20, -10, 0] minutes
```

即小时 `T` 对应：

```text
Y = [T-50, T-40, T-30, T-20, T-10, T]
```

这在新仓库中不能沿用。

当前已新增 start-aligned 逻辑：

```yaml
target_time_alignment: "start"
```

对应：

```text
[0, 10, 20, 30, 40, 50] minutes
```

新仓库设定为 `timestamp = interval start`，因此应使用：

```text
hourly X_T -> 600s Y_T, Y_{T+10}, ..., Y_{T+50}
```

如果沿用旧代码中的以下逻辑，会导致时间对齐错误：

```python
target_offsets = [-50, -40, -30, -20, -10, 0]
target_start = target_steps * k - (target_steps - 1)
target_end = target_steps * k + 1
```

input context 构造：

```yaml
context_alignment: "causal_last"
context_hours: 6
```

对应：

```text
X = [T-5h, T-4h, T-3h, T-2h, T-1h, T]
```

旧代码也支持 `centered`，但新任务建议只参考 `causal_last`。

## 6. station 维度和站点样本处理

Paris 原始文件中 station 是文件维度：

```text
u/v shape = [station, time, altitude]
```

旧代码逐 station 生成样本：

```python
for station_index, station_id in enumerate(station_ids):
    ...
```

站点 ID 来源：

```python
hourly["station"]
```

保存到 `.pt`：

```text
station_id
station_index
station_lat
station_lon
station_altitude
station_height
```

如果某个 station 的样本有效率过低，样本会被过滤；不是整站一次性删除。

## 7. NaN、-999、缺失值处理和 mask 机制

缺失值配置：

```yaml
missing_value: -999.0
```

mask 生成函数：

```python
_valid_mask(array, missing_value)
```

定义：

```python
np.isfinite(array) & (array != missing_value)
```

mask 语义：

```text
True  = valid
False = invalid
```

保存前会把 NaN 替换为 `-999.0`：

```python
x_hourly = np.where(np.isnan(x_hourly), missing_value, x_hourly)
y_10min = np.where(np.isnan(y_10min), missing_value, y_10min)
```

保存的 mask：

```text
x_valid_mask  # input mask
valid_mask    # target mask
```

## 8. 质量控制 flags 如何转成 mask

旧代码没有把 QC flag 转成 mask。

原始 Paris 文件中可见：

```text
flag_suspect_retrieval_warn
flag_suspect_retrieval_removed
```

它们单位是 `%`，更像 aggregation interval 内的 percentage occurrence。

但旧代码：

- 没有在 `open_nc_arrays()` 中读取这些 flag。
- 没有使用 `flag_suspect_retrieval_warn`。
- 没有使用 `flag_suspect_retrieval_removed`。
- 没有使用 `flag_low_signal_warn`。
- mask 只来自 NaN / -999 / finite check。

新仓库需要重新定义 QC 规则。  
具体 flag 是否是 0/1 还是 percentage occurrence，需要新仓库 `inspect_raw_nc.py` 验证。

## 9. 数据 shape 和最终 dataset keys

原始读取 shape：

```text
u/v: [station, time, altitude]
```

旧 `.pt` 保存 shape：

```text
X_hourly: [2, H, T_ctx]
Y_10min: [2, H, T_out]
x_valid_mask: [2, H, T_ctx]
valid_mask: [2, H, T_out]
```

常见 Paris 设置：

```text
X_hourly: [2, 6, 6]
Y_10min: [2, 6, 6]
```

主要 keys：

```text
X_hourly
Y_10min
x_valid_mask
valid_mask
baseline_repeat
baseline_linear
station_id
station_index
center_time
context_times
target_times
context_hour_indices
target_10min_indices
target_time_alignment
context_alignment
selected_heights
height_indices
target_heights_asl
actual_heights_asl
actual_heights_agl_true
station_lat
station_lon
station_altitude
station_height
source_3600s_files
source_600s_files
channel_names
```

注意：`baseline_repeat` / `baseline_linear` 是旧模型实验用字段，新仓库 preprocessing 不一定需要。

## 10. train / val / test 划分方式

划分函数：

```python
write_time_splits()
```

逻辑：

1. 按 `center_time` 分组。
2. 对 unique center times 排序。
3. 按比例顺序切分。

配置：

```yaml
split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  split_by_unique_time: true
```

不是随机划分。  
不是按 station 划分。  
它通过按时间排序切分，避免同一 `center_time` 同时进入 train 和 test。

输出：

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

## 11. normalization 和样本过滤规则

归一化统计函数：

```python
compute_normalization_stats()
```

统计对象：

```text
X_hourly
Y_10min
```

只使用 train split：

```python
compute_normalization_stats(splits["train"], ...)
```

invalid 值不参与统计：

```python
xm = sample["x_valid_mask"] & finite & != missing_value
ym = sample["valid_mask"] & finite & != missing_value
```

保存：

```text
norm_stats.pt
```

包含：

```text
x_mean
x_std
y_mean
y_std
channel_names
```

样本过滤条件：

```yaml
min_valid_ratio_x: 0.8
min_valid_ratio_x_per_hour: 0.8
min_valid_ratio_x_center: 1.0
min_valid_ratio_y: 0.8
```

对应过滤：

- 整个 X 有效率过低则丢弃。
- X 任一小时有效率过低则丢弃。
- `causal_last` 的最后一个小时有效率不足则丢弃。
- Y 有效率过低则丢弃。

## Can Reuse as Reference / Must Change for New Repo

### Can Reuse as Reference

- 文件按 `*_3600s.nc` / `*_600s.nc` prefix 配对的思路。
- 原始变量名：`u`, `v`, `time`, `station`, `altitude`。
- station metadata：`station_lat`, `station_lon`, `station_altitude`, `station_height`。
- 高度按目标物理高度 nearest layer 选择的参考逻辑。
- mask 语义：`True = valid`。
- train/val/test 按时间顺序划分，避免时间泄漏。
- 只用 train split 计算 normalization stats。

### Must Change for New Repo

- 时间对齐必须使用 interval start：

```text
X_T -> [Y_T, Y_{T+10}, ..., Y_{T+50}]
```

- 不要沿用旧的 end-aligned offsets：

```text
[T-50, ..., T]
```

- 不要沿用旧 shape：

```text
[2, H, T]
```

如果新仓库使用：

```text
X: [N, L, H, C]
Y: [N, T_out, H, C]
```

则 preprocessing 输出也应按这个约定设计。

- 不要迁移模型相关字段：

```text
baseline_repeat
baseline_linear
model input mask channel 拼接逻辑
```

- QC flags 不能照旧忽略；新仓库应明确检查并定义规则。

- `flag_low_signal_warn` 是否存在和如何使用：不确定，需要新仓库 `inspect_raw_nc.py` 验证。
