"""Raw NetCDF reading, pairing, and inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RawFilePair:
    """Matched 3600 s and 600 s NetCDF files."""

    hourly_path: Path
    target_path: Path
    prefix: str


def _decode_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_attr_value(value.item())
        if value.dtype.kind in {"S", "O"}:
            return [_decode_attr_value(v) for v in value.tolist()]
    return value


class H5Array:
    """Small xarray-like wrapper for NetCDF4/HDF5 datasets."""

    def __init__(self, dataset, name: str):
        self._dataset = dataset
        self.name = name
        self.dims = tuple(self._dimension_names())
        self.shape = dataset.shape
        self.dtype = dataset.dtype
        self.attrs = {k: _decode_attr_value(v) for k, v in dataset.attrs.items()}

    @property
    def values(self):
        values = self._dataset[()]
        if getattr(values, "dtype", None) is not None and values.dtype.kind == "S":
            return values.astype(str)
        return values

    def _dimension_names(self) -> list[str]:
        names: list[str] = []
        for i in range(len(self._dataset.shape)):
            keys = list(self._dataset.dims[i].keys())
            names.append(str(keys[0]) if keys else f"dim_{i}")
        if not names and self.name in self._dataset.file:
            obj = self._dataset.file[self.name]
            if obj.attrs.get("CLASS") == b"DIMENSION_SCALE":
                names = [self.name]
        return names

    def transpose(self, *dims: str):
        axis_by_dim = {dim: i for i, dim in enumerate(self.dims)}
        try:
            axes = [axis_by_dim[dim] for dim in dims]
        except KeyError as exc:
            raise ValueError(f"Variable {self.name} dims {self.dims} do not contain requested dims {dims}") from exc
        values = np.transpose(self.values, axes)
        return NumpyArray(values, dims=dims, attrs=self.attrs, name=self.name)


class NumpyArray:
    """Minimal array wrapper returned by H5Array.transpose."""

    def __init__(self, values, dims: tuple[str, ...], attrs: dict[str, Any], name: str):
        self._values = values
        self.dims = tuple(dims)
        self.attrs = attrs
        self.name = name
        self.shape = values.shape
        self.dtype = values.dtype

    @property
    def values(self):
        return self._values


class H5Dataset:
    """Small xarray-like dataset wrapper backed by h5py."""

    def __init__(self, path: Path):
        import h5py

        self.path = path
        self._file = h5py.File(path, "r")
        self.attrs = {k: _decode_attr_value(v) for k, v in self._file.attrs.items()}
        self._arrays = {
            name: H5Array(obj, name)
            for name, obj in self._file.items()
            if hasattr(obj, "shape")
        }
        self.coords = {
            name: arr
            for name, arr in self._arrays.items()
            if arr.attrs.get("CLASS") == "DIMENSION_SCALE" or name in {"time", "station", "altitude"}
        }
        self.data_vars = {name: arr for name, arr in self._arrays.items() if name not in self.coords}
        self.sizes = {name: int(arr.shape[0]) for name, arr in self.coords.items() if len(arr.shape) == 1}

    def __contains__(self, name: str) -> bool:
        return name in self._arrays

    def __getitem__(self, name: str) -> H5Array:
        return self._arrays[name]

    def close(self) -> None:
        self._file.close()


def open_dataset(path: str | Path) -> Any:
    """Open a NetCDF dataset with xarray."""

    import xarray as xr

    path = Path(path)
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        try:
            return H5Dataset(path)
        except Exception as h5_exc:
            raise RuntimeError(
                f"Unable to open NetCDF file {path}. Install a compatible xarray "
                "backend such as netCDF4 or h5netcdf, or convert the file to a "
                "supported NetCDF format."
            ) from h5_exc


def _scan_by_suffix(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.exists():
        return {}
    files = sorted(directory.glob(f"*{suffix}"))
    return {path.name[: -len(suffix)]: path for path in files}


def pair_nc_files(
    raw_3600s_dir: str | Path,
    raw_600s_dir: str | Path,
    hourly_suffix: str = "_3600s.nc",
    target_suffix: str = "_600s.nc",
) -> list[RawFilePair]:
    """Pair raw 3600 s and 600 s files by stripped filename prefix."""

    hourly = _scan_by_suffix(Path(raw_3600s_dir), hourly_suffix)
    target = _scan_by_suffix(Path(raw_600s_dir), target_suffix)
    prefixes = sorted(set(hourly) & set(target))
    return [RawFilePair(hourly[p], target[p], p) for p in prefixes]


def dataset_time_values(ds: Any, time_name: str) -> np.ndarray:
    """Read a time coordinate as minute-resolution ``datetime64`` values."""

    if time_name not in ds:
        raise KeyError(f"Time variable not found in dataset: {time_name}")
    values = np.asarray(ds[time_name].values)
    if np.issubdtype(values.dtype, np.datetime64):
        return values.astype("datetime64[m]")

    attrs = getattr(ds[time_name], "attrs", {})
    units = _decode_attr_value(attrs.get("units", ""))
    decoded = decode_cf_time_values(values, str(units))
    return decoded.astype("datetime64[m]")


def decode_cf_time_values(values: np.ndarray, units: str) -> np.ndarray:
    """Decode simple CF time units such as ``hours since YYYY-mm-dd HH:MM:SS``."""

    match = re.match(r"^(seconds|minutes|hours|days)\s+since\s+(.+)$", units)
    if not match:
        raise ValueError(f"Unsupported time units: {units!r}")
    unit, base_text = match.groups()
    base_text = base_text.strip().replace(" ", "T")
    base = np.datetime64(base_text)
    if unit == "seconds":
        delta_unit = "s"
    elif unit == "minutes":
        delta_unit = "m"
    elif unit == "hours":
        delta_unit = "h"
    elif unit == "days":
        delta_unit = "D"
    else:
        raise ValueError(f"Unsupported time unit: {unit}")
    return base + values.astype("timedelta64[" + delta_unit + "]")


def require_variables(ds: Any, names: Iterable[str], source: Path) -> None:
    """Raise a clear error if any required variables are missing."""

    missing = [name for name in names if name and name not in ds]
    if missing:
        raise KeyError(f"{source} is missing required variables: {missing}")


def variable_nan_count(values: np.ndarray) -> int | None:
    """Return NaN count for numeric arrays, otherwise ``None``."""

    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.number):
        return int(np.isnan(array).sum())
    return None


def variable_missing_count(values: np.ndarray, missing_value: float | None) -> int | None:
    """Return missing sentinel count for numeric arrays."""

    if missing_value is None:
        return None
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.number):
        return int((array == missing_value).sum())
    return None


def variable_attrs(ds: Any, name: str) -> dict[str, Any]:
    """Return variable attributes as plain Python values."""

    return dict(getattr(ds[name], "attrs", {}))


def summarize_netcdf(path: str | Path, missing_value: float | None = -999.0) -> dict[str, Any]:
    """Return dimensions, variables, time ranges, shapes, NaNs, and attrs."""

    path = Path(path)
    ds = open_dataset(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "dimensions": {name: int(size) for name, size in ds.sizes.items()},
        "coordinates": {},
        "variables": {},
        "time_ranges": {},
        "attrs": dict(ds.attrs),
    }

    for name, coord in ds.coords.items():
        summary["coordinates"][name] = {
            "dims": list(coord.dims),
            "shape": list(coord.shape),
            "dtype": str(coord.dtype),
            "attrs": dict(coord.attrs),
        }

    for name, var in ds.data_vars.items():
        values = var.values
        summary["variables"][name] = {
            "dims": list(var.dims),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "nan_count": variable_nan_count(values),
            "missing_value_count": variable_missing_count(values, missing_value),
            "attrs": variable_attrs(ds, name),
        }

    for name in list(ds.coords) + list(ds.data_vars):
        if "time" not in name.lower():
            continue
        values = np.asarray(ds[name].values).reshape(-1)
        if values.size == 0:
            summary["time_ranges"][name] = {"start": None, "end": None, "count": 0}
            continue
        summary["time_ranges"][name] = {
            "start": str(values[0]),
            "end": str(values[-1]),
            "count": int(values.size),
            "dtype": str(values.dtype),
            "attrs": variable_attrs(ds, name),
        }
    return summary
