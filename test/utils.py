import os
import shutil
import xarray as xr
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, ds):
        self.xarray_dataset = ds

    def get(self):
        return self

    def add_kazarr_attributes(self):
        pass

    def save(self, name, to_netcdf=False):
        root = os.environ.get("TEST_TMP_FOLDER", "tmp")
        if to_netcdf:
            name = f"{name}.nc" if not name.endswith(".nc") else name
            self.xarray_dataset.to_netcdf(os.path.join(root, name))
        else:
            name = f"{name}.zarr" if not name.endswith(".zarr") else name
            self.xarray_dataset.to_zarr(os.path.join(root, name), mode="w")

    def __getattr__(self, name):
        return getattr(self.xarray_dataset, name)

    def __getitem__(self, key):
        return self.xarray_dataset[key]

    def __repr__(self):
        return repr(self.xarray_dataset)

    def __dir__(self):
        return dir(self.xarray_dataset)


class DatasetGenerator:
    def __init__(self, description, dimensions=None):
        self.description = description
        self.dimensions = dimensions or {}

    def generate(self):
        coords = {}
        data_vars = {}
        dims_sizes = {}

        base_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(base_dir, "samples")

        loaded_variables = {}

        # Pass 1: Process 'load' types
        # Must be done first to resolve dimensions that depend on loaded data
        for var_name, config in self.description.items():
            if config.get("type") == "load":
                sample_file = config.get("sample")
                file_path = os.path.join(samples_dir, sample_file)

                if not os.path.exists(file_path):
                    file_path_abs = os.path.abspath(sample_file)
                    if os.path.exists(file_path_abs):
                        file_path = file_path_abs

                try:
                    ds_loaded = xr.open_dataset(file_path)
                    var_source_name = config.get("variable")

                    if var_source_name in ds_loaded:
                        da = ds_loaded[var_source_name]
                        loaded_variables[var_name] = da

                        target_dims = config.get("dimensions", [])
                        if len(target_dims) == len(da.dims):
                            for i, target_dim in enumerate(target_dims):
                                size = da.shape[i]
                                dims_sizes[target_dim] = size
                                dims_sizes[f"{var_name}.{target_dim}"] = size

                        data_vars[var_name] = (target_dims, da.values)
                    else:
                        # Missing variable
                        pass
                except Exception:
                    # Can't open dataset
                    pass
            elif config.get("type") == "array" and "values" in config:
                target_dims = config.get("dimensions", [])
                da = xr.DataArray(config["values"], dims=target_dims)
                if len(target_dims) == len(da.dims):
                    for i, target_dim in enumerate(target_dims):
                        size = da.shape[i]
                        dims_sizes[target_dim] = size
                        dims_sizes[f"{var_name}.{target_dim}"] = size

                    data_vars[var_name] = (target_dims, da.values)

        # Pass 2: Dimensions and Coords
        for var_name, config in self.description.items():
            if config.get("type") == "load" or config.get("type") == "array":
                continue

            dims_config = config.get("dimensions", [])
            for dim_info in dims_config:
                dim_name = dim_info["name"]
                size_ref = dim_info.get("size")

                size = 10
                if isinstance(size_ref, int):
                    size = size_ref
                elif isinstance(size_ref, str):
                    if size_ref in dims_sizes:
                        size = dims_sizes[size_ref]

                dims_sizes[dim_name] = size

                if dim_name not in coords and dim_info.get("type") is not None:
                    dim_type = dim_info.get("type")
                    values = self._generate_values(dim_type, size, dim_info)
                    if dim_type == "steps":
                        ref_time = dim_info.get("start", "2026-01-01")
                        data_vars["reference_time"] = ([], ref_time)
                        coords[dim_name] = (
                            [dim_name],
                            values,
                            {
                                "units": "s",
                                "long_name": "Time steps since reference time",
                            },
                        )
                    else:
                        coords[dim_name] = values

        # Pass 3: Generate Data
        for var_name, config in self.description.items():
            if config.get("type") == "load" or config.get("type") == "array":
                continue

            var_dims = [d["name"] for d in config.get("dimensions", [])]
            var_shape = tuple([dims_sizes.get(d, 10) for d in var_dims])

            data_type = config.get("type")

            data_vars[var_name] = (
                var_dims,
                self._generate_values(data_type, var_shape, config),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        return Dataset(ds)

    def _generate_values(self, values_type, size, options):
        if values_type == "date":
            start = options.get("start", "2026-01-01")
            freq = options.get("freq", "D")
            if start and freq:
                return pd.date_range(start, periods=size, freq=freq)
        elif values_type == "steps":
            # Array of seconds since start date
            start = options.get("start", "2026-01-01")
            freq = options.get("freq", "D")
            dates = pd.date_range(start, periods=size, freq=freq)
            start_time = pd.to_datetime(start)
            return (dates - start_time).total_seconds()
        elif values_type == "array":
            return np.array(options.get("values", [])).reshape(size)
        elif values_type == "latitude" or values_type == "longitude":
            start = options.get("start")
            end = options.get("end")
            step = options.get("step")
            center = options.get("center", False)
            if start is not None and end is not None:
                return np.linspace(start, end, size)
            elif start is not None and step is not None:
                if center:
                    return np.linspace(
                        start - step * (size - 1) / 2,
                        start + step * (size - 1) / 2,
                        size,
                    )
                else:
                    return np.linspace(start, start + step * (size - 1), size)
            elif center and isinstance(center, float):
                step = step or 0.1
                return np.linspace(
                    center - step * (size - 1) / 2, center + step * (size - 1) / 2, size
                )
            else:
                return (
                    np.linspace(-90, 90, size)
                    if values_type == "latitude"
                    else np.linspace(-180, 180, size)
                )
        else:
            # Consider float by default
            if isinstance(size, tuple):
                total_points = np.prod(size)
            elif size == 0:
                return np.array([])
            else:
                total_points = size

            method = options.get("method", "linear")
            step = options.get("step")
            start = options.get("start", 0)
            bounds = options.get("bounds", {"min": 0, "max": 100})
            periods = options.get("periods", 1)

            if method == "linear":
                if step is not None:
                    normalized_data = np.arange(
                        start, start + total_points * step, step
                    )
                else:
                    normalized_data = np.linspace(
                        bounds["min"], bounds["max"], total_points
                    )
            else:  # method == "sin" or method == "cos"
                flat_data = np.linspace(0, periods * 2 * np.pi, total_points)
                raw_data = np.sin(flat_data) if method == "sin" else np.cos(flat_data)
                d_min = bounds["min"]
                d_max = bounds["max"]

                normalized_data = ((raw_data + 1) / 2) * (d_max - d_min) + d_min

            return normalized_data.reshape(size)


def cleanup_test_files():
    for root, dirs, files in os.walk(os.environ.get("TEST_TMP_FOLDER", "tmp")):
        for dirname in dirs:
            if dirname.endswith(".zarr"):
                found_path = os.path.join(root, dirname)
                shutil.rmtree(found_path, ignore_errors=True)
        for filename in files:
            if filename.endswith(".nc"):
                found_path = os.path.join(root, filename)
                try:
                    os.remove(found_path)
                except OSError:
                    pass


def get_value(shape, method="linear", bounds={"min": 0, "max": 100}, periods=1):
    total_points = np.prod(shape)
    if method == "linear":
        return np.linspace(bounds["min"], bounds["max"], total_points).reshape(shape)
    else:  # method == "sin" or method == "cos"
        flat_data = np.linspace(0, periods * 2 * np.pi, total_points)
        raw_data = np.sin(flat_data) if method == "sin" else np.cos(flat_data)
        d_min = bounds["min"]
        d_max = bounds["max"]
        normalized_data = ((raw_data + 1) / 2) * (d_max - d_min) + d_min
        return normalized_data.reshape(shape)
