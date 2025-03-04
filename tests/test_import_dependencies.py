import pytest

# List of dependencies
dependencies = [
    "geos5fp",
    "keras",
    "koppengeiger",
    "MCD12C1_2019_v006",
    "netCDF4",
    "numpy",
    "pandas",
    "rasters",
    "sentinel_tiles",
    "solar_apparent_time",
    "sun_angles",
    "tensorflow"
]

# Generate individual test functions for each dependency
@pytest.mark.parametrize("dependency", dependencies)
def test_dependency_import(dependency):
    __import__(dependency)
