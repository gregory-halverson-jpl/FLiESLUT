import netCDF4
import numpy as np


def query_FLiES(
        filename: str,
        ctype: np.ndarray,
        atype: np.ndarray,
        ctop_index: np.ndarray,
        albedo_index: np.ndarray,
        The0_index: np.ndarray,
        ctauref_index: np.ndarray,
        tauref_index: np.ndarray) -> np.ndarray:
    """Queries a look-up table representation of the Forest Light Environmental Simulator (FLiES) radiative transfer model.

    This function extracts shortwave incoming radiation (`SWin`) values from a netCDF file containing FLiES
    model output. The file is assumed to have a variable named 'SWin' with dimensions corresponding to
    cloud type, aerosol type, cloud top pressure level, albedo, solar zenith angle, and atmospheric conditions.

    Args:
      filename: The path to the netCDF file containing the FLiES look-up table.
      ctype:  Cloud type index (or indices).
      atype:  Aerosol type index (or indices).
      ctop_index: Cloud top pressure level index (or indices).
      albedo_index: Albedo index (or indices).
      The0_index: Solar zenith angle index (or indices).
      ctauref_index: Cloud optical depth index (or indices).
      tauref_index: Aerosol optical depth index (or indices).

    Returns:
      A NumPy array containing the `SWin` values corresponding to the input indices.
    """
    with netCDF4.Dataset(filename, 'r') as f:
        return f['SWin'][0][ctype, atype, ctop_index, albedo_index, The0_index, ctauref_index, tauref_index]
