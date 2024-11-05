import warnings

import numpy as np

from .interpolate_The0 import interpolate_The0

def interpolate_ctauref(
        filename: str,
        ctype: np.ndarray,
        atype: np.ndarray,
        ctop_index: np.ndarray,
        albedo_index: np.ndarray,
        The0: np.ndarray,
        ctauref: np.ndarray,
        tauref_index: np.ndarray) -> np.ndarray:
    """Interpolates shortwave incoming radiation for given cloud optical depths.

    This function interpolates the `SWin` values from the FLiES look-up table for arbitrary cloud optical
    depths (`ctauref`). It uses linear interpolation between the two closest `ctauref` values available
    in the look-up table.

    Args:
      filename: The path to the netCDF file containing the FLiES look-up table.
      ctype:  Cloud type index (or indices).
      atype:  Aerosol type index (or indices).
      ctop_index: Cloud top pressure level index (or indices).
      albedo_index: Albedo index (or indices).
      The0: Solar zenith angle (or angles) in degrees.
      ctauref: Cloud optical depth (or depths).
      tauref_index: Aerosol optical depth index (or indices).

    Returns:
      A NumPy array containing the interpolated `SWin` values.
    """
    ctauref_factors = np.array([0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 110])
    ctauref_index = np.digitize(np.clip(ctauref, 0.1, 110), (ctauref_factors)[:-1])

    ctauref_index = np.where(np.isnan(ctauref), 0, ctauref_index)

    ctauref_intermediate = ctauref - ctauref_factors[ctauref_index]

    warnings.filterwarnings('ignore')

    ctauref_index2 = np.clip(np.where(ctauref_intermediate < 0, ctauref_index - 1, ctauref_index + 1), 0, 9)

    warnings.resetwarnings()

    ctauref_delta = ctauref_factors[ctauref_index2] - ctauref_factors[ctauref_index]

    SWin_ctauref1 = interpolate_The0(
        filename,
        ctype,
        atype,
        ctop_index,
        albedo_index,
        The0,
        ctauref_index,
        tauref_index
    )

    SWin_ctauref2 = interpolate_The0(
        filename,
        ctype,
        atype,
        ctop_index,
        albedo_index,
        The0,
        ctauref_index2,
        tauref_index
    )

    warnings.filterwarnings('ignore')

    ctauref_slope = (SWin_ctauref2 - SWin_ctauref1) / ctauref_delta

    warnings.resetwarnings()

    correction = ctauref_slope * ctauref_intermediate
    SWin = SWin_ctauref1 + np.where(np.isnan(correction), 0, correction)

    return SWin
