import numpy as np
import rasters as rt

from .query_FLiES import query_FLiES

def interpolate_The0(filename: str,
                     ctype: np.ndarray,
                     atype: np.ndarray,
                     ctop_index: np.ndarray,
                     albedo_index: np.ndarray,
                     The0: np.ndarray,
                     ctauref_index: np.ndarray,
                     tauref_index: np.ndarray) -> np.ndarray:
    """Interpolates shortwave incoming radiation for given solar zenith angles.

    This function interpolates the `SWin` values from the FLiES look-up table for arbitrary solar zenith
    angles (`The0`). It uses linear interpolation between the two closest `The0` values available in the
    look-up table.

    Args:
      filename: The path to the netCDF file containing the FLiES look-up table.
      ctype:  Cloud type index (or indices).
      atype:  Aerosol type index (or indices).
      ctop_index: Cloud top pressure level index (or indices).
      albedo_index: Albedo index (or indices).
      The0: Solar zenith angle (or angles) in degrees.
      ctauref_index: Cloud optical depth index (or indices).
      tauref_index: Aerosol optical depth index (or indices).

    Returns:
      A NumPy array containing the interpolated `SWin` values.
    """
    # constrain solar zenith angle
    The0 = rt.clip(The0, 5, 85)

    # get low index for solar zenith angle
    The0_index_low = np.clip(np.floor(The0 / 5.0).astype(np.int32) - 1, 0, 16).astype(np.int32)

    # get high index for solar zenith angle
    The0_index_high = np.clip(np.ceil(The0 / 5.0).astype(np.int32) - 1, 0, 16).astype(np.int32)

    # query closest incoming shortwave
    SWin_The0_low = query_FLiES(
        filename,
        ctype,
        atype,
        ctop_index,
        albedo_index,
        The0_index_low,
        ctauref_index,
        tauref_index
    )

    # query next closest incoming shortwave
    SWin_The0_high = query_FLiES(
        filename,
        ctype,
        atype,
        ctop_index,
        albedo_index,
        The0_index_high,
        ctauref_index,
        tauref_index
    )

    The0_slope = (SWin_The0_high - SWin_The0_low) / 5.0
    The0_intermediate = The0 - np.floor(The0 / 5.0) * 5.0
    SWin = SWin_The0_low + The0_slope * The0_intermediate

    return SWin