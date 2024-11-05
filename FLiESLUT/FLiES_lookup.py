import numpy as np

from .constants import *
from .interpolate_ctauref import interpolate_ctauref
from .interpolate_The0 import interpolate_The0


def FLiES_lookup(
        ctype: np.ndarray,
        atype: np.ndarray,
        ctop: np.ndarray,
        albedo: np.ndarray,
        The0: np.ndarray,
        ctauref: np.ndarray,
        tauref: np.ndarray,
        LUT_filename: str = None,
        interpolate_cot: bool = True) -> np.ndarray:
    """Calculates shortwave incoming radiation using the FLiES look-up table.

    This function calculates `SWin` using the FLiES look-up table and performs interpolation for
    parameters not directly available in the table. It handles various input parameters related to
    cloud type, aerosol type, cloud top pressure, albedo, solar zenith angle, and optical depths.

    Args:
      ctype: Cloud type index (or indices).
      atype: Aerosol type index (or indices).
      ctop: Cloud top pressure level (or levels) in Pa.
      albedo: Surface albedo (or albedos).
      The0: Solar zenith angle (or angles) in degrees.
      ctauref: Cloud optical depth (or depths).
      tauref: Aerosol optical depth (or depths).
      LUT_filename: Optional path to the netCDF file containing the FLiES look-up table.
                    Defaults to the value specified in `constants.LUT_FILENAME`.
      interpolate_cot: Flag indicating whether to perform interpolation for cloud optical depth.
                       Defaults to True.

    Returns:
      A NumPy array containing the calculated `SWin` values.
    """
    if LUT_filename is None:
        LUT_filename = LUT_FILENAME

    ctop = np.where(np.isnan(ctop), 0.1, ctop)
    ctop = np.where(ctop > 10000, 10000, ctop)
    ctop = np.where(ctop <= 0, 100, ctop)
    ctop_factors = np.linspace(1000, 9000, 5)
    ctop_breaks = (ctop_factors[1:] + ctop_factors[:-1]) / 2.0
    ctop_index = np.digitize(ctop, ctop_breaks, right=True)

    albedo = np.where(np.isnan(albedo), 0.01, albedo)
    albedo = np.where(albedo <= 0, 0.01, albedo)
    albedo = np.where(albedo > 0.9, 0.9, albedo)
    albedo_factors = np.linspace(0.1, 0.7, 3)
    albedo_breaks = albedo_factors[1:] + albedo_factors[:-1] / 2.0
    albedo_index = np.digitize(albedo, albedo_breaks, right=True)

    tauref = np.where(np.isnan(tauref), 0.1, tauref)
    tauref = np.where(tauref > 1, 1, tauref)
    tauref_factors = np.linspace(0.1, 0.9, 5)[:-1]
    tauref_breaks = (tauref_factors[1:] + tauref_factors[:-1]) / 2.0
    tauref_index = np.digitize(tauref, tauref_breaks, right=True)

    ctauref = np.where(np.isnan(ctauref), 0.1, ctauref)
    ctauref = np.where(ctauref > 130, 130, ctauref)
    ctauref = np.where(ctauref <= 0, 0.01, ctauref)

    if interpolate_cot:
        SWin = interpolate_ctauref(
            LUT_filename,
            ctype,
            atype,
            ctop_index,
            albedo_index,
            The0,
            ctauref,
            tauref_index
        )
    else:
        ctauref_factors = np.array([0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 110])
        ctauref_breaks = (ctauref_factors[1:] + ctauref_factors[:-1]) / 2.0
        ctauref_index = np.digitize(ctauref, ctauref_breaks, right=True)

        SWin = interpolate_The0(
            LUT_filename,
            ctype,
            atype,
            ctop_index,
            albedo_index,
            The0,
            ctauref_index,
            tauref_index
        )

    SWin = np.where(np.isinf(SWin), np.nan, SWin)

    return SWin