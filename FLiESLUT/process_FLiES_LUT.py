import numpy as np

from .FLiES_lookup import FLiES_lookup

def process_FLiES_LUT(
        doy: np.ndarray,
        cloud_mask: np.ndarray,
        COT: np.ndarray,
        koppen_geiger: np.ndarray,
        IGBP: np.ndarray,
        cloud_top: np.ndarray,
        albedo: np.ndarray,
        SZA: np.ndarray,
        AOT: np.ndarray) -> np.ndarray:
    """Processes the FLiES look-up table to calculate shortwave incoming radiation.

    This function uses the FLiES look-up table implementation to calculate `SWin` based on various
    input parameters. It determines cloud type and aerosol type, then uses the `FLiES_lookup` function
    to calculate `SWin`. Finally, it constrains `SWin` to the top-of-atmosphere value and masks invalid data.

    Args:
      doy: Day of year (or days of year).
      cloud_mask: Boolean array indicating cloud presence (True for cloudy, False for clear).
      COT: Cloud optical thickness (or thicknesses).
      koppen_geiger: Koppen-Geiger climate classification code (or codes).
      IGBP: International Geosphere-Biosphere Programme land cover classification code (or codes).
      cloud_top: Cloud top pressure level (or levels) in Pa.
      albedo: Surface albedo (or albedos).
      SZA: Solar zenith angle (or angles) in degrees.
      AOT: Aerosol optical thickness (or thicknesses).

    Returns:
      A NumPy array containing the calculated `SWin` values.
    """

    # set cloud type by cloud mask and koppen geiger
    # 0: cloud-free
    # 1: stratus continental
    # 2: cumulous continental
    ctype = np.where(np.logical_and(cloud_mask, koppen_geiger == 1), 2, 1)
    ctype = np.where(np.logical_not(cloud_mask), 0, ctype)

    # set aerosol type by IGBP
    atype = np.where(IGBP == 13, 1, 0)

    # calculate incoming shortwave using FLiES lookup table
    SWin = FLiES_lookup(
        ctype,
        atype,
        cloud_top,
        albedo,
        SZA,
        COT,
        AOT
    )

    # constrain incoming shortwave to top of atmosphere
    SWin_toa = 1370 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)) * np.sin(np.radians(90 - SZA))
    SWin = np.clip(SWin, None, SWin_toa)

    # mask SWin to COT to prevent garbage data in the gap between swaths
    SWin = np.where(np.isnan(COT), np.nan, SWin)

    return SWin