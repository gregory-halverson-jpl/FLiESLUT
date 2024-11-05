import numpy as np

from .FLiES_lookup import FLiES_lookup

def FLiES_LUT(
        doy,
        cloud_mask,
        COT,
        koppen_geiger,
        IGBP,
        cloud_top,
        albedo,
        SZA,
        AOT):
    """
    This function processes the lookup-table implementation of the Forest Light Environmental Simulator.
    :param doy: day of year
    :param cloud_mask:
    :param COT:
    :param koppen_geiger:
    :param IGBP:
    :param cloud_top:
    :param albedo:
    :param SZA:
    :param AOT:
    :return:
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
