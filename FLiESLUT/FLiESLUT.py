"""
Forest Light Environmental Simulator (FLiES)
Artificial Neural Network Implementation
for the Breathing Earth Systems Simulator (BESS)
"""

from typing import Union
import logging
import warnings
from datetime import datetime
from os.path import join, abspath, dirname
from typing import Callable

import netCDF4
from dateutil import parser

import numpy as np
import rasters as rt
from rasters import RasterGeometry, Raster

from geos5fp import GEOS5FP
from MCD12C1_2019_v006 import load_MCD12C1_IGBP
from koppengeiger import load_koppen_geiger
from solar_apparent_time import UTC_to_solar
from sun_angles import calculate_SZA_from_DOY_and_hour

__author__ = "Gregory Halverson, Robert Freepartner"

MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")

DEFAULT_WORKING_DIRECTORY = "."
DEFAULT_FLIES_INTERMEDIATE = "FLiESLUT_intermediate"
LUT_FILENAME = join(abspath(dirname(__file__)), "FLiESLUT.nc")
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_INCLUDE_PREVIEW = True
DEFAULT_RESAMPLING = "cubic"
DEFAULT_SAVE_INTERMEDIATE = True
DEFAULT_SHOW_DISTRIBUTION = True
DEFAULT_DYNAMIC_ATYPE_CTYPE = False


def query_FLiES(filename, ctype, atype, ctop_index, albedo_index, The0_index, ctauref_index, tauref_index):
    with netCDF4.Dataset(filename, 'r') as f:
        return f['SWin'][0][ctype, atype, ctop_index, albedo_index, The0_index, ctauref_index, tauref_index]


def interpolate_The0(filename, ctype, atype, ctop_index, albedo_index, The0, ctauref_index, tauref_index):
    # constrain solar zenith angle
    The0 = rt.clip(The0, 5, 85)

    # get low index for solar zenith angle
    The0_index_low = np.clip(np.floor(The0 / 5.0).astype(np.int32) - 1, 0, 16).astype(np.int32)

    # get high index for solar zenith angle
    The0_index_high = np.clip(np.ceil(The0 / 5.0).astype(np.int32) - 1, 0, 16).astype(np.int32)

    # query closest incoming shortwave
    SWin_The0_low = query_FLiES(filename, ctype, atype, ctop_index, albedo_index, The0_index_low, ctauref_index,
                                tauref_index)

    # query next closest incoming shortwave
    SWin_The0_high = query_FLiES(filename, ctype, atype, ctop_index, albedo_index, The0_index_high, ctauref_index,
                                 tauref_index)

    The0_slope = (SWin_The0_high - SWin_The0_low) / 5.0
    The0_intermediate = The0 - np.floor(The0 / 5.0) * 5.0
    SWin = SWin_The0_low + The0_slope * The0_intermediate

    return SWin


def interploate_ctauref(
        filename,
        ctype,
        atype,
        ctop_index,
        albedo_index,
        The0,
        ctauref,
        tauref_index):
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


def FLiES_lookup(
        ctype: np.ndarray,
        atype: np.ndarray,
        ctop: np.ndarray,
        albedo: np.ndarray,
        The0: np.ndarray,
        ctauref: np.ndarray,
        tauref: np.ndarray,
        LUT_filename: str = None,
        interpolate_cot: bool = True):
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
        SWin = interploate_ctauref(
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

    return SWin  # , ctop_index, albedo_index, ctauref_index, tauref_index


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

def process_FLiES_LUT_raster(
        geometry: RasterGeometry,
        time_UTC: Union[datetime, str],
        cloud_mask: Raster = None,
        COT: Raster = None,
        koppen_geiger: Raster = None,
        IGBP: Raster = None,
        cloud_top: Raster = None,
        albedo: Raster = None,
        SZA: Raster = None,
        AOT: Raster = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_directory: str = "."):
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    date_UTC = time_UTC.date()
    time_solar = UTC_to_solar(time_UTC, lon=geometry.centroid_latlon.x)
    date_solar = time_solar.date()
    day_of_year = date_solar.timetuple().tm_yday

    if cloud_mask is None:
        cloud_mask = np.full(geometry.shape, 0)
    else:
        cloud_mask = np.array(cloud_mask)

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP(working_directory=GEOS5FP_directory)

    if COT is None:
        COT = GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)

    COT = np.clip(COT, 0, None)
    COT = np.where(COT < 0.001, 0, COT)
    COT = np.array(COT)

    if koppen_geiger is None:
        koppen_geiger = load_koppen_geiger(geometry=geometry)

    koppen_geiger = np.array(koppen_geiger)
    
    if IGBP is None:
        IGBP = load_MCD12C1_IGBP(geometry=geometry)
    
    IGBP = np.array(IGBP)

    if cloud_top is None:
        cloud_top = np.full(geometry.shape, np.nan)
    else:
        cloud_top = np.array(cloud_top)

    albedo = np.array(albedo)

    if SZA is None:
        # SZA = GEOS5FP_connection.SZA(day_of_year=day_of_year, hour_of_day=hour_of_day, geometry=geometry)
        SZA = calculate_SZA_from_DOY_and_hour(
            lat=geometry.lat,
            lon=geometry.lon,
            DOY=day_of_year, 
            hour=time_solar.hour
        )

    SZA = np.array(SZA)

    if AOT is None:
        AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)

    AOT = np.array(AOT)

    SWin = FLiES_LUT(
        doy=day_of_year,
        cloud_mask=cloud_mask,
        COT=COT,
        koppen_geiger=koppen_geiger,
        IGBP=IGBP,
        cloud_top=cloud_top,
        albedo=albedo,
        SZA=SZA,
        AOT=AOT
    )

    SWin = Raster(SWin, geometry=geometry)

    return SWin
