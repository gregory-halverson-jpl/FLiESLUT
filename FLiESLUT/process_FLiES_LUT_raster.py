from datetime import datetime, date
from typing import Union

import numpy as np
from MCD12C1_2019_v006 import load_MCD12C1_IGBP
from dateutil import parser
from GEOS5FP import GEOS5FP
from koppengeiger import load_koppen_geiger
from rasters import RasterGeometry, Raster
from solar_apparent_time import UTC_to_solar
from sun_angles import calculate_SZA_from_DOY_and_hour

from .process_FLiES_LUT import process_FLiES_LUT


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
        GEOS5FP_connection: GEOS5FP = None) -> Raster:
    """Calculates shortwave incoming radiation for a raster using the FLiES look-up table.

    This function processes a raster to calculate `SWin` using the FLiES look-up table. It handles
    various input rasters, including cloud mask, cloud optical thickness, Koppen-Geiger climate
    classification, IGBP land cover, cloud top pressure, albedo, solar zenith angle, and aerosol
    optical thickness. It can also use a `GEOS5FP` connection to retrieve missing data.

    Args:
      geometry: The raster geometry defining the spatial extent and resolution.
      time_UTC: The time in UTC for the calculation, either as a datetime object or a string.
      cloud_mask: Optional cloud mask raster (True for cloudy, False for clear).
      COT: Optional cloud optical thickness raster.
      koppen_geiger: Optional Koppen-Geiger climate classification raster.
      IGBP: Optional IGBP land cover classification raster.
      cloud_top: Optional cloud top pressure level raster in Pa.
      albedo:  Optional surface albedo raster.
      SZA: Optional solar zenith angle raster in degrees.
      AOT: Optional aerosol optical thickness raster.
      GEOS5FP_connection: Optional `GEOS5FP` connection object for retrieving missing data.

    Returns:
      A `Raster` object containing the calculated `SWin` values.
    """
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    date_UTC: date = time_UTC.date()
    time_solar = UTC_to_solar(time_UTC, lon=geometry.centroid_latlon.x)
    date_solar: date = time_solar.date()
    day_of_year = date_solar.timetuple().tm_yday

    if cloud_mask is None:
        cloud_mask = np.full(geometry.shape, 0)
    else:
        cloud_mask = np.array(cloud_mask)

    if GEOS5FP_connection is None:
        GEOS5FP_connection = GEOS5FP()

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

    SWin = process_FLiES_LUT(
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
