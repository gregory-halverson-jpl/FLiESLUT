from datetime import datetime, date
from typing import Union

import numpy as np
from dateutil import parser
from rasters import RasterGeometry, Raster, MultiPoint
from solar_apparent_time import UTC_to_solar, calculate_solar_day_of_year
from sun_angles import calculate_SZA_from_DOY_and_hour

from .FLiES_lookup import FLiES_lookup

try:
    from MCD12C1_2019_v006 import load_MCD12C1_IGBP
    from GEOS5FP import GEOS5FP
    from koppengeiger import load_koppen_geiger
    _HAS_RASTER_DEPS = True
except ImportError:
    _HAS_RASTER_DEPS = False

def process_FLiESLUT(
        DOY: Union[np.ndarray, int, float] = None,
        cloud_mask: Union[np.ndarray, Raster] = None,
        COT: Union[np.ndarray, Raster] = None,
        koppen_geiger: Union[np.ndarray, Raster] = None,
        IGBP: Union[np.ndarray, Raster] = None,
        cloud_top: Union[np.ndarray, Raster] = None,
        albedo: Union[np.ndarray, Raster] = None,
        SZA: Union[np.ndarray, Raster] = None,
        AOT: Union[np.ndarray, Raster] = None,
        time_UTC: Union[datetime, str] = None,
        geometry: Union[RasterGeometry, Raster, MultiPoint] = None,
        GEOS5FP_connection=None) -> Union[np.ndarray, Raster]:
    """Processes the FLiES look-up table to calculate shortwave incoming radiation.

    This function uses the FLiES look-up table implementation to calculate `SWin` based on various
    input parameters. It can handle both numpy arrays and Raster objects as inputs, returning 
    the same type as provided. When using Raster inputs, additional parameters for time and 
    geometry are required, and missing raster data can be automatically retrieved.

    Args:
      DOY: Day of year (or days of year). If not provided, calculated from time_UTC and geometry.
      cloud_mask: Boolean array/raster indicating cloud presence (True for cloudy, False for clear).
      COT: Cloud optical thickness (or thicknesses).
      koppen_geiger: Koppen-Geiger climate classification code (or codes).
      IGBP: International Geosphere-Biosphere Programme land cover classification code (or codes).
      cloud_top: Cloud top pressure level (or levels) in Pa.
      albedo: Surface albedo (or albedos).
      SZA: Solar zenith angle (or angles) in degrees.
      AOT: Aerosol optical thickness (or thicknesses).
      time_UTC: The time in UTC. Required if DOY not provided or when using Raster inputs.
      geometry: The raster geometry, Raster, or MultiPoint. Required when using Raster inputs or calculating DOY.
      GEOS5FP_connection: Optional GEOS5FP connection for retrieving missing data.

    Returns:
      A NumPy array or Raster object containing the calculated `SWin` values.
    """

    # Calculate DOY if not provided
    if DOY is None:
        if time_UTC is None:
            raise ValueError("Either DOY or time_UTC must be provided")
        if geometry is None:
            raise ValueError("geometry parameter is required when calculating DOY from time_UTC")
        
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)
        
        # Calculate solar day of year using geometry centroid
        if hasattr(geometry, 'centroid_latlon'):
            lon = geometry.centroid_latlon.x
        elif hasattr(geometry, 'lon'):
            lon = geometry.lon
        else:
            raise ValueError("geometry must have either centroid_latlon.x or lon attribute")
            
        DOY = calculate_solar_day_of_year(time_UTC=time_UTC, lon=lon)

    # Detect if we're in raster mode (any input is a Raster object)
    raster_mode = any(isinstance(param, Raster) for param in [
        cloud_mask, COT, koppen_geiger, IGBP, cloud_top, albedo, SZA, AOT
    ])
    
    if raster_mode:
        # Raster mode - handle raster processing
        if not _HAS_RASTER_DEPS:
            raise ImportError("Raster dependencies not available. Install MCD12C1_2019_v006, GEOS5FP, and koppengeiger packages.")
        
        if geometry is None:
            raise ValueError("geometry parameter is required when using Raster inputs")
        if time_UTC is None:
            raise ValueError("time_UTC parameter is required when using Raster inputs")
            
        # Process time for raster mode calculations
        if isinstance(time_UTC, str):
            time_UTC = parser.parse(time_UTC)

        date_UTC: date = time_UTC.date()
        time_solar = UTC_to_solar(time_UTC, lon=geometry.centroid_latlon.x)
        date_solar: date = time_solar.date()

        # Handle raster inputs and convert to arrays
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
                DOY=DOY,
                hour=time_solar.hour
            )

        SZA = np.array(SZA)

        if AOT is None:
            AOT = GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)

        AOT = np.array(AOT)

    # Common processing for both modes
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
    SWin_toa = 1370 * (1 + 0.033 * np.cos(2 * np.pi * DOY / 365.0)) * np.sin(np.radians(90 - SZA))
    SWin = np.clip(SWin, None, SWin_toa)

    # mask SWin to COT to prevent garbage data in the gap between swaths
    SWin = np.where(np.isnan(COT), np.nan, SWin)

    # Return appropriate type based on input mode
    if raster_mode:
        return Raster(SWin, geometry=geometry)
    else:
        return SWin