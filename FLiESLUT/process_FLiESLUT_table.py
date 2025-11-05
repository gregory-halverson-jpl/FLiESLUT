import logging

import numpy as np
import pandas as pd
import rasters as rt
from dateutil import parser
from pandas import DataFrame
from rasters import MultiPoint, WGS84
from shapely.geometry import Point
from GEOS5FP import GEOS5FP
from .process_FLiESLUT import process_FLiESLUT

logger = logging.getLogger(__name__)

def process_FLiESLUT_table(
        input_df: DataFrame,
        GEOS5FP_connection: GEOS5FP = None,
        row_wise: bool = False) -> DataFrame:
    """
    Processes a DataFrame of FLiESLUT inputs and returns a DataFrame with FLiESLUT outputs.

    Parameters:
    input_df (pd.DataFrame): A DataFrame containing the following columns:
        - time_UTC (str or datetime): Time in UTC.
        - geometry (str or shapely.geometry.Point) or (lat, lon): Spatial coordinates. If "geometry" is a string, it should be in WKT format (e.g., "POINT (lon lat)").
        - DOY (int, optional): Day of the year. If not provided, it will be derived from "time_UTC".
        - cloud_mask (bool, optional): Boolean indicating cloud presence (True for cloudy, False for clear).
        - COT (float, optional): Cloud optical thickness.
        - koppen_geiger (int, optional): Koppen-Geiger climate classification code.
        - IGBP (int, optional): International Geosphere-Biosphere Programme land cover classification code.
        - cloud_top (float, optional): Cloud top pressure level in Pa.
        - albedo (float): Surface albedo.
        - SZA (float, optional): Solar zenith angle in degrees.
        - AOT (float, optional): Aerosol optical thickness.
    GEOS5FP_connection (GEOS5FP, optional): Connection object for GEOS-5 FP data.
    row_wise (bool, optional): If True (default), processes each row individually. If False, 
        attempts vectorized processing when possible for better performance.

    Returns:
    pd.DataFrame: A DataFrame with the same structure as the input, but with additional columns:
        - SWin: Shortwave incoming radiation calculated using FLiESLUT.

    Raises:
    KeyError: If required columns ("geometry" or "lat" and "lon") are missing.
    """

    def ensure_geometry(row):
        if "geometry" in row:
            if isinstance(row.geometry, str):
                s = row.geometry.strip()
                if s.startswith("POINT"):
                    coords = s.replace("POINT", "").replace("(", "").replace(")", "").strip().split()
                    return Point(float(coords[0]), float(coords[1]))
                elif "," in s:
                    coords = [float(c) for c in s.split(",")]
                    return Point(coords[0], coords[1])
                else:
                    coords = [float(c) for c in s.split()]
                    return Point(coords[0], coords[1])
        return row.geometry

    logger.info("started processing FLiESLUT input table")

    # Ensure geometry column is properly formatted
    input_df = input_df.copy()
    input_df["geometry"] = input_df.apply(ensure_geometry, axis=1)

    # Prepare output DataFrame
    output_df = input_df.copy()

    if row_wise:
        # Process each row individually (original behavior)
        logger.info("processing table row-wise")
        results = []
        for _, row in input_df.iterrows():
            if "geometry" in row:
                geometry = rt.Point((row.geometry.x, row.geometry.y), crs=WGS84)
            elif "lat" in row and "lon" in row:
                geometry = rt.Point((row.lon, row.lat), crs=WGS84)
            else:
                raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

            time_UTC = pd.to_datetime(row.time_UTC)
            DOY = row.DOY if "DOY" in row else time_UTC.timetuple().tm_yday

            logger.info(f"processing row with time_UTC: {time_UTC}, geometry: {geometry}")

            # Helper function to safely extract scalar values
            def safe_extract(value):
                if value is None:
                    return None
                if hasattr(value, 'item'):
                    return value.item()
                return value

            SWin_result = process_FLiESLUT(
                DOY=DOY,
                cloud_mask=safe_extract(row.get("cloud_mask")),
                COT=safe_extract(row.get("COT")),
                koppen_geiger=safe_extract(row.get("koppen_geiger")),
                IGBP=safe_extract(row.get("IGBP")),
                cloud_top=safe_extract(row.get("cloud_top")),
                albedo=safe_extract(row.albedo),
                SZA=safe_extract(row.get("SZA")),
                AOT=safe_extract(row.get("AOT")),
                time_UTC=time_UTC,
                geometry=geometry,
                GEOS5FP_connection=GEOS5FP_connection
            )

            results.append(SWin_result)

        # Add results to the output DataFrame
        output_df["SWin"] = results
    else:
        # Vectorized processing for better performance
        logger.info("processing table in vectorized mode")
        
        # Prepare geometries
        if "geometry" in input_df.columns:
            geometries = MultiPoint([(geom.x, geom.y) for geom in input_df.geometry], crs=WGS84)
        elif "lat" in input_df.columns and "lon" in input_df.columns:
            geometries = MultiPoint([(lon, lat) for lon, lat in zip(input_df.lon, input_df.lat)], crs=WGS84)
        else:
            raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")
        
        # Convert time column to datetime
        times_UTC = pd.to_datetime(input_df.time_UTC)
        
        logger.info(f"processing {len(input_df)} rows in vectorized mode")

        # Helper function to get column values or None if column doesn't exist
        def get_column_or_none(df, col_name, default_col_name=None):
            if col_name in df.columns:
                return df[col_name].values
            elif default_col_name and default_col_name in df.columns:
                return df[default_col_name].values
            else:
                return None

        # Calculate DOY array
        DOY_array = []
        for time_UTC in times_UTC:
            if "DOY" in input_df.columns:
                DOY_array = input_df.DOY.values
                break
            else:
                DOY_array.append(time_UTC.timetuple().tm_yday)
        
        if not isinstance(DOY_array, np.ndarray):
            DOY_array = np.array(DOY_array)

        # Process all rows at once using vectorized process_FLiESLUT call
        # Note: We'll process the first row to get the time_UTC and geometry for the function
        # since the function expects a single time_UTC and geometry for raster mode
        if len(input_df) > 1:
            # For multiple rows, we need to process individually due to process_FLiESLUT design
            logger.warning("Vectorized mode with multiple rows not fully supported by process_FLiESLUT, falling back to row-wise processing")
            results = []
            for i, (_, row) in enumerate(input_df.iterrows()):
                if "geometry" in row:
                    geometry = rt.Point((row.geometry.x, row.geometry.y), crs=WGS84)
                elif "lat" in row and "lon" in row:
                    geometry = rt.Point((row.lon, row.lat), crs=WGS84)
                else:
                    raise KeyError("Input DataFrame must contain either 'geometry' or both 'lat' and 'lon' columns.")

                time_UTC = times_UTC.iloc[i]
                DOY = DOY_array[i]

                # Helper function to safely extract scalar values
                def safe_extract(value):
                    if value is None:
                        return None
                    if hasattr(value, 'item'):
                        return value.item()
                    return value

                SWin_result = process_FLiESLUT(
                    DOY=DOY,
                    cloud_mask=safe_extract(row.get("cloud_mask")),
                    COT=safe_extract(row.get("COT")),
                    koppen_geiger=safe_extract(row.get("koppen_geiger")),
                    IGBP=safe_extract(row.get("IGBP")),
                    cloud_top=safe_extract(row.get("cloud_top")),
                    albedo=safe_extract(row.albedo),
                    SZA=safe_extract(row.get("SZA")),
                    AOT=safe_extract(row.get("AOT")),
                    time_UTC=time_UTC,
                    geometry=geometry,
                    GEOS5FP_connection=GEOS5FP_connection
                )
                results.append(SWin_result)
            
            output_df["SWin"] = results
        else:
            # Single row case
            row = input_df.iloc[0]
            if "geometry" in input_df.columns:
                geometry = rt.Point((input_df.geometry.iloc[0].x, input_df.geometry.iloc[0].y), crs=WGS84)
            else:
                geometry = rt.Point((input_df.lon.iloc[0], input_df.lat.iloc[0]), crs=WGS84)
            
            time_UTC = times_UTC.iloc[0]
            DOY = DOY_array[0]

            # Helper function to safely extract scalar values
            def safe_extract(value):
                if value is None:
                    return None
                if hasattr(value, 'item'):
                    return value.item()
                return value

            SWin_result = process_FLiESLUT(
                DOY=DOY,
                cloud_mask=safe_extract(row.get("cloud_mask")),
                COT=safe_extract(row.get("COT")),
                koppen_geiger=safe_extract(row.get("koppen_geiger")),
                IGBP=safe_extract(row.get("IGBP")),
                cloud_top=safe_extract(row.get("cloud_top")),
                albedo=safe_extract(row.albedo),
                SZA=safe_extract(row.get("SZA")),
                AOT=safe_extract(row.get("AOT")),
                time_UTC=time_UTC,
                geometry=geometry,
                GEOS5FP_connection=GEOS5FP_connection
            )
            
            output_df["SWin"] = [SWin_result]

    logger.info("completed processing FLiESLUT input table")

    return output_df