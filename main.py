from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import h5py
import numpy as np
import requests
import os

app = FastAPI(
    title="SMAP Scientific Extractor"
)

# =========================================================
# REQUEST MODELS
# =========================================================

class IrelandExtractRequest(BaseModel):

    date: str = "2026-05-18"

    download_url: Optional[str] = None

    nasa_token: Optional[str] = None


class PointExtractRequest(BaseModel):

    lat: float

    lon: float

    date: str = "2026-05-18"

    download_url: Optional[str] = None

    nasa_token: Optional[str] = None


# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    return {
        "message":
        "SMAP Scientific Extractor Running"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
async def health():

    return {
        "healthy": True
    }


# =========================================================
# CLEAN VALUES
# =========================================================

def clean_array(arr):

    arr = np.where(
        arr < -9990,
        np.nan,
        arr
    )

    arr = np.where(
        (arr < 0) |
        (arr > 1),
        np.nan,
        arr
    )

    return arr


def clean_scalar(v):

    if v is None:
        return None

    if np.isnan(v):
        return None

    if v < 0 or v > 1:
        return None

    return float(v)


# =========================================================
# DOWNLOAD HDF5
# =========================================================

def download_hdf5(
    download_url,
    token,
    filename
):

    headers = {
        "Authorization":
        f"Bearer {token}"
    }

    response = requests.get(
        download_url,
        headers=headers,
        stream=True,
        timeout=180
    )

    response.raise_for_status()

    with open(filename, "wb") as f:

        for chunk in response.iter_content(
            chunk_size=16 * 1024 * 1024
        ):

            if chunk:
                f.write(chunk)


# =========================================================
# FIND NEAREST PIXEL
# =========================================================

def find_nearest_pixel(
    target_lat,
    target_lon,
    lat_grid,
    lon_grid
):

    distance = np.sqrt(
        (lat_grid - target_lat) ** 2 +
        (lon_grid - target_lon) ** 2
    )

    flat_index = np.nanargmin(distance)

    row, col = np.unravel_index(
        flat_index,
        distance.shape
    )

    nearest_distance = float(
        distance[row, col]
    )

    return row, col, nearest_distance


# =========================================================
# REGIONAL EXTRACTION
# =========================================================

@app.post("/extract-ireland")
async def extract_ireland(
    request: IrelandExtractRequest
):

    filename = "smap_ireland.h5"

    try:

        download_url = request.download_url

        token = (
            request.nasa_token
            or
            os.getenv("NASA_EARTHDATA_TOKEN")
        )

        if not download_url:

            raise HTTPException(
                status_code=400,
                detail="download_url is required"
            )

        if not token:

            raise HTTPException(
                status_code=400,
                detail="NASA token is required"
            )

        print("Downloading HDF5...")

        download_hdf5(
            download_url,
            token,
            filename
        )

        print("Opening HDF5...")

        with h5py.File(
            filename,
            "r"
        ) as f:

            g = f["Geophysical_Data"]

            sm_surface = clean_array(
                g["sm_surface"][:]
            )

            sm_rootzone = clean_array(
                g["sm_rootzone"][:]
            )

            ireland_surface = sm_surface[
                140:190,
                1750:1900
            ]

            ireland_rootzone = sm_rootzone[
                140:190,
                1750:1900
            ]

            return {

                "success": True,

                "date":
                    request.date,

                "region":
                    "Ireland",

                "soil_moisture": {

                    "sm_surface": {

                        "mean":
                            float(
                                np.nanmean(
                                    ireland_surface
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    ireland_surface
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    ireland_surface
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    ireland_surface
                                )
                            ),

                        "unit":
                            "m³/m³"
                    },

                    "sm_rootzone": {

                        "mean":
                            float(
                                np.nanmean(
                                    ireland_rootzone
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    ireland_rootzone
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    ireland_rootzone
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    ireland_rootzone
                                )
                            ),

                        "unit":
                            "m³/m³"
                    }
                },

                "source":
                    "SMAP L4 Regional Extraction"
            }

    except Exception as e:

        return {

            "success": False,

            "error":
                str(e)
        }

    finally:

        if os.path.exists(filename):

            try:
                os.remove(filename)
            except:
                pass


# =========================================================
# TRUE POINT EXTRACTION
# =========================================================

@app.post("/extract-point")
async def extract_point(
    request: PointExtractRequest
):

    filename = "smap_point.h5"

    try:

        download_url = request.download_url

        token = (
            request.nasa_token
            or
            os.getenv("NASA_EARTHDATA_TOKEN")
        )

        if not download_url:

            raise HTTPException(
                status_code=400,
                detail="download_url is required"
            )

        if not token:

            raise HTTPException(
                status_code=400,
                detail="NASA token is required"
            )

        print("Downloading HDF5...")

        download_hdf5(
            download_url,
            token,
            filename
        )

        print("Opening HDF5...")

        with h5py.File(
            filename,
            "r"
        ) as f:

            g = f["Geophysical_Data"]

            # =================================================
            # LOAD VARIABLES
            # =================================================

            sm_surface = g["sm_surface"][:]

            sm_rootzone = g["sm_rootzone"][:]

            latitude = g["latitude"][:]

            longitude = g["longitude"][:]

            # =================================================
            # FIND NEAREST PIXEL
            # =================================================

            row, col, nearest_distance = (
                find_nearest_pixel(
                    request.lat,
                    request.lon,
                    latitude,
                    longitude
                )
            )

            print(
                f"Nearest pixel: row={row}, col={col}"
            )

            # =================================================
            # EXTRACT VALUES
            # =================================================

            surface_value = clean_scalar(
                sm_surface[row, col]
            )

            rootzone_value = clean_scalar(
                sm_rootzone[row, col]
            )

            return {

                "success": True,

                "date":
                    request.date,

                "location": {

                    "lat":
                        request.lat,

                    "lon":
                        request.lon
                },

                "nearest_pixel": {

                    "row":
                        int(row),

                    "col":
                        int(col),

                    "distance_degrees":
                        round(
                            nearest_distance,
                            5
                        ),

                    "pixel_lat":
                        float(
                            latitude[row, col]
                        ),

                    "pixel_lon":
                        float(
                            longitude[row, col]
                        )
                },

                "soil_moisture": {

                    "sm_surface": {

                        "value":
                            surface_value,

                        "unit":
                            "m³/m³"
                    },

                    "sm_rootzone": {

                        "value":
                            rootzone_value,

                        "unit":
                            "m³/m³"
                    }
                },

                "source":
                    "SMAP L4 Scientific Point Extraction"
            }

    except Exception as e:

        return {

            "success": False,

            "error":
                str(e)
        }

    finally:

        if os.path.exists(filename):

            try:
                os.remove(filename)
            except:
                pass
