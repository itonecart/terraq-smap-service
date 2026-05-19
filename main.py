from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

import h5py
import numpy as np
import requests
import os

app = FastAPI(
    title="SMAP Ireland Real Extractor"
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
        "SMAP Ireland Real Extractor Running"
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
# APPROXIMATE EASE2 CONVERSION
# =========================================================
#
# Temporary approximation for Ireland.
# Later we can replace with exact EASE2 projection.
#
# =========================================================

def latlon_to_smap_index(
    lat,
    lon
):

    # Approximate Ireland mapping

    row = int(
        162 - ((lat - 51.0) * 8)
    )

    col = int(
        1800 + ((lon + 10.5) * 18)
    )

    return row, col


# =========================================================
# CLEAN VALUES
# =========================================================

def clean_values(arr):

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
# IRELAND REGIONAL EXTRACTION
# =========================================================

@app.post("/extract-ireland")
async def extract_ireland(
    request: IrelandExtractRequest
):

    filename = "smap_temp.h5"

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

        print(
            f"Downloading: {download_url}"
        )

        download_hdf5(
            download_url,
            token,
            filename
        )

        print(
            "Download complete"
        )

        with h5py.File(
            filename,
            "r"
        ) as f:

            g = f["Geophysical_Data"]

            sm_surface = g[
                "sm_surface"
            ][:]

            sm_rootzone = g[
                "sm_rootzone"
            ][:]

            ireland_surface = sm_surface[
                140:190,
                1750:1900
            ]

            ireland_rootzone = sm_rootzone[
                140:190,
                1750:1900
            ]

            ireland_surface = clean_values(
                ireland_surface
            )

            ireland_rootzone = clean_values(
                ireland_rootzone
            )

            surface_valid = (
                ireland_surface.astype(
                    np.float64
                )
            )

            rootzone_valid = (
                ireland_rootzone.astype(
                    np.float64
                )
            )

            result = {

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
                                    surface_valid
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    surface_valid
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    surface_valid
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    surface_valid
                                )
                            ),

                        "unit":
                            "m³/m³"
                    },

                    "sm_rootzone": {

                        "mean":
                            float(
                                np.nanmean(
                                    rootzone_valid
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    rootzone_valid
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    rootzone_valid
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    rootzone_valid
                                )
                            ),

                        "unit":
                            "m³/m³"
                    }
                },

                "valid_pixels_percent":
                    round(
                        float(
                            (
                                ~np.isnan(
                                    surface_valid
                                )
                            ).mean() * 100
                        ),
                        2
                    ),

                "source":
                    "SMAP L4 HDF5 Extraction",

                "resolution":
                    "Approximate Ireland EASE2 regional slice"
            }

            return result

    except Exception as e:

        return {

            "success": False,

            "error":
                str(e)
        }

    finally:

        if os.path.exists(
            filename
        ):

            try:

                os.remove(
                    filename
                )

            except:
                pass


# =========================================================
# POINT EXTRACTION
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

        # =================================================
        # CONVERT LAT/LON
        # =================================================

        row, col = latlon_to_smap_index(
            request.lat,
            request.lon
        )

        print(
            f"SMAP index: row={row}, col={col}"
        )

        # =================================================
        # DOWNLOAD FILE
        # =================================================

        download_hdf5(
            download_url,
            token,
            filename
        )

        # =================================================
        # OPEN HDF5
        # =================================================

        with h5py.File(
            filename,
            "r"
        ) as f:

            g = f["Geophysical_Data"]

            sm_surface = g[
                "sm_surface"
            ]

            sm_rootzone = g[
                "sm_rootzone"
            ]

            surface_value = float(
                sm_surface[row, col]
            )

            rootzone_value = float(
                sm_rootzone[row, col]
            )

            # Clean invalid

            if (
                surface_value < 0
                or
                surface_value > 1
            ):

                surface_value = None

            if (
                rootzone_value < 0
                or
                rootzone_value > 1
            ):

                rootzone_value = None

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

                "smap_index": {

                    "row":
                        row,

                    "col":
                        col
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
                    "SMAP L4 Point Extraction",

                "note":
                    "Approximate Ireland EASE2 conversion"
            }

    except Exception as e:

        return {

            "success": False,

            "error":
                str(e)
        }

    finally:

        if os.path.exists(
            filename
        ):

            try:

                os.remove(
                    filename
                )

            except:
                pass
