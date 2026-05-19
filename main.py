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

# ─────────────────────────────────────
# REQUEST MODEL
# ─────────────────────────────────────

class ExtractRequest(BaseModel):

    date: str = "2026-05-18"

    # Passed from Supabase Edge Function
    download_url: Optional[str] = None

    nasa_token: Optional[str] = None


# ─────────────────────────────────────
# HOME
# ─────────────────────────────────────

@app.get("/")
async def home():

    return {
        "message":
        "SMAP Ireland Real Extractor Running"
    }


# ─────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────

@app.get("/health")
async def health():

    return {
        "healthy": True
    }


# ─────────────────────────────────────
# REAL SMAP EXTRACTION
# ─────────────────────────────────────

@app.post("/extract-ireland")
async def extract_ireland(
    request: ExtractRequest
):

    filename = "smap_temp.h5"

    try:

        # ─────────────────────────────
        # INPUTS
        # ─────────────────────────────

        download_url = (
            request.download_url
        )

        token = (
            request.nasa_token
            or
            os.getenv(
                "NASA_EARTHDATA_TOKEN"
            )
        )

        if not download_url:

            raise HTTPException(
                status_code=400,
                detail=
                "download_url is required"
            )

        if not token:

            raise HTTPException(
                status_code=400,
                detail=
                "NASA token is required"
            )

        # ─────────────────────────────
        # DOWNLOAD HDF5
        # ─────────────────────────────

        print(
            f"Downloading: {download_url}"
        )

        headers = {
            "Authorization":
            f"Bearer {token}"
        }

        response = requests.get(
            download_url,
            headers=headers,
            stream=True,
            timeout=120
        )

        response.raise_for_status()

        with open(
            filename,
            "wb"
        ) as f:

            for chunk in response.iter_content(
                chunk_size=16 * 1024 * 1024
            ):

                if chunk:
                    f.write(chunk)

        print(
            "Download complete"
        )

        # ─────────────────────────────
        # OPEN HDF5
        # ─────────────────────────────

        with h5py.File(
            filename,
            "r"
        ) as f:

            print(
                "Opened HDF5 successfully"
            )

            g = f[
                "Geophysical_Data"
            ]

            # Real SMAP variables

            sm_surface = g[
                "sm_surface"
            ][:]

            sm_rootzone = g[
                "sm_rootzone"
            ][:]

            # ─────────────────────────
            # TEMPORARY IRELAND REGION
            # ─────────────────────────
            #
            # Using approximate EASE2
            # regional slice for Ireland
            #
            # Exact polygon masking
            # can be added later.
            #
            # ─────────────────────────

            ireland_surface = (
                sm_surface[
                    140:190,
                    1750:1900
                ]
            )

            ireland_rootzone = (
                sm_rootzone[
                    140:190,
                    1750:1900
                ]
            )

            # ─────────────────────────
            # CLEAN INVALID VALUES
            # ─────────────────────────

            ireland_surface = np.where(
                ireland_surface < -9990,
                np.nan,
                ireland_surface
            )

            ireland_rootzone = np.where(
                ireland_rootzone < -9990,
                np.nan,
                ireland_rootzone
            )

            # ─────────────────────────
            # STATISTICS
            # ─────────────────────────

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
                                    ireland_surface.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    ireland_surface.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    ireland_surface.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    ireland_surface.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "unit":
                            "m³/m³"
                    },

                    "sm_rootzone": {

                        "mean":
                            float(
                                np.nanmean(
                                    ireland_rootzone.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "median":
                            float(
                                np.nanmedian(
                                    ireland_rootzone.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "min":
                            float(
                                np.nanmin(
                                    ireland_rootzone.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "max":
                            float(
                                np.nanmax(
                                    ireland_rootzone.astype(
                                        np.float64
                                    )
                                )
                            ),

                        "unit":
                            "m³/m³"
                    }
                },

                "valid_pixels_percent":
                    float(
                        (
                            ~np.isnan(
                                ireland_surface
                            )
                        ).mean() * 100
                    ),

                "source":
                    "SMAP L4 HDF5 Extraction",

                "note":
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

        # ─────────────────────────────
        # CLEAN TEMP FILE
        # ─────────────────────────────

        if os.path.exists(
            filename
        ):

            try:

                os.remove(
                    filename
                )

                print(
                    "Temporary file removed"
                )

            except Exception as cleanup_error:

                print(
                    f"Cleanup error: {cleanup_error}"
                )
