from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import h5py
import numpy as np
import requests
import os
from typing import Optional

app = FastAPI(title="SMAP Ireland Real Extractor")

class ExtractRequest(BaseModel):
    date: str = "2026-05-18"
    download_url: Optional[str] = None   # Pass from Supabase Edge Function
    nasa_token: Optional[str] = None

# Ireland Bounding Box
IRELAND_BBOX = (-11.0, 51.3, -5.3, 55.5)

@app.get("/")
async def home():
    return {"message": "SMAP Ireland Real Extractor Running"}

@app.post("/extract-ireland")
async def extract_ireland(request: ExtractRequest):
    try:
        download_url = request.download_url
        token = request.nasa_token or os.getenv("NASA_EARTHDATA_TOKEN")

        if not download_url:
            raise HTTPException(status_code=400, detail="download_url is required")

        if not token:
            raise HTTPException(status_code=400, detail="NASA token is required")

        # ====================== DOWNLOAD FILE ======================
        print(f"Downloading: {download_url}")
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(download_url, headers=headers, stream=True, timeout=120)
        response.raise_for_status()

        # Save to disk (better for memory management)
        filename = "smap_temp.h5"
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=16*1024*1024):
                f.write(chunk)

        # ====================== REAL EXTRACTION ======================
        with h5py.File(filename, 'r') as f:
            g = f['Geophysical_Data']

            sm_surface = g['sm_surface'][:]
            sm_rootzone = g['sm_rootzone'][:]
            lat = g['latitude'][:]
            lon = g['longitude'][:]

            # Mask for Ireland
            mask = (
                (lat >= IRELAND_BBOX[1]) & (lat <= IRELAND_BBOX[3]) &
                (lon >= IRELAND_BBOX[0]) & (lon <= IRELAND_BBOX[2])
            )

            ireland_surface = sm_surface[mask]
            ireland_rootzone = sm_rootzone[mask]

            result = {
                "success": True,
                "date": request.date,
                "region": "Ireland",
                "soil_moisture": {
                    "sm_surface": {
                        "mean": float(np.nanmean(ireland_surface)),
                        "median": float(np.nanmedian(ireland_surface)),
                        "min": float(np.nanmin(ireland_surface)),
                        "max": float(np.nanmax(ireland_surface)),
                        "unit": "m³/m³"
                    },
                    "sm_rootzone": {
                        "mean": float(np.nanmean(ireland_rootzone)),
                        "median": float(np.nanmedian(ireland_rootzone)),
                        "min": float(np.nanmin(ireland_rootzone)),
                        "max": float(np.nanmax(ireland_rootzone)),
                        "unit": "m³/m³"
                    }
                },
                "valid_pixels_percent": float((~np.isnan(ireland_surface)).mean() * 100)
            }

        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

        return result

    except Exception as e:
        return {"success": False, "error": str(e)}
