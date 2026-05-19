from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import h5py
import numpy as np
import requests
import os
import uuid
import hashlib
from datetime import datetime, timezone
import math

app = FastAPI(title="SMAP Scientific Extractor")

# =========================================================
# CACHE
# =========================================================

CACHE_DIR = "/tmp/smap_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_hdf5_path(download_url: str) -> str:
    url_hash = hashlib.md5(download_url.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"smap_{url_hash}.h5")

def ensure_downloaded(download_url: str, token: str) -> str:
    path = cached_hdf5_path(download_url)
    if os.path.exists(path):
        print(f"Cache hit: {path}")
        return path
    tmp_path = path + f".{uuid.uuid4().hex}.tmp"
    print(f"Downloading HDF5...")
    try:
        r = requests.get(
            download_url,
            headers={"Authorization": f"Bearer {token}"},
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.rename(tmp_path, path)
        print(f"Cached: {path}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


# =========================================================
# EASE-2 GLOBAL 9 km GRID  (EPSG:6933)
# =========================================================

EASE2_ROWS      = 1624
EASE2_COLS      = 3856
EASE2_MAP_SCALE = 9008.055210
EASE2_R0        = (EASE2_ROWS - 1) / 2
EASE2_S0        = (EASE2_COLS - 1) / 2
EASE2_R_MAJOR   = 6378137.0
EASE2_COS_LAT0  = math.cos(math.radians(30))

def latlon_to_ease2(lat: float, lon: float) -> tuple[int, int]:
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    x = EASE2_R_MAJOR * lon_r * EASE2_COS_LAT0
    y = (EASE2_R_MAJOR * math.sin(lat_r)) / EASE2_COS_LAT0
    col = int(round(EASE2_S0 + x / EASE2_MAP_SCALE))
    row = int(round(EASE2_R0 - y / EASE2_MAP_SCALE))
    return max(0, min(EASE2_ROWS - 1, row)), max(0, min(EASE2_COLS - 1, col))

def ease2_to_latlon(row: int, col: int) -> tuple[float, float]:
    x   = (col - EASE2_S0) * EASE2_MAP_SCALE
    y   = (EASE2_R0 - row) * EASE2_MAP_SCALE
    lon = math.degrees(x / (EASE2_R_MAJOR * EASE2_COS_LAT0))
    lat = math.degrees(math.asin(y * EASE2_COS_LAT0 / EASE2_R_MAJOR))
    return round(lat, 4), round(lon, 4)


# =========================================================
# REQUEST MODELS
# =========================================================

class IrelandExtractRequest(BaseModel):
    date: str = "2026-05-17"
    download_url: str
    nasa_token: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "date": "2026-05-17",
                "download_url": "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/SMAP/SPL4SMGP/008/2026/05/16/SMAP_L4_SM_gph_20260516T223000_Vv8011_001.h5",
                "nasa_token": "your-earthdata-token"
            }]
        }
    }

class PointExtractRequest(BaseModel):
    lat: float = 52.1
    lon: float = -9.7
    date: str = "2026-05-17"
    download_url: str
    nasa_token: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "lat": 52.1,
                "lon": -9.7,
                "date": "2026-05-17",
                "download_url": "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/SMAP/SPL4SMGP/008/2026/05/16/SMAP_L4_SM_gph_20260516T223000_Vv8011_001.h5",
                "nasa_token": "your-earthdata-token"
            }]
        }
    }


# =========================================================
# HELPERS
# =========================================================

def clean_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr[arr < -9990] = np.nan
    arr[(arr < 0) | (arr > 1)] = np.nan
    return arr

def clean_scalar(v) -> Optional[float]:
    f = float(v)
    if np.isnan(f) or f < -9990 or f < 0 or f > 1:
        return None
    return round(f, 4)

def stats(arr: np.ndarray) -> dict:
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None, "unit": "m³/m³"}
    return {
        "mean":   round(float(np.mean(valid)),   4),
        "median": round(float(np.median(valid)), 4),
        "min":    round(float(np.min(valid)),    4),
        "max":    round(float(np.max(valid)),    4),
        "unit":   "m³/m³",
    }

def token_from(request_token: Optional[str]) -> str:
    t = request_token or os.getenv("NASA_EARTHDATA_TOKEN")
    if not t:
        raise HTTPException(status_code=400, detail="NASA token required")
    return t


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
async def home():
    return {"message": "SMAP Scientific Extractor Running"}

@app.get("/health")
async def health():
    return {"healthy": True, "utc": datetime.now(timezone.utc).isoformat()}

@app.post("/debug-keys")
async def debug_keys(request: IrelandExtractRequest):
    """List all dataset paths in the HDF5 — use once to verify file structure."""
    token = token_from(request.nasa_token)
    try:
        path = ensure_downloaded(request.download_url, token)
        keys: list[str] = []
        with h5py.File(path, "r") as f:
            f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
        return {"success": True, "keys": keys}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/extract-ireland")
async def extract_ireland(request: IrelandExtractRequest):
    token = token_from(request.nasa_token)
    try:
        path = ensure_downloaded(request.download_url, token)
        with h5py.File(path, "r") as f:
            g = f["Geophysical_Data"]
            surface  = clean_array(g["sm_surface"] [140:190, 1750:1900])
            rootzone = clean_array(g["sm_rootzone"][140:190, 1750:1900])
            profile  = clean_array(g["sm_profile"] [140:190, 1750:1900])
        return {
            "success": True,
            "date": request.date,
            "region": "Ireland",
            "soil_moisture": {
                "sm_surface":  stats(surface),
                "sm_rootzone": stats(rootzone),
                "sm_profile":  stats(profile),
            },
            "source": "SMAP L4 Regional Extraction (h5py)",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/extract-point")
async def extract_point(request: PointExtractRequest):
    token = token_from(request.nasa_token)
    try:
        path = ensure_downloaded(request.download_url, token)

        # EASE-2 math replaces reading latitude/longitude datasets from HDF5
        row, col = latlon_to_ease2(request.lat, request.lon)
        pixel_lat, pixel_lon = ease2_to_latlon(row, col)

        with h5py.File(path, "r") as f:
            g = f["Geophysical_Data"]
            surface  = clean_scalar(g["sm_surface"] [row, col])
            rootzone = clean_scalar(g["sm_rootzone"][row, col])
            profile  = clean_scalar(g["sm_profile"] [row, col])

        return {
            "success": True,
            "date": request.date,
            "location": {"lat": request.lat, "lon": request.lon},
            "nearest_pixel": {
                "row": row, "col": col,
                "pixel_lat": pixel_lat,
                "pixel_lon": pixel_lon,
            },
            "soil_moisture": {
                "sm_surface":  {"value": surface,  "unit": "m³/m³"},
                "sm_rootzone": {"value": rootzone, "unit": "m³/m³"},
                "sm_profile":  {"value": profile,  "unit": "m³/m³"},
            },
            "source": "SMAP L4 Point Extraction (h5py + EASE-2)",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

