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

app = FastAPI(title="SMAP Scientific Extractor")

# =========================================================
# SIMPLE DAY-LEVEL FILE CACHE
# avoids re-downloading the 150MB file on every request
# =========================================================

CACHE_DIR = "/tmp/smap_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_hdf5_path(download_url: str) -> str:
    """Return a stable cache path for a given granule URL."""
    url_hash = hashlib.md5(download_url.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"smap_{url_hash}.h5")

def ensure_downloaded(download_url: str, token: str) -> str:
    """Download HDF5 only if not already cached. Returns local path."""
    path = cached_hdf5_path(download_url)
    if os.path.exists(path):
        print(f"Cache hit: {path}")
        return path

    # Use a temp file then rename — safe if multiple workers race
    tmp_path = path + f".{uuid.uuid4().hex}.tmp"
    print(f"Downloading HDF5 → {tmp_path}")
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
        os.rename(tmp_path, path)   # atomic on Linux
        print(f"Cached at: {path}")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


# =========================================================
# REQUEST MODELS
# =========================================================

class IrelandExtractRequest(BaseModel):
    date: str = "2026-05-18"
    download_url: str
    nasa_token: Optional[str] = None

class PointExtractRequest(BaseModel):
    lat: float
    lon: float
    date: str = "2026-05-18"
    download_url: str
    nasa_token: Optional[str] = None


# =========================================================
# HELPERS
# =========================================================

FILL_VALUE = -9999.0

def clean_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr[arr <= FILL_VALUE * 0.99] = np.nan   # fill values
    arr[(arr < 0) | (arr > 1)] = np.nan      # out-of-range
    return arr

def clean_scalar(v) -> Optional[float]:
    if v is None:
        return None
    f = float(v)
    if np.isnan(f) or f < 0 or f > 1:
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


@app.post("/extract-ireland")
async def extract_ireland(request: IrelandExtractRequest):
    token = token_from(request.nasa_token)

    try:
        path = ensure_downloaded(request.download_url, token)

        with h5py.File(path, "r") as f:
            g = f["Geophysical_Data"]

            # Ireland slice: rows 140-190, cols 1750-1900
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

        with h5py.File(path, "r") as f:
            g = f["Geophysical_Data"]

            # Read lat/lon grids to find the nearest pixel
            lat_grid = g["latitude"] [:]
            lon_grid = g["longitude"][:]

            dist = np.sqrt(
                (lat_grid - request.lat) ** 2 +
                (lon_grid - request.lon) ** 2
            )
            flat_idx     = int(np.nanargmin(dist))
            row, col     = np.unravel_index(flat_idx, dist.shape)
            nearest_dist = float(dist[row, col])

            surface  = clean_scalar(g["sm_surface"] [row, col])
            rootzone = clean_scalar(g["sm_rootzone"][row, col])
            profile  = clean_scalar(g["sm_profile"] [row, col])
            pix_lat  = float(lat_grid[row, col])
            pix_lon  = float(lon_grid[row, col])

        return {
            "success": True,
            "date": request.date,
            "location":      {"lat": request.lat, "lon": request.lon},
            "nearest_pixel": {
                "row": int(row), "col": int(col),
                "distance_degrees": round(nearest_dist, 5),
                "pixel_lat": round(pix_lat, 4),
                "pixel_lon": round(pix_lon, 4),
            },
            "soil_moisture": {
                "sm_surface":  {"value": surface,  "unit": "m³/m³"},
                "sm_rootzone": {"value": rootzone, "unit": "m³/m³"},
                "sm_profile":  {"value": profile,  "unit": "m³/m³"},
            },
            "source": "SMAP L4 Point Extraction (h5py)",
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
