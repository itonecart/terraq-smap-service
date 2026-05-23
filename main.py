from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import h5py
import numpy as np
import requests
import boto3
import os
import uuid
import hashlib
import io
from datetime import datetime, timezone
import math

app = FastAPI(title="SMAP Scientific Extractor")

# =========================================================
# CACHE
# =========================================================

CACHE_DIR = "/tmp/smap_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_hdf5_path(key: str) -> str:
    url_hash = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"smap_{url_hash}.h5")

def ensure_downloaded_s3(
    s3_bucket: str,
    s3_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str,
    s3_region: str = "us-west-2",
) -> str:
    """Download HDF5 from S3 using temporary NASA Earthdata Cloud credentials."""
    path = cached_hdf5_path(s3_key)
    if os.path.exists(path):
        print(f"[S3] Cache hit: {path}")
        return path

    print(f"[S3] Downloading s3://{s3_bucket}/{s3_key}")
    tmp_path = path + f".{uuid.uuid4().hex}.tmp"
    try:
        s3 = boto3.client(
            "s3",
            region_name=s3_region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
        s3.download_file(s3_bucket, s3_key, tmp_path)
        os.rename(tmp_path, path)
        print(f"[S3] Cached: {path}")
        return path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"S3 download failed: {e}") from e

def ensure_downloaded_https(download_url: str, token: str) -> str:
    """Fallback: download via HTTPS with Bearer token (handles EDL redirect)."""
    path = cached_hdf5_path(download_url)
    if os.path.exists(path):
        print(f"[HTTPS] Cache hit: {path}")
        return path

    print(f"[HTTPS] Downloading {download_url}")
    tmp_path = path + f".{uuid.uuid4().hex}.tmp"
    try:
        session = requests.Session()
        # Follow redirects manually — strip auth header for S3 presigned URLs
        response = session.get(
            download_url,
            headers={"Authorization": f"Bearer {token}"},
            allow_redirects=False,
            timeout=30,
        )
        while response.status_code in (301, 302, 303, 307, 308):
            redirect_url = response.headers.get("Location", "")
            is_s3 = "s3.amazonaws.com" in redirect_url or "s3-us-west" in redirect_url
            if is_s3:
                response = requests.get(redirect_url, allow_redirects=True, timeout=300, stream=True)
            else:
                response = session.get(
                    redirect_url,
                    headers={"Authorization": f"Bearer {token}"},
                    allow_redirects=False,
                    timeout=30,
                )
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.rename(tmp_path, path)
        print(f"[HTTPS] Cached: {path}")
        return path
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"HTTPS download failed: {e}") from e


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
    # S3 credentials (preferred over HTTPS)
    s3_bucket:             Optional[str] = None
    s3_key:                Optional[str] = None
    s3_region:             Optional[str] = "us-west-2"
    aws_access_key_id:     Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token:     Optional[str] = None

class PointExtractRequest(BaseModel):
    lat: float = 52.1
    lon: float = -9.7
    date: str = "2026-05-17"
    download_url: str
    nasa_token: Optional[str] = None
    # S3 credentials (preferred over HTTPS)
    s3_bucket:             Optional[str] = None
    s3_key:                Optional[str] = None
    s3_region:             Optional[str] = "us-west-2"
    aws_access_key_id:     Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token:     Optional[str] = None


# =========================================================
# SHARED DOWNLOAD LOGIC
# =========================================================

def get_hdf5_path(request) -> str:
    """S3 credentials → S3 download. Falls back to HTTPS with redirect fix."""
    if (request.s3_bucket and request.s3_key and
            request.aws_access_key_id and request.aws_secret_access_key and request.aws_session_token):
        print(f"[SMAP] Using S3 direct download")
        return ensure_downloaded_s3(
            s3_bucket=request.s3_bucket,
            s3_key=request.s3_key,
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            aws_session_token=request.aws_session_token,
            s3_region=request.s3_region or "us-west-2",
        )
    else:
        print(f"[SMAP] Using HTTPS download (S3 creds not provided)")
        token = request.nasa_token or os.getenv("NASA_EARTHDATA_TOKEN")
        if not token:
            raise HTTPException(status_code=400, detail="NASA token or S3 credentials required")
        return ensure_downloaded_https(request.download_url, token)


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
    try:
        path = get_hdf5_path(request)
        keys: list[str] = []
        with h5py.File(path, "r") as f:
            f.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Dataset) else None)
        return {"success": True, "keys": keys}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/extract-ireland")
async def extract_ireland(request: IrelandExtractRequest):
    try:
        path = get_hdf5_path(request)
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
    try:
        path = get_hdf5_path(request)
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
