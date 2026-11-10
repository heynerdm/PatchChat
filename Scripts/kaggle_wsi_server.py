#!/usr/bin/env python3
"""
==============================================================================
KAGGLE NOTEBOOK - SERVIDOR DE PROCESAMIENTO WSI (OPENSILDE + OTSU + FASTAPI)
==============================================================================
Lámina oficial seleccionada:
/kaggle/input/datasets/aryaze/tcga-brca-diagnostic-slide-17/016fc18e-1a88-478f-82a4-38f4f10a8f42.svs
==============================================================================
"""

# ===================== CELDA 1: Instalación de dependencias =====================
# !apt-get update && apt-get install -y libopenslide0 -q
# !pip install openslide-python opencv-python-headless fastapi uvicorn pyngrok nest-asyncio -q

# ===================== CELDA 2: Motor WSI con OpenSlide y Otsu =====================
import os
import cv2
import glob
import time
import base64
import numpy as np
import openslide
from io import BytesIO
from PIL import Image
import nest_asyncio
import uvicorn
import threading
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok

nest_asyncio.apply()

# Ruta exacta de la lámina configurada por el usuario
DEFAULT_WSI_PATH = "/kaggle/input/datasets/aryaze/tcga-brca-diagnostic-slide-17/016fc18e-1a88-478f-82a4-38f4f10a8f42.svs"

# Búsqueda automática en caso de ruta variable
if not os.path.exists(DEFAULT_WSI_PATH):
    found = glob.glob("/kaggle/input/**/*.svs", recursive=True)
    if found:
        DEFAULT_WSI_PATH = found[0]
        print(f"📁 Usando WSI detectada: {DEFAULT_WSI_PATH}")
    else:
        print(f"⚠️ Advertencia: No se encontró {DEFAULT_WSI_PATH}")

class WSIProcessor:
    def __init__(self, slide_path: str):
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.dimensions
        self.levels = self.slide.level_count
        self.level_dims = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        # Caché para reutilizar coords de tejido sin re-segmentar
        self._tissue_coords_cache = None
        self._mask_cache = None

    def get_thumbnail_with_contours(self, max_size=(1024, 768)):
        """Genera thumbnail, segmenta tejido (Otsu) y dibuja contornos."""
        thumb = self.slide.get_thumbnail(max_size).convert("RGB")
        thumb_np = np.array(thumb)

        # Segmentación Otsu sobre el canal de saturación HSV
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Filtrado de ruido morfológico
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self._mask_cache = mask_clean  # guardar para /wsi/mask

        tissue_pct = round((np.count_nonzero(mask_clean) / mask_clean.size) * 100, 2)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Contornos cian sobre el thumbnail
        overlay = thumb_np.copy()
        cv2.drawContours(overlay, contours, -1, (0, 242, 254), 2)
        result_img = Image.fromarray(overlay)

        return result_img, mask_clean, tissue_pct, len(contours)

    def extract_patch(self, x: int, y: int, size: int = 224, level: int = 0) -> Image.Image:
        """Extrae un parche a resolución nativa en coordenadas (x, y)."""
        return self.slide.read_region((x, y), level, (size, size)).convert("RGB")

    def get_tissue_patch_coords(self, patch_size: int = 224, level: int = 1,
                                max_patches: int = 500):
        """
        Obtiene coordenadas (en nivel 0) de parches que caen sobre tejido.
        Muestrea a 'level' para velocidad; proyecta de vuelta a nivel 0.
        Resultados cacheados para reutilización en /wsi/montage.
        """
        if self._tissue_coords_cache is not None:
            return self._tissue_coords_cache

        ds = int(self.level_downsamples[level])  # factor de downsampling
        lw, lh = self.level_dims[level]
        step = patch_size  # paso en coordenadas del nivel escogido

        # Thumbnail rápido a nivel 'level' para segmentación de tejido
        thumb = self.slide.read_region((0, 0), level, (lw, lh)).convert("RGB")
        thumb_np = np.array(thumb)
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        _, mask = cv2.threshold(hsv[:, :, 1], 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        coords = []
        for gy in range(0, lh - step, step):
            for gx in range(0, lw - step, step):
                tile_mask = mask[gy:gy + step, gx:gx + step]
                tissue_ratio = np.count_nonzero(tile_mask) / tile_mask.size
                if tissue_ratio > 0.5:
                    # Proyectar a coordenadas del nivel 0
                    coords.append((gx * ds, gy * ds, patch_size, level))
        # Muestreo uniforme si hay demasiados
        if len(coords) > max_patches:
            step_sample = len(coords) // max_patches
            coords = coords[::step_sample][:max_patches]

        self._tissue_coords_cache = coords
        return coords

    def generate_segmentation_mask_png(self) -> bytes:
        """
        Retorna una máscara de segmentación binaria como PNG en bytes.
        Blanco = tejido, negro = vidrio/fondo.
        """
        # Forzar cálculo de máscara si no está cacheada
        if self._mask_cache is None:
            self.get_thumbnail_with_contours()
        mask_bin = (self._mask_cache > 0).astype(np.uint8) * 255
        pil_mask = Image.fromarray(mask_bin, mode="L")
        buf = BytesIO()
        pil_mask.save(buf, format="PNG")
        return buf.getvalue()

    def generate_tile_montage(self, patch_size: int = 224, level: int = 1,
                              grid_cols: int = 12, max_patches: int = 300) -> bytes:
        """
        Genera una imagen única (mosaico/tiling) con todos los tiles de tejido.
        Retorna JPEG en bytes.
        """
        coords = self.get_tissue_patch_coords(
            patch_size=patch_size, level=level, max_patches=max_patches
        )
        n = len(coords)
        if n == 0:
            raise ValueError("No se encontraron parches de tejido.")

        grid_rows = int(np.ceil(n / grid_cols))
        # Montage en fondo oscuro
        montage = Image.new("RGB",
                            (grid_cols * patch_size, grid_rows * patch_size),
                            (15, 15, 20))

        for idx, (x0, y0, ps, lv) in enumerate(coords):
            try:
                patch = self.extract_patch(x0, y0, size=ps, level=0)
                col = idx % grid_cols
                row = idx // grid_cols
                montage.paste(patch, (col * ps, row * ps))
            except Exception:
                pass  # parche fuera de límites → dejar negro

        buf = BytesIO()
        montage.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

print(f"🔬 OpenSlide listo. Cargando: {os.path.basename(DEFAULT_WSI_PATH)}")
wsi_engine = WSIProcessor(DEFAULT_WSI_PATH)
print(f"✅ Dimensiones WSI: {wsi_engine.width} x {wsi_engine.height} px | Niveles: {wsi_engine.levels}")

# ===================== CELDA 3: API FastAPI y Túnel Ngrok =====================
# Pegar bloques CELDA 2b (PLIP indexing) ANTES de levantar FastAPI en el notebook
# Luego exponer: TISSUE_COORDS, PATCH_FEATURES, PLIP_MODEL, PLIP_PROC, PLIP_DEVICE

app = FastAPI(title="PredictPath WSI API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# /health  — telemetría del servidor
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_used = round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0
    return {
        "status": "online",
        "wsi_file": os.path.basename(wsi_engine.slide_path),
        "dimensions": f"{wsi_engine.width} x {wsi_engine.height}",
        "levels": wsi_engine.levels,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

# ---------------------------------------------------------------------------
# /wsi/overview  — thumbnail WSI con segmentación Otsu
# ---------------------------------------------------------------------------
@app.get("/wsi/overview")
def wsi_overview():
    """Retorna thumbnail con contornos de tejido (cian) y métricas en vivo."""
    thumb_img, mask, tissue_pct, num_contours = wsi_engine.get_thumbnail_with_contours()
    buf = BytesIO()
    thumb_img.save(buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "filename": os.path.basename(wsi_engine.slide_path),
        "width": wsi_engine.width,
        "height": wsi_engine.height,
        "tissue_coverage_pct": tissue_pct,
        "contours_count": num_contours,
        "thumbnail_b64": b64_img
    }

# ---------------------------------------------------------------------------
# /wsi/patch  — parche individual en alta resolución
# ---------------------------------------------------------------------------
@app.get("/wsi/patch")
def get_patch(x: int = Query(..., description="Coordenada X"),
              y: int = Query(..., description="Coordenada Y"),
              size: int = Query(224, description="Tamaño del parche"),
              level: int = Query(0, description="Nivel de magnificación")):
    """Devuelve un parche celular en alta resolución en las coordenadas especificadas."""
    try:
        patch = wsi_engine.extract_patch(x, y, size=size, level=level)
        buf = BytesIO()
        patch.save(buf, format="PNG")
        b64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "status": "ok",
            "x": x, "y": y, "size": size, "level": level,
            "patch_b64": b64_patch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /wsi/mask  ★ NUEVO — máscara de segmentación binaria (blanco=tejido)
# ---------------------------------------------------------------------------
@app.get("/wsi/mask")
def get_segmentation_mask():
    """
    Retorna la máscara de segmentación binaria como PNG en Base64.
    Blanco (255) = tejido viable. Negro (0) = vidrio/fondo.
    """
    try:
        mask_png_bytes = wsi_engine.generate_segmentation_mask_png()
        b64_mask = base64.b64encode(mask_png_bytes).decode("utf-8")
        return {
            "status": "ok",
            "format": "PNG",
            "description": "Máscara binaria Otsu: blanco=tejido, negro=vidrio",
            "thumbnail_width": 1024,
            "mask_b64": b64_mask
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /wsi/montage  ★ NUEVO — mosaico de tiles de tejido (tiling completo)
# ---------------------------------------------------------------------------
@app.get("/wsi/montage")
def get_tile_montage(
    patch_size: int = Query(224, description="Tamaño de cada tile en px"),
    grid_cols: int = Query(12, description="Número de columnas en el mosaico"),
    max_patches: int = Query(300, description="Máx. tiles a incluir"),
    level: int = Query(1, description="Nivel de muestreo (1=rápido, 0=nativo)")
):
    """
    Genera una única imagen (JPEG) con todos los tiles de tejido detectados.
    Implements el Tiling requerido por el profesor.
    """
    try:
        montage_bytes = wsi_engine.generate_tile_montage(
            patch_size=patch_size,
            level=level,
            grid_cols=grid_cols,
            max_patches=max_patches
        )
        b64_montage = base64.b64encode(montage_bytes).decode("utf-8")
        n_tiles = len(wsi_engine._tissue_coords_cache or [])
        return {
            "status": "ok",
            "format": "JPEG",
            "total_tiles": n_tiles,
            "grid_cols": grid_cols,
            "grid_rows": int(np.ceil(n_tiles / grid_cols)),
            "patch_size": patch_size,
            "montage_b64": b64_montage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /wsi/navigate  — Navigator PLIP: consulta textual → Top K% RoIs
# (requiere que TISSUE_COORDS, PATCH_FEATURES, PLIP_MODEL, PLIP_PROC, PLIP_DEVICE
#  estén definidos como globals desde la Celda 2b del notebook)
# ---------------------------------------------------------------------------
from pydantic import BaseModel as PydanticBaseModel

class NavigateRequest(PydanticBaseModel):
    query: str
    top_pct: float = 0.10

@app.post("/wsi/navigate")
def navigate_wsi(req: NavigateRequest):
    """
    Navigator PLIP: codifica la consulta textual, calcula similitud coseno
    contra todos los embeddings de parches y retorna el Top K% de RoIs.
    """
    import torch
    try:
        # Acceso a globals del notebook
        tissue_coords = globals().get("TISSUE_COORDS") or wsi_engine.get_tissue_patch_coords()
        patch_features = globals()["PATCH_FEATURES"]   # (N, 512) tensor normalizado
        plip_model     = globals()["PLIP_MODEL"]
        plip_proc      = globals()["PLIP_PROC"]
        device         = globals().get("PLIP_DEVICE", "cuda")
    except KeyError as ke:
        raise HTTPException(status_code=503,
            detail=f"PLIP no inicializado. Ejecuta la Celda 2b primero. ({ke})")

    with torch.no_grad():
        text_inputs = plip_proc(text=[req.query], return_tensors="pt", padding=True).to(device)
        text_out    = plip_model.get_text_features(**text_inputs)
        # Fix BaseModelOutputWithPooling (transformers >= 4.38)
        text_feat   = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        text_feat   = text_feat / text_feat.norm(dim=-1, keepdim=True)

    # Similitud coseno
    sims  = (patch_features.to(device) @ text_feat.T).squeeze(-1).cpu()
    top_k = max(1, int(len(sims) * req.top_pct))
    top_idx = sims.topk(top_k).indices.tolist()

    # Thumbnail para escalar coordenadas → thumb_x, thumb_y
    thumb_w, thumb_h = 1024, 768
    sx = thumb_w / wsi_engine.width
    sy = thumb_h / wsi_engine.height

    rois = []
    for i in top_idx:
        x0, y0, ps, _ = tissue_coords[i]
        rois.append({
            "x": x0, "y": y0,
            "thumb_x": int(x0 * sx), "thumb_y": int(y0 * sy),
            "score": float(sims[i])
        })
    rois.sort(key=lambda r: r["score"], reverse=True)

    # Parche principal en B64 (el de mayor score)
    best = rois[0]
    primary_patch = wsi_engine.extract_patch(best["x"], best["y"], size=224, level=0)
    buf = BytesIO()
    primary_patch.save(buf, format="JPEG", quality=85)
    primary_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "status": "success",
        "query": req.query,
        "top_pct": req.top_pct,
        "total_evaluated": len(tissue_coords),
        "top_k_count": top_k,
        "rois": rois,
        "primary_patch_b64": primary_b64
    }

# ===================== NGROK & LANZAMIENTO =====================
try:
    from kaggle_secrets import UserSecretsClient
    NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_TOKEN")
except Exception:
    NGROK_TOKEN = "PEGA_TU_TOKEN_AQUI_SI_NO_USAS_SECRETS"

ngrok.set_auth_token(NGROK_TOKEN)
try:
    ngrok.kill()
except Exception:
    pass

public_url = ngrok.connect(8000)
print("=" * 65)
print(f"🚀 SERVIDOR WSI ONLINE:")
print(f"   👉  {public_url}  👈")
print("=" * 65)
print("Endpoints disponibles:")
print(f"  • GET  {public_url}/health          → Telemetría GPU y estado")
print(f"  • GET  {public_url}/wsi/overview    → Thumbnail con tejido segmentado")
print(f"  • GET  {public_url}/wsi/patch       → Parche real en alta resolución")
print(f"  • GET  {public_url}/wsi/mask        ★ Máscara de segmentación binaria PNG")
print(f"  • GET  {public_url}/wsi/montage     ★ Mosaico de tiles de tejido JPEG")
print(f"  • POST {public_url}/wsi/navigate    ★ Navigator PLIP → Top K% RoIs")
print("=" * 65)

def run():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run, daemon=True)
server_thread.start()
print("✅ Servidor Uvicorn corriendo en puerto 8000.")
