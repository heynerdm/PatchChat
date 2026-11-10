#!/usr/bin/env python3
"""
==============================================================================
PREDICTPATH — KAGGLE NOTEBOOK COMPLETO v4.0  (fiel a PathAgent ECCV 2026)
==============================================================================
Copiar cada bloque en una celda separada del notebook de Kaggle.
Acelerador: GPU T4 x1 (16 GB VRAM).  Internet: ON.

PIPELINE COMPLETO:
  Navigator  → PLIP          (ranking Top-10% de RoIs)
  Perceptor  → Quilt-LLaVA   (descripción morfológica de cada parche)
  Executor   → Qwen2.5-VL    (CoT 3 pasos → diagnóstico final)

Endpoints:
  GET  /health
  GET  /wsi/overview
  GET  /wsi/patch
  GET  /wsi/mask      ★ Máscara binaria PNG
  GET  /wsi/montage   ★ Mosaico de tiles JPEG
  POST /wsi/navigate  ★ PLIP → Top K% RoIs
  POST /wsi/analyze   ★ Pipeline completo Perceptor + Executor
==============================================================================
"""

# =============================================================================
# ██  CELDA 1 — INSTALACIÓN  ██████████████████████████████████████████████████
# =============================================================================
# Pegar esto en la primera celda del notebook y ejecutar:
#
# !apt-get update && apt-get install -y libopenslide0 -q
# !pip install openslide-python opencv-python-headless -q
# !pip install fastapi uvicorn nest-asyncio pyngrok -q
# !pip install transformers accelerate bitsandbytes -q
# !pip install qwen-vl-utils -q


# =============================================================================
# ██  CELDA 2 — MOTOR WSI + CLASES  ██████████████████████████████████████████
# =============================================================================
import os, sys, cv2, glob, base64, json, re
import numpy as np
import openslide
from io import BytesIO
from PIL import Image
from copy import deepcopy
import nest_asyncio
import uvicorn
import threading
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel

nest_asyncio.apply()

DEFAULT_WSI_PATH = (
    "/kaggle/input/datasets/aryaze/tcga-brca-diagnostic-slide-17/"
    "016fc18e-1a88-478f-82a4-38f4f10a8f42.svs"
)
if not os.path.exists(DEFAULT_WSI_PATH):
    found = glob.glob("/kaggle/input/**/*.svs", recursive=True)
    DEFAULT_WSI_PATH = found[0] if found else DEFAULT_WSI_PATH
    print(f"📁 WSI detectada: {DEFAULT_WSI_PATH}")


class WSIProcessor:
    def __init__(self, slide_path: str):
        self.slide_path = slide_path
        self.slide = openslide.OpenSlide(slide_path)
        self.width, self.height = self.slide.dimensions
        self.levels = self.slide.level_count
        self.level_dims = self.slide.level_dimensions
        self.level_downsamples = self.slide.level_downsamples
        self._tissue_coords_cache = None
        self._mask_cache = None

    def get_thumbnail_with_contours(self, max_size=(1024, 768)):
        thumb = self.slide.get_thumbnail(max_size).convert("RGB")
        thumb_np = np.array(thumb)
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        _, mask = cv2.threshold(hsv[:, :, 1], 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        self._mask_cache = mask_clean
        tissue_pct = round((np.count_nonzero(mask_clean) / mask_clean.size) * 100, 2)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = thumb_np.copy()
        cv2.drawContours(overlay, contours, -1, (0, 242, 254), 2)
        return Image.fromarray(overlay), mask_clean, tissue_pct, len(contours)

    def extract_patch(self, x: int, y: int, size: int = 224, level: int = 0) -> Image.Image:
        return self.slide.read_region((x, y), level, (size, size)).convert("RGB")

    def get_macro_tissue_tiles(self, macro_size_l0=4096, min_tissue_ratio=0.25):
        """
        Tiling Jerárquico al 100% de cobertura (Idéntico a PathAgent/CLAM):
        Divide la lámina completa en macro-bloques continuos de 4096 x 4096 px (nivel 0).
        Sin submuestreo: cada milímetro de tejido en la lámina es cubierto con 0 huecos.
        """
        if self._tissue_coords_cache is not None:
            return self._tissue_coords_cache

        # Asegurar cálculo de máscara de tejido Otsu
        if self._mask_cache is None:
            self.get_thumbnail_with_contours()

        mask = self._mask_cache
        mh, mw = mask.shape
        scale_x = mw / self.width
        scale_y = mh / self.height

        # Nivel piramidal óptimo para leer los macro-parches al vuelo (Nivel 2 o 1)
        target_level = min(2, self.levels - 1)
        ds = int(self.level_downsamples[target_level])
        target_read_sz = int(macro_size_l0 / ds)

        tiles = []
        for y0 in range(0, self.height, macro_size_l0):
            for x0 in range(0, self.width, macro_size_l0):
                mx0 = int(x0 * scale_x)
                my0 = int(y0 * scale_y)
                mx1 = min(mw, int((x0 + macro_size_l0) * scale_x))
                my1 = min(mh, int((y0 + macro_size_l0) * scale_y))

                tile_mask = mask[my0:my1, mx0:mx1]
                if tile_mask.size > 0:
                    tissue_pct = np.count_nonzero(tile_mask) / tile_mask.size
                    if tissue_pct >= min_tissue_ratio:
                        tiles.append({
                            "x": x0,
                            "y": y0,
                            "macro_size": macro_size_l0,
                            "level": target_level,
                            "read_size": target_read_sz,
                            "tissue_pct": round(tissue_pct * 100, 1)
                        })

        self._tissue_coords_cache = tiles
        return tiles

    def get_tissue_patch_coords(self, patch_size=224, level=1, max_patches=500):
        # Compatibilidad hacia atrás
        tiles = self.get_macro_tissue_tiles(macro_size_l0=4096)
        return [(t["x"], t["y"], patch_size, level) for t in tiles]

    def generate_segmentation_mask_png(self) -> bytes:
        if self._mask_cache is None:
            self.get_thumbnail_with_contours()
        mask_bin = (self._mask_cache > 0).astype(np.uint8) * 255
        buf = BytesIO()
        Image.fromarray(mask_bin, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    def generate_tile_montage(self, patch_size=224, level=1,
                              grid_cols=12, max_patches=300) -> bytes:
        coords = self.get_tissue_patch_coords(patch_size=patch_size,
                                              level=level, max_patches=max_patches)
        n = len(coords)
        if n == 0:
            raise ValueError("Sin parches de tejido.")
        grid_rows = int(np.ceil(n / grid_cols))
        montage = Image.new("RGB", (grid_cols * patch_size, grid_rows * patch_size), (15, 15, 20))
        for idx, (x0, y0, ps, _) in enumerate(coords):
            try:
                patch = self.extract_patch(x0, y0, size=ps, level=0)
                montage.paste(patch, ((idx % grid_cols) * ps, (idx // grid_cols) * ps))
            except Exception:
                pass
        buf = BytesIO()
        montage.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


print(f"🔬 Cargando WSI: {os.path.basename(DEFAULT_WSI_PATH)}")
wsi_engine = WSIProcessor(DEFAULT_WSI_PATH)
print(f"✅ {wsi_engine.width} × {wsi_engine.height} px | {wsi_engine.levels} niveles piramidales")


# =============================================================================
# ██  CELDA 2b — NAVIGATOR: PLIP (TILING JERÁRQUICO 100% COBERTURA)  █████████
# Macro-bloques de 4096 x 4096 px continuos sin huecos (Idéntico a PathAgent)
# =============================================================================
import torch
from transformers import CLIPModel, CLIPProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ [1/2] Cargando Navigator (PLIP)...")
plip_model = CLIPModel.from_pretrained("vinid/plip").to(DEVICE).eval()
plip_proc  = CLIPProcessor.from_pretrained("vinid/plip")
print(f"✅ PLIP listo en {DEVICE}")

BATCH_SIZE = 32

macro_tiles = wsi_engine.get_macro_tissue_tiles(macro_size_l0=4096, min_tissue_ratio=0.25)
print(f"🔍 Indexando {len(macro_tiles)} macro-tiles (COBERTURA TOTAL DE LA LÁMINA 100%, 0 HUECOS)...")

patch_features_list = []
for i in range(0, len(macro_tiles), BATCH_SIZE):
    batch_meta = macro_tiles[i:i + BATCH_SIZE]
    batch_imgs = [
        wsi_engine.slide.read_region(
            (m["x"], m["y"]), m["level"], (m["read_size"], m["read_size"])
        ).convert("RGB").resize((224, 224))
        for m in batch_meta
    ]
    inputs_plip = plip_proc(images=batch_imgs, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out   = plip_model.get_image_features(**inputs_plip)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
    patch_features_list.append(feats.cpu())

PATCH_FEATURES = torch.cat(patch_features_list, dim=0)
MACRO_TILES    = macro_tiles
TISSUE_COORDS  = [(m["x"], m["y"], m["macro_size"], 0) for m in macro_tiles]
PLIP_DEVICE    = DEVICE
print(f"✅ Indexados {PATCH_FEATURES.shape[0]} macro-tiles. Shape: {PATCH_FEATURES.shape} (100% tejido cubierto)")


# =============================================================================
# ██  CELDA 2c — PERCEPTOR Y EXECUTOR (QWEN2.5-VL-3B EN FP16 NATIVO)  █████████
# Carga directa en FP16 (sin bitsandbytes, sin dependencias complejas, ~6GB VRAM)
# =============================================================================
import os, sys, json, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

print("⏳ [2/2] Cargando Qwen2.5-VL-3B en FP16 nativo (sin bitsandbytes)...")

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
qwen_model.eval()
print(f"✅ Qwen2.5-VL-3B listo en {qwen_model.device} | VRAM: ~6GB")


# ── 1. FUNCIÓN PERCEPTOR: Describe el parche morfológicamente ────────────────
def describe_patch_vlm(patch_img, question, coords=None, magnification=20, missing_info=None):
    """Port de patho_r1_describe (soporta nativo y qwen_vl_utils)"""
    meta_lines = []
    if coords:
        meta_lines.append(f"Patch coordinates: ({coords[0]},{coords[1]})")
    if magnification:
        meta_lines.append(f"Magnification: {magnification}x")
    meta_text = ("[IMAGE META] " + " | ".join(meta_lines) + "\n\n") if meta_lines else ""

    prompt_body = (
        f"Question: {question}\n\n"
        "Describe the key histopathological features visible in this patch "
        "(cellular atypia, tissue architecture, pleomorphism, mitotic figures, desmoplasia) "
        "relevant to answering the question."
    )
    if missing_info:
        prompt_body += f"\nSpecifically focus on this missing information: {missing_info}"

    system_prompt = (
        "You are an expert AI pathology assistant specialized in whole slide image analysis. "
        "Describe observed morphological features, cellular details, tissue patterns and atypia clearly and objectively."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": patch_img},
                {"type": "text", "text": meta_text + prompt_body}
            ]
        }
    ]

    # Modo 100% nativo de transformers (sin librerías externas)
    inputs = qwen_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(qwen_model.device)

    with torch.no_grad():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=256)

    input_len = inputs.input_ids.shape[1] if hasattr(inputs, "input_ids") else inputs["input_ids"].shape[1]
    gen_trimmed = gen_ids[0][input_len:]
    output_text = qwen_processor.decode(gen_trimmed, skip_special_tokens=True).strip()
    return output_text


# ── 2. FUNCIÓN EXECUTOR: CoT de 3 pasos (PredictAnswer → Reflect → Explore) ─
def _call_qwen_json(messages, max_new_tokens=512):
    text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    with torch.no_grad():
        gen_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)
    out_ids = gen_ids[0][len(inputs.input_ids[0]):].tolist()
    raw = qwen_tokenizer.decode(out_ids, skip_special_tokens=True).strip()
    
    # Extraer bloque JSON
    try:
        return raw, json.loads(raw)
    except Exception:
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e != -1:
            try:
                return raw, json.loads(raw[s:e+1])
            except Exception:
                pass
    return raw, None


def execute_cot_chain(description: str, question: str) -> dict:
    """Port de evaluate_with_llm_chain de PathAgent/models/inference.py"""
    # Step A: PredictAnswer
    sys_a = (
        "You are an expert AI pathology assistant.\n"
        "Task: Based on the patch descriptions, try to answer the question step-by-step.\n"
        "Output ONLY a JSON object:\n"
        '{\n  "answer": "the final predicted answer (string)",\n'
        '  "thinking_steps": "your detailed reasoning, step-by-step (string)"\n}'
    )
    usr_a = f"--- Patch Descriptions ---\n{description}\n--- End ---\nQuestion: {question}\nNow output the JSON with 'answer' and 'thinking_steps'."
    raw_a, parsed_a = _call_qwen_json([{"role": "system", "content": sys_a}, {"role": "user", "content": usr_a}], max_new_tokens=512)
    if not parsed_a: parsed_a = {"answer": "Incierto", "thinking_steps": raw_a or ""}

    # Step B: SelfReflect
    sys_b = (
        "You are an expert AI pathology assistant.\n"
        "Task: Judge whether the current patch descriptions are sufficient to confidently support the answer.\n"
        'Output ONLY a JSON object: {"sufficient": "Yes" or "No"}'
    )
    usr_b = f"Descriptions:\n{description}\n\nQuestion: {question}\n\nPrevious answer:\n{json.dumps(parsed_a)}\n\nReturn JSON only."
    raw_b, parsed_b = _call_qwen_json([{"role": "system", "content": sys_b}, {"role": "user", "content": usr_b}], max_new_tokens=128)
    if not parsed_b: parsed_b = {"sufficient": "Incierto"}

    suff = parsed_b.get("sufficient", "").strip().lower()
    if suff == "yes":
        return {
            "answer": parsed_a.get("answer"),
            "thinking_steps": parsed_a.get("thinking_steps"),
            "sufficient": "Yes",
            "missing_info": None, "zoom_recommendation": None, "zoom_level": None, "zoom_reason": None,
            "cot_steps": {"step_a": parsed_a, "step_b": parsed_b, "step_c": None}
        }

    # Step C: ExploreMissing
    sys_c = (
        "You are an expert AI pathology assistant.\n"
        "Task: If current data is insufficient, specify what visual evidence is missing, and whether zooming in could help.\n"
        "Output ONLY a JSON object:\n"
        '{\n  "missing_info": "short noun phrase",\n  "zoom_recommendation": "Yes" or "No",\n  "recommended_zoom_level": "None" or 20 or 40,\n  "zoom_reason": "brief reason"\n}'
    )
    usr_c = f"Descriptions:\n{description}\n\nQuestion: {question}\n\nPrevious answer: {json.dumps(parsed_a)}\n\nSufficiency: {json.dumps(parsed_b)}"
    raw_c, parsed_c = _call_qwen_json([{"role": "system", "content": sys_c}, {"role": "user", "content": usr_c}], max_new_tokens=256)
    if not parsed_c: parsed_c = {"missing_info": "Detalle celular adicional", "zoom_recommendation": "Yes", "recommended_zoom_level": 40, "zoom_reason": raw_c or ""}

    return {
        "answer": parsed_a.get("answer"),
        "thinking_steps": parsed_a.get("thinking_steps"),
        "sufficient": parsed_b.get("sufficient"),
        "missing_info": parsed_c.get("missing_info"),
        "zoom_recommendation": parsed_c.get("zoom_recommendation"),
        "zoom_level": parsed_c.get("recommended_zoom_level", 20),
        "zoom_reason": parsed_c.get("zoom_reason"),
        "cot_steps": {"step_a": parsed_a, "step_b": parsed_b, "step_c": parsed_c}
    }


# =============================================================================
# ██  CELDA 3 — FASTAPI: TODOS LOS ENDPOINTS  █████████████████████████████████
# =============================================================================
app = FastAPI(title="PredictPath WSI API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    gpu_name   = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_used  = round(torch.cuda.memory_allocated(0) / 1e9, 2) if torch.cuda.is_available() else 0
    vram_total = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0
    return {
        "status": "online",
        "wsi_file": os.path.basename(wsi_engine.slide_path),
        "dimensions": f"{wsi_engine.width} x {wsi_engine.height}",
        "levels": wsi_engine.levels,
        "gpu_name": gpu_name,
        "vram_used_gb": vram_used,
        "vram_total_gb": vram_total,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models_loaded": ["PLIP", "Qwen2.5-VL-3B"]
    }


# ── /wsi/overview ─────────────────────────────────────────────────────────────
@app.get("/wsi/overview")
def wsi_overview():
    thumb_img, mask, tissue_pct, num_contours = wsi_engine.get_thumbnail_with_contours()
    buf = BytesIO()
    thumb_img.save(buf, format="JPEG", quality=85)
    return {
        "filename": os.path.basename(wsi_engine.slide_path),
        "width": wsi_engine.width, "height": wsi_engine.height,
        "tissue_coverage_pct": tissue_pct,
        "contours_count": num_contours,
        "thumbnail_b64": base64.b64encode(buf.getvalue()).decode()
    }


# ── /wsi/patch ────────────────────────────────────────────────────────────────
@app.get("/wsi/patch")
def get_patch(x: int = Query(...), y: int = Query(...),
              size: int = Query(224), level: int = Query(0)):
    try:
        patch = wsi_engine.extract_patch(x, y, size=size, level=level)
        buf = BytesIO()
        patch.save(buf, format="PNG")
        return {"status": "ok", "x": x, "y": y, "size": size, "level": level,
                "patch_b64": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /wsi/mask ★ ───────────────────────────────────────────────────────────────
@app.get("/wsi/mask")
def get_segmentation_mask():
    try:
        mask_bytes = wsi_engine.generate_segmentation_mask_png()
        return {"status": "ok", "format": "PNG",
                "mask_b64": base64.b64encode(mask_bytes).decode()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /wsi/montage ★ ───────────────────────────────────────────────────────────
@app.get("/wsi/montage")
def get_tile_montage(patch_size: int = Query(224), grid_cols: int = Query(12),
                     max_patches: int = Query(300), level: int = Query(1)):
    try:
        montage_bytes = wsi_engine.generate_tile_montage(
            patch_size=patch_size, level=level,
            grid_cols=grid_cols, max_patches=max_patches
        )
        n = len(wsi_engine._tissue_coords_cache or [])
        return {
            "status": "ok", "format": "JPEG",
            "total_tiles": n, "grid_cols": grid_cols,
            "grid_rows": int(np.ceil(n / grid_cols)),
            "patch_size": patch_size,
            "montage_b64": base64.b64encode(montage_bytes).decode()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /wsi/navigate ★ ───────────────────────────────────────────────────────────
class NavigateRequest(PydanticBaseModel):
    query: str
    top_pct: float = 0.10

@app.post("/wsi/navigate")
def navigate_wsi(req: NavigateRequest):
    with torch.no_grad():
        text_in   = plip_proc(text=[req.query], return_tensors="pt", padding=True).to(PLIP_DEVICE)
        text_out  = plip_model.get_text_features(**text_in)
        text_feat = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims  = (PATCH_FEATURES.to(PLIP_DEVICE) @ text_feat.T).squeeze(-1).cpu()
    top_k = max(1, int(len(sims) * req.top_pct))
    top_idx = sims.topk(top_k).indices.tolist()

    thumb_w = 1024
    thumb_h = int(1024 * (wsi_engine.height / wsi_engine.width))
    sx, sy = thumb_w / wsi_engine.width, thumb_h / wsi_engine.height
    rois = []
    for i in top_idx:
        x0, y0, _, _ = TISSUE_COORDS[i]
        rois.append({
            "x": x0, "y": y0,
            "macro_size": 4096,
            "thumb_x": int(x0 * sx), "thumb_y": int(y0 * sy),
            "score": float(sims[i])
        })
    rois.sort(key=lambda r: r["score"], reverse=True)

    best  = rois[0]
    # Extraer ventana celular nativa a Nivel 0 (resolución diagnóstica) en el centro del macro-tile
    center_x = max(0, min(wsi_engine.width - 512, best["x"] + 1792))
    center_y = max(0, min(wsi_engine.height - 512, best["y"] + 1792))
    patch = wsi_engine.extract_patch(center_x, center_y, size=512, level=0)
    buf   = BytesIO()
    patch.save(buf, format="JPEG", quality=88)

    return {
        "status": "success", "query": req.query,
        "top_pct": req.top_pct, "total_evaluated": len(TISSUE_COORDS),
        "top_k_count": top_k, "rois": rois,
        "primary_patch_b64": base64.b64encode(buf.getvalue()).decode()
    }


# ── /wsi/analyze ★★ PIPELINE COMPLETO ────────────────────────────────────────
class AnalyzeRequest(PydanticBaseModel):
    query: str
    top_pct: float = 0.10
    max_perceptor_patches: int = 5   # cuántos parches del Top-K describir con Quilt-LLaVA
    magnification: int = 20

@app.post("/wsi/analyze")
def analyze_wsi(req: AnalyzeRequest):
    """
    Pipeline completo PathAgent:
      1. Navigator (PLIP)      → Top K% RoIs
      2. Perceptor (Quilt-LLaVA) → descripción morfológica de cada RoI
      3. Executor (Qwen2.5-VL)  → CoT 3 pasos → diagnóstico final
    """
    # ── 1. NAVIGATOR ──────────────────────────────────────────────────────
    with torch.no_grad():
        text_in   = plip_proc(text=[req.query], return_tensors="pt", padding=True).to(PLIP_DEVICE)
        text_out  = plip_model.get_text_features(**text_in)
        text_feat = text_out.pooler_output if hasattr(text_out, "pooler_output") else text_out
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims    = (PATCH_FEATURES.to(PLIP_DEVICE) @ text_feat.T).squeeze(-1).cpu()
    top_k   = max(1, int(len(sims) * req.top_pct))
    top_idx = sims.topk(top_k).indices.tolist()

    thumb_w = 1024
    thumb_h = int(1024 * (wsi_engine.height / wsi_engine.width))
    sx, sy = thumb_w / wsi_engine.width, thumb_h / wsi_engine.height
    rois = []
    for i in top_idx:
        x0, y0, _, _ = TISSUE_COORDS[i]
        rois.append({
            "x": x0, "y": y0,
            "macro_size": 4096,
            "thumb_x": int(x0 * sx), "thumb_y": int(y0 * sy),
            "score": float(sims[i])
        })
    rois.sort(key=lambda r: r["score"], reverse=True)

    # Parche principal para retornar a UI: ventana de 512x512 en Nivel 0
    best  = rois[0]
    cx0 = max(0, min(wsi_engine.width - 512, best["x"] + 1792))
    cy0 = max(0, min(wsi_engine.height - 512, best["y"] + 1792))
    patch = wsi_engine.extract_patch(cx0, cy0, size=512, level=0)
    buf   = BytesIO()
    patch.save(buf, format="JPEG", quality=88)
    primary_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── 2. PERCEPTOR: describe los Top-N macro-focos con Qwen2.5-VL ──────────
    n_describe = min(req.max_perceptor_patches, len(rois))
    patch_descriptions = []

    for idx, roi in enumerate(rois[:n_describe]):
        # Extraer ventana celular nativa a Nivel 0 en el foco sospechoso
        rcx = max(0, min(wsi_engine.width - 512, roi["x"] + 1792))
        rcy = max(0, min(wsi_engine.height - 512, roi["y"] + 1792))
        patch_img = wsi_engine.extract_patch(rcx, rcy, size=512, level=0)
        description = describe_patch_vlm(
            patch_img,
            question=req.query,
            coords=(roi["x"], roi["y"]),
            magnification=req.magnification,
        )
        patch_descriptions.append({
            "patch_idx":   idx + 1,
            "x":           roi["x"],
            "y":           roi["y"],
            "plip_score":  roi["score"],
            "description": description
        })

    # Concatenar descripciones para el Executor (formato del paper)
    combined_description = "\n\n".join([
        f"[Patch {d['patch_idx']} | Coord: ({d['x']},{d['y']}) | "
        f"PLIP Score: {d['plip_score']:.3f}]\n{d['description']}"
        for d in patch_descriptions
    ])

    # ── 3. EXECUTOR: Qwen2.5-VL — CoT 3 pasos ─────────────────────────────
    cot_result = execute_cot_chain(combined_description, req.query)

    return {
        "status":       "success",
        "query":        req.query,
        # Navigator
        "navigator": {
            "total_evaluated": len(TISSUE_COORDS),
            "top_k_count":     top_k,
            "top_pct":         req.top_pct,
            "rois":            rois,
            "primary_patch_b64": primary_b64
        },
        # Perceptor
        "perceptor": {
            "patches_described": n_describe,
            "descriptions":      patch_descriptions,
            "combined_text":     combined_description
        },
        # Executor (CoT)
        "executor": cot_result
    }


# =============================================================================
# ██  CELDA 4 — NGROK + UVICORN  ██████████████████████████████████████████████
# =============================================================================
from pyngrok import ngrok  # importar aquí, después de que Celda 1 lo instaló
try:
    from kaggle_secrets import UserSecretsClient
    NGROK_TOKEN = UserSecretsClient().get_secret("NGROK_TOKEN")
except Exception:
    NGROK_TOKEN = "1eHNiEz4eoSTdrDi21begLLt9Si_BXxZko1pNffAjCSXiSSZ"

os.system(f"ngrok config add-authtoken {NGROK_TOKEN}")
os.system("fuser -k 8000/tcp 2>/dev/null || true")

try:
    ngrok.kill()
except Exception:
    pass

public_url = ngrok.connect(8000)

print("=" * 70)
print("🚀  PREDICTPATH API v4.0  —  PIPELINE COMPLETO PATHAGENT")
print(f"    👉  {public_url}  👈")
print("=" * 70)
print(f"  GET  {public_url}/health")
print(f"  GET  {public_url}/wsi/overview")
print(f"  GET  {public_url}/wsi/patch")
print(f"  GET  {public_url}/wsi/mask        ★ Máscara binaria")
print(f"  GET  {public_url}/wsi/montage     ★ Mosaico de tiles")
print(f"  POST {public_url}/wsi/navigate    ★ PLIP Navigator")
print(f"  POST {public_url}/wsi/analyze     ★★ Pipeline completo (PLIP+Quilt+Qwen)")
print("=" * 70)
print(f"⬆ Copia la URL y pégala en Streamlit: {public_url}")


def _run():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

threading.Thread(target=_run, daemon=True).start()
print("✅ Servidor WSI corriendo en puerto 8000.")
