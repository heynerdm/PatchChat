#!/usr/bin/env python3
"""
PREDICTPATH // NEURAL WSI COCKPIT v3.0
==============================================================================
Interfaz de Telepatología y Diagnóstico Agéntico Multimodal
Diseño: Cyber-Medical Dark Glassmorphism
Arquitectura: PathAgent (Navigator PLIP + Zoom Adaptativo + Executor Llama-3)
==============================================================================
"""

import streamlit as st
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFilter
from io import BytesIO
import base64
import time
import requests
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="PredictPath // Neural Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILOS CSS FUTURISTAS (CYBER-MEDICAL GLASSMORPHISM)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');

    /* Variables de Color */
    :root {
        --bg-main: #060911;
        --card-bg: rgba(13, 20, 36, 0.75);
        --neon-cyan: #00f2fe;
        --neon-blue: #4facfe;
        --neon-violet: #8a2be2;
        --neon-green: #00f5a0;
        --neon-pink: #ff007f;
        --border-glass: rgba(0, 242, 254, 0.22);
        --text-muted: #8492a6;
    }

    /* Fondo principal */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #060911 90%);
        color: #e2e8f0;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header Superior */
    .top-telemetry-bar {
        background: rgba(10, 16, 30, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 242, 254, 0.25);
        border-radius: 12px;
        padding: 12px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 30px rgba(0, 242, 254, 0.1);
    }

    .brand-title {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: 2px;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #9b51e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 245, 160, 0.1);
        border: 1px solid rgba(0, 245, 160, 0.4);
        color: #00f5a0;
        padding: 4px 14px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 1px;
    }

    .status-pulse {
        width: 8px;
        height: 8px;
        background: #00f5a0;
        border-radius: 50%;
        box-shadow: 0 0 10px #00f5a0;
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.7; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.7; }
    }

    /* Tarjetas Glassmorphism */
    .glass-panel {
        background: var(--card-bg);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        transition: border 0.3s ease;
    }
    .glass-panel:hover {
        border-color: rgba(0, 242, 254, 0.45);
    }

    /* Títulos de Panel */
    .panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--neon-cyan);
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px dashed rgba(0, 242, 254, 0.2);
        padding-bottom: 8px;
    }

    /* Métricas HUD */
    .hud-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 10px;
    }
    .hud-card {
        background: rgba(18, 28, 51, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .hud-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 800;
        color: #ffffff;
    }
    .hud-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* CoT (Chain of Thought) Timeline */
    .cot-step {
        position: relative;
        padding-left: 24px;
        padding-bottom: 14px;
        border-left: 2px solid rgba(0, 242, 254, 0.3);
        margin-left: 8px;
    }
    .cot-step::before {
        content: '';
        position: absolute;
        left: -6px;
        top: 2px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: var(--neon-cyan);
        box-shadow: 0 0 8px var(--neon-cyan);
    }
    .cot-step.complete::before {
        background: var(--neon-green);
        box-shadow: 0 0 8px var(--neon-green);
    }
    .cot-step.active::before {
        background: var(--neon-pink);
        box-shadow: 0 0 10px var(--neon-pink);
    }
    .cot-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .cot-desc {
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.4;
        margin-top: 2px;
    }

    /* Reporte Clínico */
    .report-card {
        background: linear-gradient(135deg, rgba(16, 27, 49, 0.9) 0%, rgba(10, 16, 30, 0.95) 100%);
        border: 1px solid rgba(79, 172, 254, 0.4);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.08);
    }
    .diagnosis-tag {
        background: rgba(255, 0, 127, 0.15);
        border: 1px solid rgba(255, 0, 127, 0.5);
        color: #ff3399;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }

    /* Retícula de Enfoque */
    .reticle-box {
        position: relative;
        border: 1px solid rgba(0, 242, 254, 0.4);
        border-radius: 8px;
        overflow: hidden;
    }
    .reticle-overlay {
        position: absolute;
        top: 10px;
        left: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        background: rgba(0, 0, 0, 0.65);
        color: var(--neon-cyan);
        padding: 4px 8px;
        border-radius: 4px;
        border: 1px solid rgba(0, 242, 254, 0.3);
        pointer-events: none;
    }

    /* Mosaico de Tiles */
    .montage-wrapper {
        border: 1px solid rgba(138, 43, 226, 0.4);
        border-radius: 10px;
        overflow: hidden;
        background: #060911;
        padding: 4px;
    }
    .montage-stats {
        display: flex;
        gap: 16px;
        margin-top: 10px;
        flex-wrap: wrap;
    }
    .montage-stat-chip {
        background: rgba(138, 43, 226, 0.12);
        border: 1px solid rgba(138, 43, 226, 0.4);
        color: #bf80ff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 10px;
        border-radius: 20px;
    }

    /* Máscara de Segmentación */
    .mask-wrapper {
        border: 1px solid rgba(0, 245, 160, 0.3);
        border-radius: 10px;
        overflow: hidden;
        background: #000;
        padding: 4px;
    }
    .mask-legend {
        display: flex;
        gap: 14px;
        margin-top: 8px;
        font-size: 0.72rem;
        color: #94a3b8;
        font-family: 'JetBrains Mono', monospace;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# GENERACIÓN DE MUESTRAS VISUALES SINTÉTICAS (Para prototipado visual)
# ==============================================================================
@st.cache_data
def generate_synthetic_wsi_overview():
    """Genera una imagen macro simulada de WSI con mapa de calor y cuadrícula RoI."""
    w, h = 600, 400
    img = Image.new("RGB", (w, h), (18, 24, 38))
    draw = ImageDraw.Draw(img)
    
    # Dibujar contorno de tejido (forma orgánica en tonos rosáceos/morados)
    tissue_color = (180, 85, 140)
    stroma_color = (130, 60, 110)
    
    draw.ellipse([80, 60, 480, 320], fill=stroma_color)
    draw.ellipse([140, 90, 440, 290], fill=tissue_color)
    draw.ellipse([200, 120, 320, 220], fill=(210, 50, 90)) # Foco atípico
    
    # Suavizado simulando slide
    img = img.filter(ImageFilter.GaussianBlur(12))
    draw = ImageDraw.Draw(img)
    
    # Dibujar cuadrícula de escaneo (scanner grid)
    for x in range(0, w, 40):
        draw.line([(x, 0), (x, h)], fill=(30, 45, 70), width=1)
    for y in range(0, h, 40):
        draw.line([(0, y), (w, y)], fill=(30, 45, 70), width=1)
        
    # Dibujar Bounding Boxes de los parches Top 10% seleccionados por el Navigator (PLIP)
    rois = [
        (200, 120, 240, 160),
        (240, 140, 280, 180),
        (260, 160, 300, 200),
        (180, 180, 220, 220),
        (320, 140, 360, 180),
    ]
    for r in rois:
        draw.rectangle(r, outline=(0, 242, 254), width=2)
        
    # Parche de enfoque primario (en magenta neón)
    primary = (240, 140, 280, 180)
    draw.rectangle(primary, outline=(255, 0, 127), width=3)
    
    return img

@st.cache_data
def generate_synthetic_highres_patch():
    """Genera un parche celular simulado en alta resolución."""
    w, h = 400, 400
    img = Image.new("RGB", (w, h), (220, 160, 190))
    draw = ImageDraw.Draw(img)
    
    # Dibujar núcleos atípicos hipercromáticos (células cancerígenas simuladas)
    np.random.seed(42)
    for _ in range(85):
        cx = np.random.randint(20, 380)
        cy = np.random.randint(20, 380)
        rad = np.random.randint(6, 16)
        color = (np.random.randint(40, 70), np.random.randint(15, 35), np.random.randint(70, 110))
        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=color)
        
    img = img.filter(ImageFilter.GaussianBlur(1))
    return img

# ==============================================================================
# VERIFICACIÓN DE CONEXIÓN CON KAGGLE Y FETCH WSI
# ==============================================================================
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Connection": "close"
}

def check_kaggle_connection(url: str):
    """Verifica el estado del puente FastAPI/Ngrok en Kaggle."""
    if not url or "ngrok" not in url or "pathology-node" in url:
        return {"online": False, "latency": 0, "gpu": "DESCONECTADO", "vram": "0/0 GB", "device": "N/A"}
    url = url.rstrip("/")
    try:
        t0 = time.time()
        resp = requests.get(f"{url}/health", headers=NGROK_HEADERS, timeout=4.0)
        latency_ms = round((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "online": True,
                "latency": latency_ms,
                "gpu": data.get("gpu_name", data.get("wsi_file", "Kaggle WSI Server")),
                "vram": f"{data.get('vram_used_gb', 0)} / {data.get('vram_total_gb', 0)} GB" if "vram_total_gb" in data else "WSI Engine",
                "device": data.get("device", "cuda"),
                "raw": data
            }
    except Exception:
        pass
    return {"online": False, "latency": 0, "gpu": "ERROR CONEXIÓN", "vram": "0/0 GB", "device": "N/A"}

@st.cache_data(ttl=60)
def fetch_wsi_overview(url: str):
    """Descarga el thumbnail con contornos y métricas de la lámina WSI en Kaggle."""
    try:
        resp = requests.get(f"{url.rstrip('/')}/wsi/overview", headers=NGROK_HEADERS, timeout=20.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60)
def fetch_wsi_patch(url: str, x: int, y: int, size: int = 224, level: int = 0):
    """Descarga un parche en ultra resolución en coordenadas x, y desde Kaggle."""
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/wsi/patch?x={x}&y={y}&size={size}&level={level}",
            headers=NGROK_HEADERS,
            timeout=10.0
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("patch_b64"):
                b64_bytes = base64.b64decode(data["patch_b64"])
                return Image.open(BytesIO(b64_bytes))
    except Exception:
        pass
    return None

@st.cache_data(ttl=120)
def fetch_segmentation_mask(url: str):
    """Descarga la máscara de segmentación binaria (PNG) desde Kaggle."""
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/wsi/mask",
            headers=NGROK_HEADERS,
            timeout=20.0
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("mask_b64"):
                return base64.b64decode(data["mask_b64"]), data
    except Exception:
        pass
    return None, None

def fetch_tile_montage(url: str, patch_size: int = 224,
                       grid_cols: int = 12, max_patches: int = 300, level: int = 1):
    """Genera y descarga el mosaico de tiles de tejido desde Kaggle (sin caché — puede ser lento)."""
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/wsi/montage"
            f"?patch_size={patch_size}&grid_cols={grid_cols}"
            f"&max_patches={max_patches}&level={level}",
            headers=NGROK_HEADERS,
            timeout=120.0  # puede tardar varios minutos para WSIs grandes
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("montage_b64"):
                img_bytes = base64.b64decode(data["montage_b64"])
                return Image.open(BytesIO(img_bytes)), data
    except Exception as e:
        st.error(f"Error al obtener mosaico: {e}")
    return None, None

def execute_wsi_navigation(url: str, query: str, top_pct: float = 0.10):
    """Envía la consulta clínica al Navigator (PLIP) en Kaggle y recibe el Top 10% de RoIs."""
    for attempt in range(2):
        try:
            with requests.Session() as session:
                resp = session.post(
                    f"{url.rstrip('/')}/wsi/navigate",
                    json={"query": query, "top_pct": top_pct},
                    headers=NGROK_HEADERS,
                    timeout=45.0
                )
                if resp.status_code == 200:
                    return resp.json()
                elif attempt == 1:
                    st.error(f"Error {resp.status_code} del Navigator: {resp.text[:200]}")
        except Exception as e:
            if attempt == 1:
                st.error(f"Error de conexión con el Navigator: {e}")
            time.sleep(1)
    return None

def execute_full_analysis(url: str, query: str, top_pct: float = 0.10,
                          max_perceptor_patches: int = 5, magnification: int = 20):
    """
    Pipeline completo PathAgent: Navigator (PLIP) → Perceptor (Qwen) → Executor (Qwen CoT).
    Llama a POST /wsi/analyze en el servidor Kaggle.
    """
    for attempt in range(2):
        try:
            with requests.Session() as session:
                resp = session.post(
                    f"{url.rstrip('/')}/wsi/analyze",
                    json={
                        "query": query,
                        "top_pct": top_pct,
                        "max_perceptor_patches": max_perceptor_patches,
                        "magnification": magnification,
                    },
                    headers=NGROK_HEADERS,
                    timeout=300.0  # pipeline completo puede tomar 1-3 min
                )
                if resp.status_code == 200:
                    return resp.json()
                elif attempt == 1:
                    st.error(f"Error {resp.status_code} del servidor: {resp.text[:200]}")
        except Exception as e:
            if attempt == 1:
                st.error(f"Error de conexión con el pipeline: {e}")
            time.sleep(1)
    return None

# ==============================================================================
# SIDEBAR // PARÁMETROS DEL AGENTE Y CARGA DE LÁMINA
# ==============================================================================
with st.sidebar:
    st.markdown("### Conexión Neuronal")
    kaggle_url = st.text_input(
        "Túnel Ngrok (Kaggle T4):",
        value=st.session_state.get("kaggle_url", "https://771f-34-32-188-228.ngrok-free.app"),
        placeholder="https://xxxxx.ngrok-free.app",
        help="Endpoint del servidor FastAPI en Kaggle."
    )
    st.session_state.kaggle_url = kaggle_url

    btn_ping = st.button("Probar Conexión", use_container_width=True)
    conn_info = check_kaggle_connection(kaggle_url)

    if conn_info["online"]:
        st.markdown(f"""
        <div style="background:rgba(0,245,160,0.1); border:1px solid rgba(0,245,160,0.3); border-radius:8px; padding:8px 12px; margin-top:8px;">
            <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#00f5a0; font-weight:700;">PUENTE ACTIVO</div>
            <div style="font-size:0.7rem; color:#cbd5e1; margin-top:2px;">Dispositivo: <b>{conn_info['gpu']}</b></div>
            <div style="font-size:0.7rem; color:#8492a6;">VRAM: {conn_info['vram']} | Latencia: {conn_info['latency']}ms</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,0,127,0.1); border:1px solid rgba(255,0,127,0.3); border-radius:8px; padding:8px 12px; margin-top:8px;">
            <div style="font-family:'JetBrains Mono'; font-size:0.75rem; color:#ff3399; font-weight:700;">DESCONECTADO</div>
            <div style="font-size:0.7rem; color:#94a3b8; margin-top:2px;">Inicia el script en Kaggle y pega la URL de Ngrok.</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Muestra Histopatológica (WSI)")
    
    wsi_source = st.selectbox(
        "Seleccionar Entrada WSI:",
        [
            "044f28b0-564a-42d8-9df1-ba3fd4d6c014.svs (TCGA-BRCA Mama)",
            "GCHTID-GAST-08-TUM (Adenocarcinoma Gástrico)",
            "ARCH-BENCH-44 (Muestra de Validación)",
            "[+] Subir archivo local (.svs, .ndpi, .tiff)"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Parámetros del Agente")
    
    mag_base = st.select_slider(
        "Magnificación Inicial (Prospección):",
        options=["2.5x", "5x", "10x"],
        value="5x"
    )
    
    roi_top_pct = st.slider(
        "Filtro Navigator (Top % Parches RoI):",
        min_value=5, max_value=25, value=10, step=5,
        help="Según el paper de PathAgent, el 10% ofrece el pico de máxima precisión diagnóstica."
    )
    
    max_iterations = st.number_input("Máx. Iteraciones de Razonamiento:", min_value=1, max_value=5, value=3)
    
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem; color:#64748b; text-align:center; font-family:\"JetBrains Mono\";'>"
        "PREDICTPATH CORE v3.0-WSI<br>LABORATORIO DE PATOLOGÍA DIGITAL</div>",
        unsafe_allow_html=True
    )

# ==============================================================================
# HEADER SUPERIOR // TELEMETRÍA DEL SISTEMA
# ==============================================================================
badge_color = "#00f5a0" if conn_info["online"] else "#ff3399"
badge_text = f"KAGGLE ONLINE ({conn_info['gpu']})" if conn_info["online"] else "KAGGLE DESCONECTADO"
latency_text = f"{conn_info['latency']}ms" if conn_info["online"] else "-- ms"

st.markdown(f"""
<div class="top-telemetry-bar">
    <div class="brand-title">
        PREDICTPATH <span style="font-weight:300; opacity:0.6;">// NEURAL WSI COCKPIT</span>
    </div>
    <div style="display:flex; gap:16px; align-items:center;">
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#8492a6;">
            LATENCIA: <span style="color:#00f2fe;">{latency_text}</span> | ENGINE: <span style="color:#fff;">PATHAGENT-DUAL</span>
        </span>
        <div class="status-badge" style="border-color:{badge_color}66; color:{badge_color}; background:{badge_color}1a;">
            <div class="status-pulse" style="background:{badge_color}; box-shadow:0 0 10px {badge_color};"></div> {badge_text}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# CUERPO PRINCIPAL // DIVISIÓN EN 2 COLUMNAS ESTRATÉGICAS
# ==============================================================================
col_vision, col_agent = st.columns([1.15, 1.0], gap="medium")

# ------------------------------------------------------------------------------
# COLUMNA 1: DECK DE VISIÓN (MACRO WSI + ZOOM DE ALTA RESOLUCIÓN)
# ------------------------------------------------------------------------------
with col_vision:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-header">NAVEGADOR MACROSCÓPICO // VISTA COMPLETA DE LÁMINA</div>',
        unsafe_allow_html=True
    )
    
    # Comprobar si tenemos conexión viva para cargar la WSI real desde Kaggle
    real_wsi = None
    if conn_info["online"] and kaggle_url:
        real_wsi = fetch_wsi_overview(kaggle_url)

    nav_data = st.session_state.get("nav_res")

    if real_wsi and "thumbnail_b64" in real_wsi:
        b64_bytes = base64.b64decode(real_wsi["thumbnail_b64"])
        wsi_thumb = Image.open(BytesIO(b64_bytes)).convert("RGB")
        
        # Si ya se ejecutó la navegación con PLIP, pintar los macro-bloques del Top 10%
        if nav_data and "rois" in nav_data:
            draw = ImageDraw.Draw(wsi_thumb)
            scale_x = wsi_thumb.width / real_wsi['width']
            scale_y = wsi_thumb.height / real_wsi['height']

            for r in nav_data["rois"]:
                tx = int(r["x"] * scale_x)
                ty = int(r["y"] * scale_y)
                macro_sz = r.get("macro_size", 4096)
                bw = max(8, int(macro_sz * scale_x))
                bh = max(8, int(macro_sz * scale_y))
                draw.rectangle([tx, ty, tx + bw, ty + bh],
                               outline=(0, 242, 254), width=2)

            # Foco #1 en magenta neón (mayor score semántico)
            b0 = nav_data["rois"][0]
            tx0 = int(b0["x"] * scale_x)
            ty0 = int(b0["y"] * scale_y)
            macro_sz0 = b0.get("macro_size", 4096)
            bw0 = max(8, int(macro_sz0 * scale_x))
            bh0 = max(8, int(macro_sz0 * scale_y))
            draw.rectangle([tx0 - 1, ty0 - 1, tx0 + bw0 + 1, ty0 + bh0 + 1],
                           outline=(255, 0, 127), width=3)

        dim_str = f"{real_wsi['width']:,} × {real_wsi['height']:,}"
        cov_str = f"{real_wsi['tissue_coverage_pct']}%"
        cont_str = f"{nav_data['top_k_count']} RoIs (Top 10%)" if nav_data else f"{real_wsi['contours_count']} Focos"
        caption_str = f"WSI Real: {real_wsi['filename']} | Segmentación Otsu + Top RoIs PLIP" if nav_data else f"WSI Real: {real_wsi['filename']} | Segmentación Otsu (Contorno Cian)"
    else:
        wsi_thumb = generate_synthetic_wsi_overview()
        dim_str = "118,400 × 86,200"
        cov_str = "34.8%"
        cont_str = "4,280 / 142"
        caption_str = "WSI Radar: Segmentación de tejido (violeta) + Cuadrícula RoI (modo simulación)"

    st.image(wsi_thumb, use_container_width=True, caption=caption_str)
    
    # Telemetría de la lámina
    st.markdown(f"""
    <div class="hud-grid">
        <div class="hud-card">
            <div class="hud-val">{dim_str}</div>
            <div class="hud-label">Dimensiones (px)</div>
        </div>
        <div class="hud-card">
            <div class="hud-val">{cont_str}</div>
            <div class="hud-label">Regiones Detectadas</div>
        </div>
        <div class="hud-card">
            <div class="hud-val">{cov_str}</div>
            <div class="hud-label">Cobertura Tejido</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Panel de Enfoque Microscópico Dinámico
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-header">RETÍCULA DE ALTA RESOLUCIÓN // ZOOM ADAPTATIVO</div>',
        unsafe_allow_html=True
    )
    
    col_zoom_btn1, col_zoom_btn2, col_zoom_btn3, col_zoom_btn4 = st.columns(4)
    with col_zoom_btn1:
        st.button("5x Panorámica", use_container_width=True)
    with col_zoom_btn2:
        st.button("10x Estructura", use_container_width=True)
    with col_zoom_btn3:
        st.button("20x Núcleos", use_container_width=True, type="primary")
    with col_zoom_btn4:
        st.button("40x Mitosis", use_container_width=True)
        
    # Obtener parche real devuelto por el Navigator o extracción por defecto
    patch_img = None
    coord_caption = ""

    if nav_data and nav_data.get("primary_patch_b64"):
        b64_p = base64.b64decode(nav_data["primary_patch_b64"])
        patch_img = Image.open(BytesIO(b64_p))
        b0 = nav_data["rois"][0]
        coord_caption = f"FOCO #1 NAVIGATOR (PLIP): (X: {b0['x']:,}, Y: {b0['y']:,}) | SIMILITUD: {b0['score']*100:.1f}%"
    elif conn_info["online"] and kaggle_url:
        real_patch = fetch_wsi_patch(kaggle_url, x=15000, y=15000, size=350, level=0)
        if real_patch:
            patch_img = real_patch
            coord_caption = "COORD REAL: (X: 15,000, Y: 15,000) | RESOLUCIÓN ULTRA: Nivel 0"

    if patch_img is None:
        patch_img = generate_synthetic_highres_patch()
        coord_caption = "COORD: (X: 16,016, Y: 7,240) | MAG: 20x | RESOLUCIÓN: 0.25 µm/px"
    
    # Envolver en retícula con coordenadas
    st.markdown('<div class="reticle-box">', unsafe_allow_html=True)
    st.image(patch_img, use_container_width=True)
    st.markdown(
        f'<div class="reticle-overlay">{coord_caption}</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# COLUMNA 2: CENTRO DE COGNICIÓN Y RAZONAMIENTO DEL AGENTE (PATHAGENT)
# ------------------------------------------------------------------------------
with col_agent:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-header">CONSOLA AGÉNTICA // CONSULTA DIAGNÓSTICA</div>',
        unsafe_allow_html=True
    )

    query_preset = st.selectbox(
        "Consulta de Interés Clínico:",
        [
            "Diagnóstico de subtipo y grado nuclear de la lesión",
            "Evaluación de invasión estromal y márgenes quirúrgicos",
            "Caracterización de infiltración linfocítica peritumoral",
            "Generar reporte histopatológico integral completo"
        ]
    )

    user_prompt = st.text_area(
        "Instrucción detallada:",
        value="Identifica el tipo de carcinoma presente en el corte, evalúa la atipia celular nuclear y determina si se observan patrones infiltrativos en el estroma perilesional.",
        height=75
    )

    # Parámetros avanzados del pipeline completo
    with st.expander("Parámetros del Pipeline Completo", expanded=False):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            n_perceptor = st.slider("Parches a describir (Perceptor):", 1, 10, 5,
                help="Cuántos RoIs describe Quilt-LLaVA antes del razonamiento CoT")
        with col_p2:
            mag_analyze = st.select_slider("Magnificación de referencia:",
                options=[5, 10, 20, 40], value=20)

    col_btn1, col_btn2 = st.columns([1.4, 1])
    with col_btn1:
        btn_full = st.button(
            "PIPELINE COMPLETO (PLIP + Qwen CoT)",
            use_container_width=True,
            type="primary",
            key="btn_full_pipeline"
        )
    with col_btn2:
        btn_nav_only = st.button(
            "Solo Navigator (PLIP)",
            use_container_width=True,
            key="btn_nav_only"
        )

    # ── Botón: Pipeline Completo ──────────────────────────────────────────
    if btn_full:
        if not conn_info["online"]:
            st.error("Sin conexión al servidor Kaggle. Verifica la URL de Ngrok.")
        else:
            progress_bar = st.progress(0, text="Iniciando pipeline PathAgent...")
            progress_bar.progress(10, text="Navigator (PLIP): rankeando RoIs...")
            full_result = execute_full_analysis(
                kaggle_url, user_prompt,
                top_pct=roi_top_pct / 100.0,
                max_perceptor_patches=n_perceptor,
                magnification=mag_analyze
            )
            if full_result and full_result.get("status") == "success":
                progress_bar.progress(100, text="Pipeline completo finalizado.")
                st.session_state["full_result"] = full_result
                # También guardar nav_res para que el thumbnail dibuje los bounding boxes
                st.session_state["nav_res"] = full_result["navigator"]
                st.success(
                    f"Pipeline completo: {full_result['navigator']['top_k_count']} RoIs navegados · "
                    f"{full_result['perceptor']['patches_described']} parches descritos · "
                    f"CoT Qwen2.5-VL finalizado."
                )
                st.rerun()
            else:
                progress_bar.empty()
                st.error("El pipeline no pudo completarse. Verifica que los modelos estén cargados en Kaggle.")

    # ── Botón: Solo Navigator ──────────────────────────────────────────────
    if btn_nav_only:
        if not conn_info["online"]:
            st.error("Sin conexión al servidor Kaggle.")
        else:
            with st.spinner("Navigator (PLIP) escaneando parches..."):
                nav_result = execute_wsi_navigation(kaggle_url, user_prompt, top_pct=roi_top_pct / 100.0)
                if nav_result and nav_result.get("status") == "success":
                    st.session_state["nav_res"] = nav_result
                    st.success(f"Navigator aisló {nav_result['top_k_count']} parches (Top {roi_top_pct}%).")
                    st.rerun()
                else:
                    st.error("No se pudo completar la navegación.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Recuperar resultado del pipeline de sesión ─────────────────────────
    full_result = st.session_state.get("full_result")
    nav_data    = st.session_state.get("nav_res")

    # TRAYECTORIA DE RAZONAMIENTO (CHAIN OF THOUGHT)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-header">TRAYECTORIA MULTI-PASO (CHAIN-OF-THOUGHT)</div>',
        unsafe_allow_html=True
    )

    if full_result:
        # ── CoT REAL del pipeline completo ────────────────────────────────
        nav   = full_result["navigator"]
        perc  = full_result["perceptor"]
        exe   = full_result["executor"]
        b0    = nav["rois"][0]
        cot   = exe.get("cot_steps", {})
        suff  = exe.get("sufficient", "?")
        suff_color = "#00f5a0" if str(suff).lower() == "yes" else "#ff3399"

        # Paso 1 — Navigator
        st.markdown(f"""
        <div class="cot-step complete">
            <div class="cot-title">PASO 1 ▸ NAVIGATOR — PLIP (Top {roi_top_pct}% RoIs)</div>
            <div class="cot-desc">
                <b>{nav['total_evaluated']}</b> parches evaluados.
                Consulta: <i>"{full_result['query']}"</i><br>
                Foco primario → Coord (<b>{b0['x']:,}</b>, <b>{b0['y']:,}</b>) |
                Score PLIP: <b>{b0['score']*100:.1f}%</b> |
                RoIs aislados: <b>{nav['top_k_count']}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Paso 2 — Perceptor
        st.markdown(f"""
        <div class="cot-step complete">
            <div class="cot-title">PASO 2 ▸ PERCEPTOR — QUILT-LLAVA (descripción morfológica)</div>
            <div class="cot-desc">
                <b>{perc['patches_described']}</b> parches descritos morfológicamente.<br>
                <span style="color:#94a3b8; font-size:0.72rem;">
                {perc['descriptions'][0]['description'][:220]}…
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Paso 3A — PredictAnswer
        step_a = cot.get("step_a", {})
        st.markdown(f"""
        <div class="cot-step complete">
            <div class="cot-title">PASO 3A ▸ EXECUTOR — PREDICT ANSWER (Qwen2.5-VL)</div>
            <div class="cot-desc">
                <b>Respuesta preliminar:</b> {step_a.get('answer','—')}<br>
                <span style="color:#94a3b8; font-size:0.72rem;">
                {str(step_a.get('thinking_steps',''))[:280]}…
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Paso 3B — SelfReflect
        st.markdown(f"""
        <div class="cot-step {'complete' if str(suff).lower()=='yes' else 'active'}">
            <div class="cot-title">PASO 3B ▸ EXECUTOR — SELF-REFLECT (suficiencia de evidencia)</div>
            <div class="cot-desc">
                Dictamen de suficiencia:
                <strong style="color:{suff_color};">{'SÍ ✅' if str(suff).lower()=='yes' else 'NO ⚠️'}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Paso 3C — ExploreMissing (solo si no fue suficiente)
        step_c = cot.get("step_c")
        if step_c:
            st.markdown(f"""
            <div class="cot-step active">
                <div class="cot-title">PASO 3C ▸ EXECUTOR — EXPLORE MISSING (zoom adaptativo)</div>
                <div class="cot-desc">
                    Información faltante: <b>{step_c.get('missing_info','—')}</b><br>
                    Zoom recomendado: <b>{step_c.get('zoom_recommendation','—')}</b>
                    {'→ Nivel ' + str(step_c.get('recommended_zoom_level','')) if step_c.get('zoom_recommendation','').lower()=='yes' else ''}<br>
                    <span style="color:#94a3b8; font-size:0.72rem;">{step_c.get('zoom_reason','')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    elif nav_data and "rois" in nav_data:
        # ── Solo Navigator ejecutado (sin Perceptor/Executor) ─────────────
        b0 = nav_data["rois"][0]
        st.markdown(f"""
        <div class="cot-step complete">
            <div class="cot-title">PASO 1 ▸ NAVIGATOR — PLIP (Top {roi_top_pct}% RoIs)</div>
            <div class="cot-desc">
                {nav_data['total_evaluated']} parches evaluados. Foco primario en
                ({b0['x']:,}, {b0['y']:,}) con score {b0['score']*100:.1f}%.
                <br><span style="color:#ff9900; font-size:0.72rem;">
                ⚠️ Ejecuta el <b>Pipeline Completo</b> para obtener descripciones Quilt-LLaVA y el CoT de Qwen.
                </span>
            </div>
        </div>
        <div class="cot-step">
            <div class="cot-title">PASO 2 ▸ PERCEPTOR — pendiente</div>
            <div class="cot-desc">Quilt-LLaVA no ejecutado.</div>
        </div>
        <div class="cot-step">
            <div class="cot-title">PASO 3 ▸ EXECUTOR CoT — pendiente</div>
            <div class="cot-desc">Qwen2.5-VL no ejecutado.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cot-step">
            <div class="cot-title">PASO 1 ▸ NAVIGATOR — en espera</div>
            <div class="cot-desc">Presiona <strong>PIPELINE COMPLETO</strong> para iniciar.</div>
        </div>
        <div class="cot-step">
            <div class="cot-title">PASO 2 ▸ PERCEPTOR — en espera</div>
            <div class="cot-desc">Quilt-LLaVA describirá los parches del Top-K.</div>
        </div>
        <div class="cot-step">
            <div class="cot-title">PASO 3 ▸ EXECUTOR CoT — en espera</div>
            <div class="cot-desc">Qwen2.5-VL ejecutará el razonamiento de 3 pasos.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # REPORTE CLÍNICO FINAL ESTRUCTURADO
    st.markdown('<div class="report-card">', unsafe_allow_html=True)

    if full_result:
        exe = full_result["executor"]
        nav = full_result["navigator"]
        b0  = nav["rois"][0]
        answer_text = exe.get("answer", "—")
        thinking    = exe.get("thinking_steps", "")
        suff        = exe.get("sufficient", "?")

        st.markdown(f"""
        <div class="diagnosis-tag">DIAGNÓSTICO AGÉNTICO: {answer_text.upper()}</div>
        <p style="font-size:0.8rem; color:#cbd5e1; line-height:1.6;">
            <strong>RAZONAMIENTO (CADENA DE PENSAMIENTO QWEN2.5-VL):</strong><br>
            {thinking[:600]}{'…' if len(thinking)>600 else ''}
        </p>
        <div style="display:flex; justify-content:space-between; margin-top:12px;
                    border-top:1px solid rgba(255,255,255,0.1); padding-top:10px; flex-wrap:wrap; gap:8px;">
            <span style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#00f5a0;">
                PLIP SCORE FOCO #1: {b0['score']*100:.1f}%
            </span>
            <span style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#4facfe;">
                RoIs: {nav['top_k_count']} | Parches descritos: {full_result['perceptor']['patches_described']}
            </span>
            <span style="font-family:'JetBrains Mono'; font-size:0.72rem;
                         color:{'#00f5a0' if str(suff).lower()=='yes' else '#ff3399'};">
                SUFICIENCIA: {suff.upper() if suff else '—'}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Expandir descripciones del Perceptor
        with st.expander("Ver descripciones morfológicas (Quilt-LLaVA)", expanded=False):
            for d in full_result["perceptor"]["descriptions"]:
                st.markdown(f"""
                <div style="border-left:2px solid rgba(0,242,254,0.4); padding:8px 12px;
                            margin-bottom:10px; background:rgba(0,0,0,0.2); border-radius:0 6px 6px 0;">
                    <div style="font-family:'JetBrains Mono'; font-size:0.7rem; color:#4facfe; margin-bottom:4px;">
                        PARCHE #{d['patch_idx']} | Coord ({d['x']:,}, {d['y']:,}) |
                        PLIP: {d['plip_score']*100:.1f}%
                    </div>
                    <div style="font-size:0.78rem; color:#cbd5e1; line-height:1.5;">
                        {d['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Botón exportar JSON real
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            report_json = json.dumps(full_result, indent=2, ensure_ascii=False)
            st.download_button(
                "Exportar Informe (JSON)",
                data=report_json,
                file_name="predictpath_report.json",
                mime="application/json",
                use_container_width=True
            )
        with col_exp2:
            st.button("Generar PDF Clínico", use_container_width=True)

    elif nav_data and "rois" in nav_data:
        b0 = nav_data["rois"][0]
        st.markdown(f"""
        <div class="diagnosis-tag">ESTADO: NAVIGATOR COMPLETADO — EXECUTOR PENDIENTE</div>
        <p style="font-size:0.8rem; color:#cbd5e1; line-height:1.5;">
            PLIP aisló <strong>{nav_data['top_k_count']} regiones críticas</strong>.
            Foco primario: coord ({b0['x']:,}, {b0['y']:,}) con {b0['score']*100:.1f}% de similitud.<br>
            <span style="color:#ff9900;">Ejecuta el <b>Pipeline Completo</b> para obtener el diagnóstico de Qwen2.5-VL.</span>
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.button("Exportar Informe (JSON)", use_container_width=True)
        with col_exp2:
            st.button("Generar PDF Clínico", use_container_width=True)
    else:
        st.markdown("""
        <div class="diagnosis-tag">ESTADO: EN ESPERA DE CONSULTA CLÍNICA</div>
        <p style="font-size:0.8rem; color:#cbd5e1; line-height:1.5;">
            Selecciona una consulta médica y pulsa <strong>PIPELINE COMPLETO</strong>
            para activar Navigator → Quilt-LLaVA → Qwen2.5-VL CoT.
        </p>
        """, unsafe_allow_html=True)
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            st.button("Exportar Informe (JSON)", use_container_width=True)
        with col_exp2:
            st.button("Generar PDF Clínico", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# SECCIÓN INFERIOR // TILING Y SEGMENTACIÓN
# ==============================================================================
st.markdown("---")
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; font-weight:700;
            color:#bf80ff; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:16px;">
    TILING & SEGMENTACIÓN
</div>
""", unsafe_allow_html=True)

tab_tiling, tab_mask = st.tabs(["Mosaico de Tiles (Tiling)", "Máscara de Segmentación"])

# ---------------------------------------------------------------------------
# TAB 1: MOSAICO DE TILES
# ---------------------------------------------------------------------------
with tab_tiling:
    st.markdown("""
    <div class="glass-panel">
    <div class="panel-header">TILING — MOSAICO COMPLETO DE TEJIDO WSI</div>
    <p style="font-size:0.8rem; color:#94a3b8; line-height:1.6; margin-bottom:12px;">
        Genera una única imagen que contiene <strong>todos los tiles de tejido</strong> detectados en la lámina WSI,
        organizados en una cuadrícula uniforme.
        Solo se incluyen tiles con &gt;50% de tejido viable (descarta vidrio/artefactos).
    </p>
    </div>
    """, unsafe_allow_html=True)

    col_mont_cfg1, col_mont_cfg2, col_mont_cfg3 = st.columns(3)
    with col_mont_cfg1:
        mont_patch_size = st.select_slider(
            "Tamaño de tile (px):",
            options=[112, 224, 336],
            value=224,
            key="mont_ps"
        )
    with col_mont_cfg2:
        mont_cols = st.slider("Columnas en cuadrícula:", min_value=6, max_value=20, value=12, key="mont_cols")
    with col_mont_cfg3:
        mont_max = st.slider("Máx. tiles:", min_value=100, max_value=500, value=300, step=50, key="mont_max")

    btn_montage = st.button(
        "GENERAR MOSAICO DE TILES",
        use_container_width=True,
        type="primary",
        key="btn_gen_montage"
    )

    if btn_montage:
        if not conn_info["online"]:
            st.error("Conecta Kaggle primero (URL Ngrok en la barra lateral).")
        else:
            with st.spinner(f"Generando mosaico ({mont_max} tiles a {mont_patch_size}px)... puede tardar 1-3 min en Kaggle..."):
                montage_img, montage_meta = fetch_tile_montage(
                    kaggle_url,
                    patch_size=mont_patch_size,
                    grid_cols=mont_cols,
                    max_patches=mont_max,
                    level=1
                )
            if montage_img:
                st.session_state["montage_img"] = montage_img
                st.session_state["montage_meta"] = montage_meta
                st.success(f"Mosaico generado: {montage_meta['total_tiles']} tiles · "
                           f"{montage_meta['grid_cols']}×{montage_meta['grid_rows']} cuadrícula")
            else:
                st.error("No se pudo generar el mosaico. Verifica que el servidor Kaggle esté corriendo.")

    # Mostrar mosaico si está en sesión
    montage_img = st.session_state.get("montage_img")
    montage_meta = st.session_state.get("montage_meta", {})

    if montage_img:
        st.markdown('<div class="montage-wrapper">', unsafe_allow_html=True)
        st.image(montage_img, caption="Mosaico de Tiles WSI — Solo tejido viable (Otsu > 50%)",
                 use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="montage-stats">
            <span class="montage-stat-chip">Tiles totales: {montage_meta.get('total_tiles', '—')}</span>
            <span class="montage-stat-chip">Cuadrícula: {montage_meta.get('grid_cols', 12)}×{montage_meta.get('grid_rows', '—')}</span>
            <span class="montage-stat-chip">Tamaño tile: {montage_meta.get('patch_size', 224)}×{montage_meta.get('patch_size', 224)} px</span>
            <span class="montage-stat-chip">Formato: {montage_meta.get('format', 'JPEG')}</span>
        </div>
        """, unsafe_allow_html=True)

        # Botón de descarga
        buf_mont = BytesIO()
        montage_img.save(buf_mont, format="JPEG", quality=90)
        st.download_button(
            label="Descargar Mosaico JPEG",
            data=buf_mont.getvalue(),
            file_name=f"predictpath_tiling_{montage_meta.get('total_tiles', 'N')}_tiles.jpg",
            mime="image/jpeg",
            use_container_width=True
        )
    else:
        st.info("Presiona GENERAR MOSAICO DE TILES para crear el tiling completo de la lámina WSI.")

# ---------------------------------------------------------------------------
# TAB 2: MÁSCARA DE SEGMENTACIÓN
# ---------------------------------------------------------------------------
with tab_mask:
    st.markdown("""
    <div class="glass-panel">
    <div class="panel-header">MÁSCARA DE SEGMENTACIÓN BINARIA (OTSU)</div>
    <p style="font-size:0.8rem; color:#94a3b8; line-height:1.6; margin-bottom:12px;">
        Extrae la <strong>máscara de segmentación</strong> de la lámina WSI como PNG binario.
        <span style="color:#fff; font-weight:600;">Blanco = Tejido viable</span> |
        <span style="color:#555;">Negro = Vidrio / Fondo</span>.
        Generada con umbralización Otsu sobre el canal de saturación HSV (sin aprendizaje profundo).
    </p>
    </div>
    """, unsafe_allow_html=True)

    btn_mask = st.button(
        "EXTRAER MÁSCARA DE SEGMENTACIÓN",
        use_container_width=True,
        type="primary",
        key="btn_gen_mask"
    )

    if btn_mask:
        if not conn_info["online"]:
            st.error("Conecta Kaggle primero.")
        else:
            with st.spinner("Calculando máscara de segmentación Otsu en Kaggle..."):
                mask_bytes, mask_meta = fetch_segmentation_mask(kaggle_url)
            if mask_bytes:
                st.session_state["seg_mask_bytes"] = mask_bytes
                st.session_state["seg_mask_meta"] = mask_meta
                st.success("Máscara de segmentación extraída correctamente.")
            else:
                st.error("No se pudo obtener la máscara. Verifica la conexión con Kaggle.")

    # Mostrar máscara si está en sesión
    seg_mask_bytes = st.session_state.get("seg_mask_bytes")
    seg_mask_meta  = st.session_state.get("seg_mask_meta", {})

    if seg_mask_bytes:
        mask_pil = Image.open(BytesIO(seg_mask_bytes))
        col_mask_img, col_mask_info = st.columns([1.5, 1])

        with col_mask_img:
            st.markdown('<div class="mask-wrapper">', unsafe_allow_html=True)
            st.image(mask_pil, caption="Máscara Binaria: Blanco=Tejido · Negro=Vidrio",
                     use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="mask-legend">
                <span><span class="legend-dot" style="background:#ffffff; border:1px solid #555;"></span> Tejido viable</span>
                <span><span class="legend-dot" style="background:#111; border:1px solid #555;"></span> Vidrio / Fondo</span>
            </div>
            """, unsafe_allow_html=True)

        with col_mask_info:
            # Calcular estadísticas de la máscara
            mask_arr = np.array(mask_pil)
            total_px = mask_arr.size
            tissue_px = int(np.count_nonzero(mask_arr))
            tissue_pct = round(tissue_px / total_px * 100, 2)
            glass_pct = round(100 - tissue_pct, 2)

            st.markdown(f"""
            <div class="glass-panel">
                <div class="panel-header">ESTADÍSTICAS</div>
                <div class="hud-grid" style="grid-template-columns: 1fr 1fr;">
                    <div class="hud-card">
                        <div class="hud-val" style="color:#00f5a0;">{tissue_pct}%</div>
                        <div class="hud-label">Tejido</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-val" style="color:#8492a6;">{glass_pct}%</div>
                        <div class="hud-label">Vidrio</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-val">{mask_pil.width}</div>
                        <div class="hud-label">Ancho (px)</div>
                    </div>
                    <div class="hud-card">
                        <div class="hud-val">{mask_pil.height}</div>
                        <div class="hud-label">Alto (px)</div>
                    </div>
                </div>
                <div style="margin-top:10px; font-size:0.72rem; color:#64748b; font-family:'JetBrains Mono';">
                    Método: Otsu (canal S, HSV)<br>
                    Post-proc: Morfológico (cierre 5×5 × 2 iter)<br>
                    Formato: PNG binario (L mode)
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Botón de descarga
        st.download_button(
            label="Descargar Máscara PNG",
            data=seg_mask_bytes,
            file_name="predictpath_segmentation_mask.png",
            mime="image/png",
            use_container_width=True
        )
    else:
        st.info("Presiona EXTRAER MÁSCARA para obtener la máscara de segmentación binaria de la lámina.")
