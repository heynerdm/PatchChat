#!/usr/bin/env python3
"""
CHAT PATHOLOGY v3 - MULTIMODAL
================================
Análisis histopatológico con arquitectura multimodal:
  PLIP (encoder visual) + Capa de Proyección + Llama-3-8B
  Inferencia en GPU de Kaggle via Ngrok
"""

import streamlit as st
from datetime import datetime
from io import BytesIO
from PIL import Image
import hashlib
import json
import os
import base64
import requests

# ==============================================================================
# CONFIGURACIÓN - PRODUCCIÓN
# ==============================================================================

st.set_page_config(
    page_title="Chat Pathology - Análisis Histopatológico",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración
BENCHMARK_PATH = "embeddings_benchmark.json"
INFERENCE_TIMEOUT = 300  # 5 minutos para cold start
APP_VERSION = "3.0-multimodal"

# ==============================================================================
# FUNCIONES DE CARGA DE DATOS
# ==============================================================================

@st.cache_resource
def load_benchmark_metadata():
    """
    Carga el archivo JSON con embeddings y metadatos clínicos.
    
    Estructura esperada:
    {
        "hash_md5_1": {
            "filename": "ADI_1245.png",
            "embedding": [768 valores float],
            "metadata_clinica": "Paciente: 45M, diagnóstico: adenocarcinoma mucinoso, T1bN0M0"
        },
        ...
    }
    
    Returns:
        dict: Diccionario con datos del benchmark (vacío si hay error)
    """
    if not os.path.exists(BENCHMARK_PATH):
        return {}
    
    try:
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error cargando benchmark: {e}")
        return {}


def analyze_image(image: Image.Image, prompt: str, api_url: str) -> str:
    """
    Envía imagen + prompt al servidor multimodal en Kaggle vía Ngrok.
    Endpoint: POST /analyze

    Args:
        image:   Imagen PIL del parche histopatológico
        prompt:  Prompt con contexto clínico + tarea de análisis
        api_url: URL pública de Ngrok
    Returns:
        Texto generado por el modelo (reporte patológico)
    """
    if not api_url:
        raise ValueError(
            "URL de Ngrok no configurada.\n"
            "Pega la URL del servidor Kaggle en el panel izquierdo."
        )

    # Codificar imagen PIL a base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    endpoint = f"{api_url.rstrip('/')}/analyze"
    payload = {
        "image_b64": image_b64,
        "prompt": prompt,
        "max_new_tokens": 512,
        "temperature": 0.7,
    }

    response = requests.post(endpoint, json=payload, timeout=INFERENCE_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if data.get('status') == 'error':
        raise RuntimeError(data.get('generated_text', 'Error desconocido en el servidor.'))

    return data.get('generated_text', 'Error: No se recibió texto generado.')


# ==============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ==============================================================================

if "image_data" not in st.session_state:
    st.session_state.image_data = None

if "image_hash" not in st.session_state:
    st.session_state.image_hash = None

if "metadata_clinica" not in st.session_state:
    st.session_state.metadata_clinica = None

if "is_benchmark_match" not in st.session_state:
    st.session_state.is_benchmark_match = False

if "embedding_vector" not in st.session_state:
    st.session_state.embedding_vector = None


# ==============================================================================
# INTERFAZ - SIDEBAR (Carga de imagen)
# ==============================================================================

st.sidebar.markdown("## 🌐 Conexión al Servidor (Kaggle)")
kaggle_api_url = st.sidebar.text_input(
    "URL de Ngrok:",
    placeholder="https://xxxxx.ngrok-free.app",
    help="Pega aquí la URL pública que te arrojó el notebook de Kaggle."
)
st.sidebar.markdown("---")

st.sidebar.markdown("## 📥 Carga de Parche Histopatológico")

uploaded_file = st.sidebar.file_uploader(
    "Selecciona archivo (PNG, JPG, BMP)",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded_file:
    # PASO B: Calcular hash MD5 de los bytes crudos
    bytes_data = uploaded_file.getvalue()
    file_hash = hashlib.md5(bytes_data).hexdigest()
    
    # Cargar imagen en memoria
    st.session_state.image_data = Image.open(BytesIO(bytes_data))
    st.session_state.image_hash = file_hash
    
    # Buscar en benchmark
    benchmark_data = load_benchmark_metadata()
    
    if file_hash in benchmark_data:
        # MATCH ENCONTRADO: Extraer metadata y embedding reales
        patch_info = benchmark_data[file_hash]
        st.session_state.metadata_clinica = patch_info.get("metadata_clinica", "")
        st.session_state.embedding_vector = patch_info.get("embedding", [])
        st.session_state.is_benchmark_match = True
        
        st.sidebar.success(
            f"🔬 **Parche Identificado**\n\n"
            f"✅ Hash encontrado en base de datos de Harbin\n"
            f"📄 Archivo: {patch_info.get('filename', 'N/A')}\n"
            f"🔐 ID: {file_hash[:16]}..."
        )
    else:
        # SIN MATCH: Usar valores por defecto
        st.session_state.metadata_clinica = "Muestra externa al dataset de referencia. Proceder con análisis de IA."
        st.session_state.embedding_vector = None
        st.session_state.is_benchmark_match = False
        
        st.sidebar.warning(
            f"⚠️ **Parche no identificado**\n\n"
            f"No coincide con base de datos de Harbin.\n"
            f"🔐 ID: {file_hash[:16]}..."
        )
    
    # Mostrar preview
    st.sidebar.markdown("### 🖼️ Vista Previa")
    st.sidebar.image(st.session_state.image_data, use_column_width=True)
    
    # Mostrar metadatos
    st.sidebar.markdown("### 📊 Información")
    col_h, col_w = st.sidebar.columns(2)
    with col_h:
        st.write(f"**Alto:** {st.session_state.image_data.size[1]}px")
    with col_w:
        st.write(f"**Ancho:** {st.session_state.image_data.size[0]}px")
    
    st.sidebar.write(f"**Modo:** {st.session_state.image_data.mode}")
    st.sidebar.write(f"**Estado:** {'✅ Base de datos' if st.session_state['is_benchmark_match'] else '🔄 Análisis genérico'}")


# ==============================================================================
# INTERFAZ - MAIN (Análisis)
# ==============================================================================

st.title("🔬 PredictPath — Análisis Histopatológico Multimodal")
st.markdown(
    "Arquitectura multimodal: **PLIP** (encoder visual) → **Capa de Proyección** → **Llama-3-8B** "
    "| Inferencia en GPU de Kaggle"
)
st.caption(f"v{APP_VERSION}")

# Bloqueo condicional: Solo mostrar botones si hay imagen
if st.session_state.image_data is None:
    st.warning(
        "⚠️ **Carga un parche histopatológico primero** en el panel izquierdo para habilitar análisis."
    )
else:
    st.markdown("---")
    st.markdown("### 📊 Análisis Disponibles")
    
    # Grid 2x2 de botones
    col1, col2 = st.columns(2)
    
    with col1:
        btn_morfo = st.button(
            "🔍 Análisis Morfológico",
            use_container_width=True,
            help="Caracterización de patrones celulares y arquitectura tisular"
        )
    
    with col2:
        btn_clinico = st.button(
            "🩺 Diagnóstico Clínico",
            use_container_width=True,
            help="Diagnóstico probable y diagnósticos diferenciales"
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        btn_struct = st.button(
            "🔬 Identificación de Estructuras",
            use_container_width=True,
            help="Identificación de células y estructuras tisulares"
        )
    
    with col4:
        btn_reporte = st.button(
            "📋 Generar Reporte",
            use_container_width=True,
            help="Reporte formal de patología"
        )
    
    # Determinar qué análisis se solicitó
    analysis_type = None
    if btn_morfo:
        analysis_type = "morfologico"
    elif btn_clinico:
        analysis_type = "clinico"
    elif btn_struct:
        analysis_type = "estructuras"
    elif btn_reporte:
        analysis_type = "reporte"
    
    # PASO C: Procesar análisis si se seleccionó uno
    if analysis_type:
        st.markdown("---")
        
        # Definir prompts según tipo de análisis
        analysis_prompts = {
            "morfologico": (
                "Analiza los patrones celulares y la arquitectura tisular visible en esta muestra histopatológica. "
                "Describe: disposición celular, características nucleares, densidad celular y hallazgos morfológicos principales."
            ),
            "clinico": (
                "Basándote en los hallazgos histopatológicos, proporciona un diagnóstico clínico probable. "
                "Incluye: diagnóstico principal, diagnósticos diferenciales, y correlación clínica recomendada."
            ),
            "estructuras": (
                "Identifica y describe las principales estructuras celulares y tipos de células presentes. "
                "Especifica: tipos celulares, organización tisular, características distintivas."
            ),
            "reporte": (
                "Genera un reporte formal de patología siguiendo este formato: "
                "RESUMEN EJECUTIVO | HALLAZGOS PRINCIPALES | INTERPRETACIÓN | CONCLUSIONES | RECOMENDACIONES"
            )
        }
        
        # Construir prompt dinámico (metadata_clinica + tarea específica)
        clinical_context = st.session_state.metadata_clinica
        task_prompt = analysis_prompts[analysis_type]
        
        full_prompt = f"""CONTEXTO CLÍNICO:
{clinical_context}

---

TAREA DE ANÁLISIS:
{task_prompt}

Proporciona una respuesta estructurada y profesional."""
        
        # Mostrar UI de carga
        st.subheader(f"🔄 Analizando ({analysis_type.capitalize()})...")

        # Indicar si la imagen es del benchmark o nueva
        if st.session_state.is_benchmark_match:
            st.info(
                "🔬 **Imagen del benchmark identificada** — "
                "El modelo recibirá la imagen directamente para análisis visual real."
            )
        else:
            st.warning(
                "⚠️ **Imagen externa al benchmark** — "
                "PLIP extraerá el embedding visual en tiempo real en Kaggle."
            )

        with st.spinner(
            "⏳ Enviando imagen a Llama-3-8B multimodal en Kaggle...\n"
            "Pipeline: imagen → PLIP → Proyector → Llama-3\n"
            "(Puede tardar 30-90 segundos)"
        ):
            try:
                # Enviar prompt al servidor Llama-3
                response_text = analyze_image(
                    image=st.session_state.image_data,
                    prompt=full_prompt,
                    api_url=kaggle_api_url
                )

                # Renderizar resultado
                st.success(f"✅ Análisis completado a las {datetime.now().strftime('%H:%M:%S')}")
                st.markdown("---")
                st.markdown(response_text)
                st.markdown("---")
                st.info(
                    "📌 **Nota:** Análisis generado por IA multimodal (visión real). "
                    "Requiere validación por patólogo certificado."
                )

            except Exception as e:
                st.error(f"❌ Error en inferencia: {str(e)}")
                st.warning(
                    "**Posibles causas:**\n"
                    "- La URL de Ngrok no está pegada en el panel izquierdo.\n"
                    "- El notebook de Kaggle (inferencia multimodal) se apagó.\n"
                    "- Ngrok cambió la URL — reinicia el notebook y copia la nueva URL."
                )
