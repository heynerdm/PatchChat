# PredictPath — Plataforma Agéntica de Telepatología WSI

**PredictPath** es una suite de diagnóstico histopatológico asistido por IA para imágenes de portaobjetos completos (Whole Slide Images - WSI en formato `.svs` de gigapíxeles). Diseñado bajo la arquitectura agéntica de **PathAgent (ECCV 2026)** y ejecutado en un entorno híbrido cliente-servidor (GitHub Codespaces + Kaggle GPU T4).

---

## Arquitectura del Sistema

```
                  ┌────────────────────────────────────────┐
                  │          KAGGLE GPU T4 BACKEND         │
                  │  (FastAPI + OpenSlide + PLIP + Qwen)   │
                  └───────────────────▲────────────────────┘
                                      │  (Túnel Ngrok HTTPS)
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       GITHUB CODESPACES / LOCAL        │
                  │   Streamlit Cockpit (Cyber-Medical)    │
                  │        app_wsi_futuristic.py           │
                  └────────────────────────────────────────┘
```

1. **Navigator (PLIP - `vinid/plip`):**
   - Tiling jerárquico contiguo sobre la lámina completa ($4096 \times 4096$ px a Nivel 2).
   - Cobertura de tejido del 100% (cero puntos ciegos).
   - Clasificación semántica de similitud coseno texto-imagen para aislar el Top 10% de Regiones de Interés (RoIs).

2. **Perceptor (Qwen2.5-VL-3B FP16):**
   - Inspección visual microscópica de parches a Nivel 0 (resolución nuclear nativa).
   - Descripción morfológica automatizada (atipia celular, pleomorfismo, arquitectura estromal).

3. **Executor (Qwen2.5-VL-3B Chain-of-Thought):**
   - Razonamiento multi-paso en 3 fases:
     - `PredictAnswer`: Diagnóstico preliminar basado en la evidencia microscópica.
     - `SelfReflect`: Verificación de suficiencia diagnóstica.
     - `ExploreMissing`: Recomendación de zoom adaptativo si se requiere mayor magnificación.

4. **Tiling & Segmentación de Tejido:**
   - Segmentación binaria mediante umbralización Otsu sobre el canal de saturación HSV.
   - Generación de mosaicos de tiles de tejido viable para auditoría patológica.

---

## Ejecución en GitHub Codespaces

### 1. Iniciar el Codespace
Haz clic en **Code > Codespaces > Create codespace on main**.
El entorno `.devcontainer` instalará automáticamente las dependencias de `requirements.txt`.

### 2. Lanzar la Interfaz Streamlit
En la terminal del Codespace:
```bash
streamlit run app_wsi_futuristic.py
```
Codespaces abrirá automáticamente el puerto **8501** en una pestaña del navegador.

### 3. Conectar al Backend en Kaggle
1. Abre un notebook en **Kaggle** con acelerador **GPU T4 x1** e Internet activado.
2. Copia y ejecuta el script [Scripts/kaggle_notebook_completo.py](Scripts/kaggle_notebook_completo.py).
3. Copia la URL pública generada por Ngrok (ej. `https://xxxx-xx-xx-xxx-xxx.ngrok-free.app`).
4. Pega la URL en la barra lateral del Streamlit Cockpit en el campo **Túnel Ngrok (Kaggle T4)** y pulsa **Probar Conexión**.

---

## Estructura del Proyecto

```
├── app_wsi_futuristic.py         # Cockpit Streamlit (Interfaz de diagnóstico)
├── requirements.txt              # Dependencias para Codespaces / Local
├── .devcontainer/                # Configuración automática para GitHub Codespaces
├── Scripts/
│   ├── kaggle_notebook_completo.py # Notebook todo-en-uno para Kaggle (GPU T4)
│   ├── kaggle_wsi_server.py      # Servidor FastAPI para WSI
│   └── kaggle_test_bridge.py     # Script de diagnóstico de endpoints
└── README.md
```
