# PredictPath — Resumen Ejecutivo del Contexto y Arquitectura
**Fecha:** 17 de Noviembre de 2026  
**Repositorio GitHub:** [github.com/heynerdm/PatchChat](https://github.com/heynerdm/PatchChat)

---

## 1. Arquitectura de 3 Agentes (Fiel a PathAgent ECCV 2026)

1. **Navigator (PLIP — `vinid/plip`):**
   * Cobertura de tejido al 100% mediante macro-tiling jerárquico contiguo ($4096 \times 4096$ px a Nivel 0).
   * Clasifica y rankea el **Top 10% de Regiones de Interés (RoIs)** según su afinidad semántica con la consulta clínica seleccionada.
2. **Perceptor (VLM — Quilt-LLaVA / `Qwen2.5-VL-3B-Instruct`):**
   * Inspecciona los $N$ parches más críticos (por defecto 3 focos) a resolución celular máxima en **Nivel 0 ($512 \times 512$ px, $0.25\,\mu\text{m/px}$, cubriendo $128 \times 128\,\mu\text{m}$ de tejido)**.
   * Genera descripciones morfológicas estandarizadas: arquitectura tisular, atipia citológica, pleomorfismo nuclear y formación de túbulos.
3. **Executor (Razonamiento Chain-of-Thought en 3 Pasos):**
   * **Step A (PredictAnswer):** Predicción preliminar del diagnóstico e hipótesis basada en los parches observados.
   * **Step B (SelfReflect):** Autoevaluación de suficiencia diagnóstica (*Sufficient: Yes/No*).
   * **Step C (ExploreMissing / Final Decision):** Dictamen patológico integral estructurado.

---

## 2. Entorno y Despliegue

* **Servidor Backend (GPU T4 en Kaggle):** [`Scripts/kaggle_notebook_completo.py`](file:///home/heynerr/Documentos/PredictPath/Proyecto/Scripts/kaggle_notebook_completo.py)
  * Expone API REST FastAPI vía Ngrok.
  * Cuenta con sistema de caché en disco (`/kaggle/working/wsi_cache/`) tanto para vectores PLIP como para análisis clínicos de las 5 consultas fijas, reduciendo la respuesta a < 0.2 segundos.
* **Interfaz de Diagnóstico (Streamlit Cockpit):** [`app_wsi_futuristic.py`](file:///home/heynerr/Documentos/PredictPath/Proyecto/app_wsi_futuristic.py)
  * Corriendo localmente en `http://localhost:8501`.
  * Catálogo de 5 consultas clínicas diagnósticas fijas.
  * Visor multiescala interactivo (5x, 10x, 20x, 40x).
  * Exportación de informe médico formal imprimible directamente a PDF.
  * Estado de sesión blindado en `st.session_state` sin pérdida de imagen de la lámina.
