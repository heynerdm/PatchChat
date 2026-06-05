PredictPath — Sistema de Clasificación Histopatológica Multimodal
PredictPath es un sistema de análisis histopatológico asistido por IA que combina tres componentes en una arquitectura multimodal:

PLIP (encoder visual) — extrae embeddings de 768 dimensiones de parches histopatológicos.
Capa de Proyección (~6.4M parámetros) — un MLP entrenado que traduce los embeddings visuales de PLIP al espacio latente de 4096 dimensiones de Llama-3.
Llama-3-8B (LLM) — genera reportes patológicos estructurados (análisis morfológico, diagnósticos clínicos, identificación de estructuras) a partir de los tokens visuales proyectados.
El entrenamiento de la capa de proyección se realiza en Kaggle (GPU T4) usando los datasets ARCH y GCHTID de cáncer gástrico. La inferencia se despliega como un servidor FastAPI expuesto vía Ngrok, consumido por una interfaz Streamlit que permite cargar parches, verificarlos contra una base de datos de referencia (Harbin), y solicitar análisis especializados en tiempo real.
