#!/usr/bin/env python3
"""
==============================================================================
KAGGLE NOTEBOOK - PUENTE DE PRUEBA Y TELEMETRÍA (FASTAPI + NGROK)
==============================================================================
Script ligero para verificar la conectividad bidireccional entre Kaggle y la
interfaz local (Streamlit) antes de cargar modelos pesados.

INSTRUCCIONES EN KAGGLE:
  1. Crear un Notebook nuevo en Kaggle (Acelerador: GPU T4 x1 o CPU).
  2. Agregar tu secreto NGROK_TOKEN (Add-ons -> Secrets -> NGROK_TOKEN).
  3. Ejecutar las celdas en orden.
  4. Copiar la URL pública generada (ej: https://xxxx.ngrok-free.app).
  5. Pegarla en la barra lateral de Streamlit.
==============================================================================
"""

# ===================== CELDA 1: Instalación de dependencias =====================
# En Kaggle descomentar y ejecutar:
# !pip install fastapi uvicorn pyngrok nest-asyncio -q

# ===================== CELDA 2: Servidor FastAPI de Verificación =====================
import os
import time
import torch
import uvicorn
import nest_asyncio
import threading
from datetime import datetime
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok

# Permitir bucle de eventos asíncrono anidado en notebooks
nest_asyncio.apply()

app = FastAPI(title="PredictPath Telemetry Bridge", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Telemetría de hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
VRAM_TOTAL = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0

START_TIME = time.time()

class PingRequest(BaseModel):
    client_timestamp: float
    message: str = "ping"

@app.get("/")
def root():
    return {
        "service": "PredictPath Neural Bridge",
        "status": "online",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    uptime_sec = round(time.time() - START_TIME, 1)
    vram_used = round(torch.cuda.memory_allocated(0) / (1024**3), 2) if torch.cuda.is_available() else 0
    return {
        "status": "online",
        "engine": "PathAgent-Kaggle-Bridge",
        "device": DEVICE,
        "gpu_name": GPU_NAME,
        "vram_total_gb": VRAM_TOTAL,
        "vram_used_gb": vram_used,
        "uptime_seconds": uptime_sec,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ping")
def ping(req: PingRequest):
    now = time.time()
    latency_ms = round((now - req.client_timestamp) * 1000, 2) if req.client_timestamp > 0 else 0
    return {
        "status": "pong",
        "client_message": req.message,
        "latency_one_way_ms": latency_ms,
        "server_timestamp": now
    }

# ===================== CELDA 3: Túnel Ngrok y Lanzamiento =====================
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

if not NGROK_TOKEN:
    # Intenta leer desde secrets de Kaggle si está disponible
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        NGROK_TOKEN = user_secrets.get_secret("NGROK_TOKEN")
    except Exception:
        pass

if NGROK_TOKEN:
    ngrok.set_auth_token(NGROK_TOKEN)
else:
    print("⚠️ ADVERTENCIA: NGROK_TOKEN no encontrado. Configúralo en Kaggle Secrets o pégalo directamente.")

# Cerrar túneles previos que hayan quedado abiertos
try:
    ngrok.kill()
except Exception:
    pass

# Abrir túnel público al puerto 8000
public_url = ngrok.connect(8000)

print("=" * 70)
print("🚀 PUENTE DE PRUEBA PREDICTPATH INICIADO CON ÉXITO")
print("=" * 70)
print(f"📡 URL PÚBLICA DE NGROK:")
print(f"   👉  {public_url}  👈")
print("-" * 70)
print("Endpoints disponibles:")
print(f"  • GET  {public_url}/health   → Estado y telemetría de hardware")
print(f"  • POST {public_url}/ping     → Medición de latencia bidireccional")
print("=" * 70)
print("Pega la URL anterior en el campo 'Túnel Ngrok' de tu Streamlit local.")

# Iniciar servidor Uvicorn en segundo plano
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
print("✅ Servidor Uvicorn corriendo en puerto 8000.")
