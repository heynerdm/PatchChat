#!/usr/bin/env python3
"""
==============================================================================
NOTEBOOK KAGGLE - SERVIDOR DE INFERENCIA MULTIMODAL
==============================================================================
Carga PLIP + Proyector + Llama-3 y expone API FastAPI via Ngrok.

INPUT (agregar como Kaggle Dataset):
  - projection_layer_best.pt  → Pesos entrenados del proyector
  - HeynerMarqu/pathology_llama3_completo → Desde HuggingFace

ENDPOINTS:
  POST /analyze   → Recibe imagen base64, devuelve análisis patológico
  GET  /health    → Verificación de estado del servidor
==============================================================================
"""

# ===================== CELDA 1: Instalación =====================
# !pip install fastapi uvicorn pyngrok transformers accelerate bitsandbytes open_clip_torch nest-asyncio -q

# ===================== CELDA 2: Imports =====================
import os
import json
import base64
import torch
import torch.nn as nn
import nest_asyncio
import uvicorn
import threading

from io import BytesIO
from PIL import Image
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CLIPModel,
    CLIPProcessor
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok
from huggingface_hub import hf_hub_download

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
HF_TOKEN  = os.environ.get("HF_TOKEN", "")
NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")   # Agregar como Kaggle Secret
LLAMA_ID  = "HeynerMarqu/pathology_llama3_completo"
PROJECTOR_PATH = "/kaggle/input/projection-layer/projection_layer_best.pt"

print(f"Dispositivo: {DEVICE}")
print(f"GPUs: {torch.cuda.device_count()}")


# ===================== CELDA 3: Arquitectura del Proyector =====================
class PLIPToLlamaProjector(nn.Module):
    """Debe ser idéntica a la usada en entrenamiento."""
    def __init__(self, plip_dim: int = 768, llama_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(plip_dim, llama_dim),
            nn.GELU(),
            nn.LayerNorm(llama_dim),
            nn.Linear(llama_dim, llama_dim),
        )

    def forward(self, plip_embedding: torch.Tensor) -> torch.Tensor:
        projected = self.projector(plip_embedding)
        return projected.unsqueeze(1)  # [B, 1, 4096]


# ===================== CELDA 4: Cargar PLIP =====================
print("Cargando PLIP...")
plip = CLIPModel.from_pretrained("vinid/plip", token=HF_TOKEN)
plip_processor = CLIPProcessor.from_pretrained("vinid/plip", token=HF_TOKEN)
plip = plip.to(DEVICE)
plip.eval()
for p in plip.parameters():
    p.requires_grad = False
print("✅ PLIP listo")


# ===================== CELDA 5: Cargar Proyector =====================
print("Cargando capa de proyección entrenada...")
checkpoint = torch.load(PROJECTOR_PATH, map_location=DEVICE)
config = checkpoint.get('config', {'plip_dim': 768, 'llama_dim': 4096})

projector = PLIPToLlamaProjector(**config).to(DEVICE)
projector.load_state_dict(checkpoint['projector_state_dict'])
projector.eval()
for p in projector.parameters():
    p.requires_grad = False

print(f"✅ Proyector cargado (entrenado en época {checkpoint.get('epoch','?')}, val_loss={checkpoint.get('val_loss',0):.4f})")


# ===================== CELDA 6: Cargar Llama-3 =====================
print("Cargando Llama-3-8B en 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(LLAMA_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    LLAMA_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
llama.eval()
for p in llama.parameters():
    p.requires_grad = False

print("✅ Llama-3 listo")


# ===================== CELDA 7: Función de Inferencia Multimodal =====================
@torch.no_grad()
def get_plip_embedding(image: Image.Image) -> torch.Tensor:
    """Imagen → embedding PLIP 768-dim."""
    inputs = plip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    features = plip.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    return features  # [1, 768]


@torch.no_grad()
def multimodal_generate(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    Pipeline completo de inferencia multimodal:
    imagen + prompt → análisis patológico generado

    Flujo:
    1. PLIP:      imagen → embedding [1, 768]
    2. Proyector: [1, 768] → visual_token [1, 1, 4096]
    3. Tokenizar: prompt → input_ids
    4. Llama-3:   embed_tokens(input_ids) → text_embeddings [1, seq, 4096]
    5. Concat:    [visual_token || text_embeddings] → [1, 1+seq, 4096]
    6. Generar:   Llama-3.generate(inputs_embeds=combined)
    """

    # 1. Embedding visual
    plip_emb = get_plip_embedding(image)                          # [1, 768]

    # 2. Visual token (proyectado al espacio de Llama-3)
    visual_token = projector(plip_emb.to(torch.float32))          # [1, 1, 4096]

    # 3. Tokenizar prompt del sistema
    system_prompt = f"""Eres un patólogo especialista en histopatología digital.
Analiza esta imagen histopatológica y responde de forma estructurada y precisa.

{prompt}"""

    input_ids = tokenizer(
        system_prompt,
        return_tensors='pt',
        truncation=True,
        max_length=256,
    ).input_ids.to(llama.device)

    # 4. Embeddings de texto
    text_embeddings = llama.model.embed_tokens(input_ids)          # [1, seq, 4096]
    text_embeddings = text_embeddings.to(torch.float32)

    # 5. Concatenar visual token + embeddings de texto
    visual_token = visual_token.to(llama.device)
    combined = torch.cat([visual_token, text_embeddings], dim=1)   # [1, 1+seq, 4096]

    # 6. Generar respuesta
    attention_mask = torch.ones(1, combined.shape[1], device=llama.device)

    output_ids = llama.generate(
        inputs_embeds=combined,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decodificar solo los tokens nuevos generados
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Limpiar artefactos del formato SlideChat
    generated_text = generated_text.split("ASSISTANT:")[-1]
    generated_text = generated_text.replace("<image>", "").strip()

    return generated_text


# ===================== CELDA 8: FastAPI App =====================
app = FastAPI(title="PredictPath Multimodal API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    image_b64: str        # Imagen codificada en base64
    prompt: str           # Tipo de análisis solicitado
    max_new_tokens: int = 512
    temperature: float = 0.7


class AnalyzeResponse(BaseModel):
    generated_text: str
    status: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": ["PLIP", "PLIPToLlamaProjector", "Llama-3-8B"],
        "device": DEVICE,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    try:
        # Decodificar imagen base64
        image_bytes = base64.b64decode(request.image_b64)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Inferencia multimodal
        result = multimodal_generate(
            image=image,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        return AnalyzeResponse(generated_text=result, status="success")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return AnalyzeResponse(
            generated_text=f"Error en inferencia: {str(e)}",
            status="error"
        )


# ===================== CELDA 9: Lanzar Ngrok + Servidor =====================
print("Configurando Ngrok...")
ngrok.set_auth_token(NGROK_TOKEN)

# Abrir túnel público
public_url = ngrok.connect(8000)
print(f"\n{'='*60}")
print(f"🚀 API Multimodal disponible en:")
print(f"   {public_url}")
print(f"\nEndpoints:")
print(f"   GET  {public_url}/health   → Estado del servidor")
print(f"   POST {public_url}/analyze  → Análisis de imagen")
print(f"{'='*60}")
print(f"\nCopia la URL de arriba y pégala en Streamlit.")

# Lanzar servidor en hilo secundario
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
print("✅ Servidor en ejecución")
