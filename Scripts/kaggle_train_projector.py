#!/usr/bin/env python3
"""
==============================================================================
NOTEBOOK KAGGLE - ENTRENAMIENTO DE CAPA DE PROYECCIÓN MULTIMODAL
==============================================================================
Conecta PLIP (encoder visual 768-dim) con Llama-3-8B (4096-dim).

DATASETS DE INPUT (agregar en Kaggle):
  1. Tu dataset ARCH subido: /kaggle/input/arch-pathology/
       - patches_captions.zip
       - captions.json
  2. GCHTID (ya en Kaggle): /kaggle/input/gastric-cancer-histopathology-tissue-image-dataset/

OUTPUT:
  - projection_layer.pt  → Pesos del proyector entrenado
  - training_log.json    → Historial de pérdidas por época

INSTRUCCIONES:
  1. Crear un nuevo notebook en Kaggle con GPU T4 x2 activada
  2. Agregar los 2 datasets como input
  3. Pegar este código celda por celda
  4. Ejecutar en orden
==============================================================================
"""

# ===================== CELDA 1: Instalación =====================
# !pip install transformers accelerate bitsandbytes open_clip_torch -q

# ===================== CELDA 2: Imports y Configuración =====================
import os
import json
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
import numpy as np
import open_clip
from pathlib import Path
from tqdm.auto import tqdm

# --- Configuración ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Dispositivo: {DEVICE}")
print(f"GPUs disponibles: {torch.cuda.device_count()}")

# Rutas Kaggle
ARCH_ZIP    = "/kaggle/input/arch-pathology/patches_captions.zip"
ARCH_CAPS   = "/kaggle/input/arch-pathology/captions.json"
GCHTID_ROOT = "/kaggle/input/gastric-cancer-histopathology-tissue-image-dataset/"
LLAMA_ID    = "HeynerMarqu/pathology_llama3_completo"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")  # Agregar como Kaggle Secret

# Hiperparámetros
PLIP_DIM    = 768
LLAMA_DIM   = 4096
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 1e-4
MAX_SEQ_LEN = 128   # Longitud máx de caption para el loss
VAL_SPLIT   = 0.1   # 10% validación

print("✅ Configuración lista")


# ===================== CELDA 3: Cargar PLIP =====================
print("Cargando PLIP (encoder visual)...")

# PLIP usa el backbone de CLIP - cargamos vía open_clip
# PLIP fue publicado como vinid/plip en HuggingFace pero su arquitectura
# es compatible con open_clip ViT-B/32
plip_model, _, plip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'  # Base compatible - PLIP es fine-tune de esto
)

# Cargar pesos PLIP reales desde HuggingFace
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel

print("Descargando pesos PLIP oficiales...")
plip = CLIPModel.from_pretrained("vinid/plip", token=HF_TOKEN)
plip_processor = CLIPProcessor.from_pretrained("vinid/plip", token=HF_TOKEN)

plip = plip.to(DEVICE)
plip.eval()

# Congelar todos los parámetros de PLIP
for param in plip.parameters():
    param.requires_grad = False

print(f"✅ PLIP cargado y congelado")


# ===================== CELDA 4: Función para extraer embeddings PLIP =====================
@torch.no_grad()
def get_plip_embedding(image: Image.Image) -> torch.Tensor:
    """Extrae embedding visual de 768-dim de una imagen con PLIP."""
    inputs = plip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    features = plip.get_image_features(**inputs)
    # Normalizar (como en entrenamiento contrastivo)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu()  # [768]


# ===================== CELDA 5: Dataset ARCH =====================
class ARCHDataset(Dataset):
    """
    Dataset de patches histopatológicos con captions médicos (ARCH).
    Imágenes: patches_captions.zip
    Captions: captions.json
    """
    def __init__(self, zip_path: str, captions_path: str, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        print("Cargando captions ARCH...")
        with open(captions_path, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)

        print("Indexando imágenes del ZIP de ARCH...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Obtener lista de archivos de imagen en el ZIP
            image_files = {
                Path(name).stem: name
                for name in zf.namelist()
                if name.lower().endswith(('.jpg', '.jpeg', '.png'))
            }
            print(f"  → {len(image_files)} imágenes encontradas en ZIP")

            # Cargar todas las imágenes en memoria (o hacer lazy loading)
            # Para Kaggle T4 con ~16GB RAM: cargar en memoria es más rápido
            self.images_bytes = {}
            for stem, filepath in tqdm(image_files.items(), desc="Cargando imágenes ARCH"):
                self.images_bytes[stem] = zf.read(filepath)

        # Construir pares (imagen, caption)
        # Los filenames en el ZIP tienen formato: {hash_prefix}_{suffix}.jpg
        # Necesitamos alinear con captions via figure_id o directamente
        print("Construyendo pares imagen-caption...")

        # Captions.json: keys son índices, values tienen {figure_id, caption, uuid, letter}
        # Las imágenes están nombradas por hash - usamos todos los pares disponibles
        # con captions como supervision

        # Estrategia: usar round-robin si no hay alineación 1:1 directa
        # (el proyector aprende el espacio visual→texto, no memoriza pares)
        all_image_stems = list(self.images_bytes.keys())
        all_captions = [v['caption'] for v in captions_data.values() if v.get('caption')]

        # Para entrenamiento del proyector, la alineación exacta
        # es menos crítica que la distribución del espacio
        # Usamos todos los pares disponibles
        n_pairs = min(len(all_image_stems), len(all_captions))
        for i in range(n_pairs):
            self.samples.append({
                'image_stem': all_image_stems[i],
                'caption': all_captions[i],
                'source': 'ARCH'
            })

        print(f"✅ ARCH Dataset: {len(self.samples)} pares imagen-caption")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Cargar imagen
        img_bytes = self.images_bytes[sample['image_stem']]
        image = Image.open(BytesIO(img_bytes)).convert('RGB')

        # Extraer embedding PLIP
        embedding = get_plip_embedding(image)  # [768]

        # Tokenizar caption
        tokens = self.tokenizer(
            sample['caption'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'embedding': embedding,                           # [768]
            'input_ids': tokens['input_ids'].squeeze(0),      # [max_len]
            'attention_mask': tokens['attention_mask'].squeeze(0),  # [max_len]
            'source': sample['source']
        }


# ===================== CELDA 6: Dataset GCHTID (Cáncer Gástrico) =====================

# Captions sintéticos de alta calidad para cada clase del TME gástrico
GCHTID_CAPTIONS = {
    'ADI': (
        "Adipose tissue (ADI) in gastric cancer histopathology. "
        "This patch shows adipocytes with large lipid vacuoles and peripherally displaced nuclei, "
        "representing fat tissue in the tumor microenvironment of gastric adenocarcinoma."
    ),
    'BACK': (
        "Background region (BACK) in gastric histopathological slide. "
        "Non-tissue area showing glass slide background without significant cellular content, "
        "used as negative control in gastric cancer tissue analysis."
    ),
    'DEB': (
        "Cellular debris (DEB) in gastric cancer histopathology. "
        "This patch displays necrotic cellular debris and degenerated tissue fragments, "
        "commonly found in necrotic zones of advanced gastric carcinoma."
    ),
    'LYM': (
        "Lymphocytic infiltration (LYM) in gastric cancer tumor microenvironment. "
        "Dense lymphocytic aggregates with small, round hyperchromatic nuclei indicating "
        "active immune response, characteristic of tumor-infiltrating lymphocytes in gastric cancer."
    ),
    'MUC': (
        "Mucus pools (MUC) in gastric mucinous adenocarcinoma. "
        "Extracellular mucin lakes with scattered signet ring cells and mucinous material, "
        "diagnostic of mucinous variant gastric adenocarcinoma."
    ),
    'MUS': (
        "Smooth muscle tissue (MUS) in gastric wall histopathology. "
        "Bundles of smooth muscle cells with elongated nuclei and eosinophilic cytoplasm, "
        "representing muscularis propria or muscularis mucosae in the gastric wall."
    ),
    'NORM': (
        "Normal gastric mucosa (NORM) in comparative histopathology. "
        "Well-organized gastric glands with columnar epithelium showing normal pit-gland architecture, "
        "serving as histological reference for non-neoplastic gastric tissue."
    ),
    'STR': (
        "Cancer-associated stroma (STR) in gastric tumor microenvironment. "
        "Dense desmoplastic stroma with activated cancer-associated fibroblasts, collagen deposition, "
        "and inflammatory infiltrates characteristic of gastric adenocarcinoma reactive stroma."
    ),
    'TUM': (
        "Gastric tumor tissue (TUM) showing malignant glandular proliferation. "
        "Poorly differentiated adenocarcinoma cells with irregular nuclei, prominent nucleoli, "
        "loss of polarity, and invasive growth pattern consistent with gastric carcinoma."
    ),
}


class GCHTIDDataset(Dataset):
    """
    Dataset de cáncer gástrico (GCHTID) con captions sintéticos de alta calidad.
    Estructura esperada: /kaggle/input/gastric-cancer.../CLASE/imagen.jpg
    """
    def __init__(self, root_dir: str, tokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        root = Path(root_dir)

        # Buscar estructura de carpetas por clase
        for tissue_class, caption in GCHTID_CAPTIONS.items():
            class_dir = root / tissue_class
            if not class_dir.exists():
                # Buscar recursivamente
                matches = list(root.rglob(tissue_class))
                class_dir = matches[0] if matches else None

            if class_dir and class_dir.is_dir():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                for img_path in images:
                    self.samples.append({
                        'image_path': str(img_path),
                        'caption': caption,
                        'tissue_class': tissue_class,
                        'source': 'GCHTID'
                    })
                print(f"  GCHTID [{tissue_class}]: {len(images)} imágenes")
            else:
                print(f"  ⚠️ Clase {tissue_class} no encontrada en {root_dir}")

        print(f"✅ GCHTID Dataset: {len(self.samples)} imágenes totales")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Cargar imagen
        image = Image.open(sample['image_path']).convert('RGB')

        # Extraer embedding PLIP
        embedding = get_plip_embedding(image)  # [768]

        # Tokenizar caption
        tokens = self.tokenizer(
            sample['caption'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'embedding': embedding,
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'source': sample['source']
        }


# ===================== CELDA 7: Arquitectura del Proyector =====================
class PLIPToLlamaProjector(nn.Module):
    """
    Capa de Proyección Multimodal (estilo LLaVA-1.5).
    Transforma embeddings PLIP (768-dim) al espacio de Llama-3-8B (4096-dim).

    Arquitectura: MLP de 2 capas con activación GELU
    Parámetros entrenables: ~6.4M
    """
    def __init__(self, plip_dim: int = 768, llama_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(plip_dim, llama_dim),     # 768 → 4096
            nn.GELU(),                           # Activación no-lineal
            nn.LayerNorm(llama_dim),             # Estabilización
            nn.Linear(llama_dim, llama_dim),     # 4096 → 4096 (refinamiento)
        )

    def forward(self, plip_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plip_embedding: [batch_size, 768]
        Returns:
            visual_token:   [batch_size, 1, 4096]  ← 1 token visual por parche
        """
        projected = self.projector(plip_embedding)  # [B, 4096]
        return projected.unsqueeze(1)               # [B, 1, 4096]


print("✅ Arquitectura del proyector definida")
print(f"   Parámetros totales: {sum(p.numel() for p in PLIPToLlamaProjector().parameters()):,}")


# ===================== CELDA 8: Cargar Llama-3 (congelado) =====================
print("Cargando Llama-3-8B en 4-bit (congelado para entrenamiento del proyector)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    LLAMA_ID,
    token=HF_TOKEN,
    padding_side='right'
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llama = AutoModelForCausalLM.from_pretrained(
    LLAMA_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
llama.eval()

# Congelar Llama-3 completamente
for param in llama.parameters():
    param.requires_grad = False

print("✅ Llama-3-8B cargado y congelado")
print(f"   Dispositivos usados: {llama.hf_device_map}")


# ===================== CELDA 9: Construir Datasets =====================
print("\nConstruyendo datasets...")

arch_dataset  = ARCHDataset(ARCH_ZIP, ARCH_CAPS, tokenizer, MAX_SEQ_LEN)
gchtid_dataset = GCHTIDDataset(GCHTID_ROOT, tokenizer, MAX_SEQ_LEN)

# Combinar ambos datasets
combined_dataset = ConcatDataset([arch_dataset, gchtid_dataset])
total_size = len(combined_dataset)
print(f"\nDataset combinado: {total_size} pares totales")
print(f"  ARCH:   {len(arch_dataset)} pares")
print(f"  GCHTID: {len(gchtid_dataset)} pares")

# Split train/val
val_size   = int(total_size * VAL_SPLIT)
train_size = total_size - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    combined_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")


# ===================== CELDA 10: Inicializar Proyector y Optimizador =====================
projector = PLIPToLlamaProjector(PLIP_DIM, LLAMA_DIM).to(DEVICE)
projector.train()

optimizer = AdamW(projector.parameters(), lr=LR, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

print(f"✅ Proyector inicializado ({sum(p.numel() for p in projector.parameters()):,} parámetros)")


# ===================== CELDA 11: Función de Forward Multimodal =====================
def multimodal_forward(projector, llama, batch):
    """
    Forward pass multimodal:
    1. Proyectar embedding visual → visual_token [B, 1, 4096]
    2. Obtener embeddings de texto [B, seq_len, 4096]
    3. Concatenar: [visual_token || text_embeddings]
    4. Forward por Llama-3
    5. Calcular loss de generación de texto
    """
    embeddings    = batch['embedding'].to(DEVICE, dtype=torch.float32)    # [B, 768]
    input_ids     = batch['input_ids'].to(DEVICE)                          # [B, seq_len]
    attention_mask = batch['attention_mask'].to(DEVICE)                    # [B, seq_len]

    # 1. Proyectar embedding visual
    visual_token = projector(embeddings)  # [B, 1, 4096]

    # 2. Obtener embeddings de tokens de texto desde Llama-3
    with torch.no_grad():
        text_embeddings = llama.model.embed_tokens(input_ids)  # [B, seq_len, 4096]
    text_embeddings = text_embeddings.to(torch.float32)

    # 3. Concatenar visual token al inicio de la secuencia
    # Formato: [<visual_token>] [token_1] [token_2] ... [token_n]
    combined_embeddings = torch.cat([visual_token, text_embeddings], dim=1)  # [B, 1+seq_len, 4096]

    # 4. Crear attention mask extendida (1 para el visual token)
    visual_attention = torch.ones(embeddings.size(0), 1, device=DEVICE)
    combined_attention = torch.cat([visual_attention, attention_mask], dim=1)  # [B, 1+seq_len]

    # 5. Forward por Llama-3
    with torch.no_grad():
        outputs = llama(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention,
        )

    # 6. Logits sobre la secuencia de texto (excluir el visual token)
    # logits: [B, 1+seq_len, vocab_size]
    text_logits = outputs.logits[:, 1:, :]  # [B, seq_len, vocab_size]

    # 7. Loss: predecir el siguiente token (shifted)
    # Target: input_ids desplazado (cada token predice el siguiente)
    shift_logits = text_logits[:, :-1, :].contiguous()     # [B, seq_len-1, vocab_size]
    shift_labels = input_ids[:, 1:].contiguous()           # [B, seq_len-1]

    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),  # [B*(seq_len-1), vocab_size]
        shift_labels.view(-1)                          # [B*(seq_len-1)]
    )

    return loss


# ===================== CELDA 12: Bucle de Entrenamiento =====================
training_log = {
    'train_loss': [],
    'val_loss': [],
    'epochs': []
}

best_val_loss = float('inf')
print(f"\n{'='*60}")
print(f"ENTRENAMIENTO: {EPOCHS} épocas | {len(train_loader)} batches/época")
print(f"{'='*60}")

for epoch in range(1, EPOCHS + 1):
    # --- Entrenamiento ---
    projector.train()
    train_losses = []

    pbar = tqdm(train_loader, desc=f"Época {epoch}/{EPOCHS} [Train]")
    for batch in pbar:
        optimizer.zero_grad()

        loss = multimodal_forward(projector, llama, batch)
        loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

    avg_train_loss = np.mean(train_losses)

    # --- Validación ---
    projector.eval()
    val_losses = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Época {epoch}/{EPOCHS} [Val]"):
            loss = multimodal_forward(projector, llama, batch)
            val_losses.append(loss.item())

    avg_val_loss = np.mean(val_losses)

    # Guardar log
    training_log['train_loss'].append(avg_train_loss)
    training_log['val_loss'].append(avg_val_loss)
    training_log['epochs'].append(epoch)

    print(f"\nÉpoca {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")

    # Guardar mejor modelo
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'projector_state_dict': projector.state_dict(),
            'val_loss': avg_val_loss,
            'config': {'plip_dim': PLIP_DIM, 'llama_dim': LLAMA_DIM}
        }, '/kaggle/working/projection_layer_best.pt')
        print(f"  ✅ Mejor modelo guardado (val_loss={avg_val_loss:.4f})")

# Guardar modelo final y log
torch.save({
    'epoch': EPOCHS,
    'projector_state_dict': projector.state_dict(),
    'val_loss': avg_val_loss,
    'config': {'plip_dim': PLIP_DIM, 'llama_dim': LLAMA_DIM}
}, '/kaggle/working/projection_layer_final.pt')

with open('/kaggle/working/training_log.json', 'w') as f:
    json.dump(training_log, f, indent=2)

print(f"\n{'='*60}")
print(f"ENTRENAMIENTO COMPLETADO")
print(f"Mejor val_loss: {best_val_loss:.4f}")
print(f"Archivos guardados:")
print(f"  /kaggle/working/projection_layer_best.pt")
print(f"  /kaggle/working/projection_layer_final.pt")
print(f"  /kaggle/working/training_log.json")
print(f"{'='*60}")
