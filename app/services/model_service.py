# app/services/model_service.py
import os
import logging

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)

# --- Lista padrão, caso class_names.txt não seja encontrado ---
DEFAULT_CLASS_NAMES = [
    "healthy",
    "bacterial_blight",
    "powdery_mildew",
    "early_blight",
    "late_blight",
    "leaf_rust",
    "septoria_leaf_spot",
    "target_spot",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
    "downy_mildew",
    "spider_mites",
]

def _load_class_names() -> list[str]:
    """
    Tenta ler model/class_names.txt; se falhar, retorna DEFAULT_CLASS_NAMES.
    """
    path = os.getenv("CLASS_NAMES_PATH", "model/class_names.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
            if names:
                logger.info("Carregadas %d classes de %s", len(names), path)
                return names
            else:
                logger.warning("Arquivo %s vazio. Usando lista padrão.", path)
    else:
        logger.warning("Arquivo de classes não encontrado em %s. Usando lista padrão.", path)

    return DEFAULT_CLASS_NAMES.copy()

# Carrega a lista de classes e monta tratamentos genéricos
CLASS_NAMES   = _load_class_names()
TREATMENT_DICT = {cls: "Consulta um especialista." for cls in CLASS_NAMES}

_model = None

def _get_model_path() -> str:
    """
    Retorna o caminho do modelo via ENV ou default.
    """
    return os.getenv("MODEL_PATH", "model/plant_disease_model.h5")

def load_model(force_reload: bool = False) -> bool:
    """
    Lazy loading do modelo. Se force_reload=True, recarrega sempre.
    Retorna True em sucesso, False em falha.
    """
    global _model
    model_path = _get_model_path()

    if not os.path.exists(model_path):
        logger.error("Modelo não encontrado em: %s", model_path)
        return False

    if _model is None or force_reload:
        try:
            logger.info("Carregando modelo de %s", model_path)
            _model = tf.keras.models.load_model(model_path, compile=False)

            # sanity check: número de saídas bate com len(CLASS_NAMES)?
            num_out = _model.output_shape[-1]
            if num_out != len(CLASS_NAMES):
                raise ValueError(
                    f"Modelo com {num_out} saídas, mas CLASS_NAMES tem {len(CLASS_NAMES)} itens."
                )

            logger.info("Modelo carregado com sucesso (%d classes).", num_out)

        except Exception:
            logger.exception("Falha ao carregar modelo")
            _model = None
            return False

    return True

def predict_disease(pil_image: Image.Image) -> dict[str, str]:
    """
    Recebe PIL.Image, pré-processa, prediz e retorna
    {"disease": <classe>, "treatment": <texto>}.
    Deve chamar load_model() antes.
    """
    if _model is None:
        raise RuntimeError("Modelo não carregado. Chame load_model() primeiro.")

    # 1. Garante RGB e redimensiona
    img = pil_image.convert("RGB").resize((224, 224))

    # 2. Converte pra array e normaliza
    arr = np.array(img, dtype="float32") / 255.0
    batch = np.expand_dims(arr, axis=0)

    # 3. Predição
    preds = _model.predict(batch)
    idx = int(np.argmax(preds[0]))

    # 4. Validação de índice
    if idx < 0 or idx >= len(CLASS_NAMES):
        msg = f"Índice inválido {idx} (0–{len(CLASS_NAMES)-1})"
        logger.error(msg)
        raise ValueError(msg)

    disease   = CLASS_NAMES[idx]
    treatment = TREATMENT_DICT.get(disease, "Consulta um especialista.")
    return {"disease": disease, "treatment": treatment}
