import re

from src.neural_handwriting import ALPHABET
from src.web.models import Writing

NEURAL_CHARACTERS = ALPHABET[1:]
MODEL_INFO = {
    "font": {"label": "Fonte manuscrita", "kind": "Renderizador local", "writers": 0, "bias": None,
             "description": "Fonte manuscrita com variação controlada. Números usam caneta e textos usam grafite.",
             "charset": "Letras com acentos, números e pontuação comum", "version": "Fonte Caveat incluída no OCRSD"},
    "sjvasquez": {"label": "Síntese neural · 13 estilos", "kind": "Rede neural recorrente", "writers": 13, "bias": 0.75,
                  "description": "Modelo de síntese de escrita com 13 estilos de escritor pré-gravados.",
                  "charset": "Sem acentos, sem Q, X e Z maiúsculos e sem barra", "version": "sjvasquez/handwriting-synthesis"},
    "aaron1259": {"label": "Síntese neural · 512 escritores", "kind": "Rede neural recorrente", "writers": 512, "bias": 2.1,
                  "description": "Modelo neural com 512 identidades de escritor e controle de regularidade.",
                  "charset": "Sem acentos, sem Q, X e Z maiúsculos e sem barra", "version": "aaron1259/handwriting-synthesis"},
    "wpvnteam": {"label": "Síntese neural · variante wpvnteam", "kind": "Rede neural recorrente", "writers": 512, "bias": 2.1,
                 "description": "Variante publicada do modelo de 512 escritores, com os mesmos pesos verificados.",
                 "charset": "Sem acentos, sem Q, X e Z maiúsculos e sem barra", "version": "wpvnteam/handwriting"},
}
TECHNICAL_NAME = re.compile(r"^(mixed_)?[a-z0-9]+_.*(estilo|reg-|esp-)")


def model_defaults(model: str, current: dict | None = None) -> dict:
    defaults = Writing(model=model).model_dump()
    if MODEL_INFO[model]["bias"] is not None:
        defaults["bias"] = MODEL_INFO[model]["bias"]
    if current:
        defaults.update(handwriting=current.get("handwriting", True), text_handwriting=current.get("text_handwriting", True))
    return defaults


def adapt_to_model(writing: dict, model: str, previous: str) -> dict:
    adapted = {**writing, "model": model}
    limit = MODEL_INFO[model]["writers"]
    if adapted.get("writer") is not None and (not limit or not isinstance(adapted["writer"], int) or adapted["writer"] >= limit):
        adapted["writer"] = None
    if model != previous and MODEL_INFO[model]["bias"] is not None:
        adapted["bias"] = MODEL_INFO[model]["bias"]
    return adapted


def preset_title(name: str, writing: dict | None) -> str:
    if not TECHNICAL_NAME.match(name):
        return name
    if not writing:
        return "Preset misto do desktop"
    writer = "estilo aleatório" if writing.get("writer") is None else f"estilo {writing['writer']}"
    return f"{MODEL_INFO[writing['model']]['label'].split(' · ')[0]} · {writing['model']} · {writer}"


def writing_summary(writing: dict | None) -> str:
    if not writing:
        return "Combinação de modelos disponível somente no desktop"
    info = MODEL_INFO.get(writing["model"], MODEL_INFO["font"])
    if writing["model"] == "font":
        return f"{info['label']} · variação {round(writing['intensity'] * 100)}%"
    writer = "escritor aleatório" if writing.get("writer") is None else f"escritor {writing['writer']}"
    return f"{writing['model']} · {writer} · {writing['width']:g} px"
