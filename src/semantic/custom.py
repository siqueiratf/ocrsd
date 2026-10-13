from __future__ import annotations

import random
import re
from datetime import date, time

from src.semantic.registry import REGISTRY
from src.semantic.resolve import medical_term


CUSTOM_KEY = re.compile(r"custom(?:\.[a-z][a-z0-9_]*){1,5}\Z")
DATA_TYPES = frozenset({"text", "integer", "boolean", "date", "time"})


class CustomGenerationUnavailable(ValueError):
    pass


def _plain_text(value: object, field: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} deve ser um texto")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > maximum or any(ord(character) < 32 for character in cleaned):
        raise ValueError(f"{field} deve conter de 1 a {maximum} caracteres válidos")
    return cleaned


def _choice(value: object, data_type: str) -> str | int | bool:
    if data_type == "text":
        return _plain_text(value, "Valor da opção", 120)
    if data_type == "integer":
        if type(value) is not int or abs(value) > 1_000_000:
            raise ValueError("Opções inteiras devem estar entre -1000000 e 1000000")
        return value
    if data_type == "boolean":
        if type(value) is not bool:
            raise ValueError("Opções booleanas devem ser true ou false")
        return value
    if not isinstance(value, str):
        raise ValueError("Datas e horas devem usar texto ISO")
    try:
        parsed = date.fromisoformat(value) if data_type == "date" else time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Data ou hora inválida; use o formato ISO") from exc
    if data_type == "time" and parsed.tzinfo is not None:
        raise ValueError("Horas com fuso horário não são aceitas")
    return parsed.isoformat()


def validate_custom_definition(key: object, label: object, description: object, data_type: object,
                               generator: object) -> dict:
    if not isinstance(key, str) or len(key) > 100 or CUSTOM_KEY.fullmatch(key) is None or key in REGISTRY:
        raise ValueError("Identificador semântico inválido; use custom.<nome> com letras minúsculas")
    label = _plain_text(label, "Rótulo", 100)
    description = _plain_text(description, "Descrição", 500)
    if not isinstance(data_type, str) or data_type not in DATA_TYPES:
        raise ValueError("Tipo semântico deve ser text, integer, boolean, date ou time")
    if not isinstance(generator, dict) or generator.get("type") not in {"choices", "integer_range", "unavailable"}:
        raise ValueError("Gerador deve ser choices, integer_range ou unavailable")
    method = generator["type"]
    if medical_term(f"{key} {label} {description}") and method != "unavailable":
        raise ValueError("Campos médicos exigem uma fonte e regras de validação documentadas antes da geração")
    if method == "unavailable":
        if set(generator) != {"type"}:
            raise ValueError("Gerador unavailable não aceita parâmetros")
        normalized = {"type": method}
    elif method == "integer_range":
        if data_type != "integer" or set(generator) != {"type", "min", "max"}:
            raise ValueError("integer_range exige tipo integer, min e max")
        minimum, maximum = generator["min"], generator["max"]
        if type(minimum) is not int or type(maximum) is not int or not (-1_000_000 <= minimum <= maximum <= 1_000_000):
            raise ValueError("Intervalo inteiro inválido ou fora dos limites")
        normalized = {"type": method, "min": minimum, "max": maximum}
    else:
        if set(generator) != {"type", "values"} or not isinstance(generator["values"], list):
            raise ValueError("choices exige uma lista de valores")
        values = generator["values"]
        if not 1 <= len(values) <= 50:
            raise ValueError("choices exige de 1 a 50 valores")
        normalized_values = [_choice(value, data_type) for value in values]
        if len({(type(value), value) for value in normalized_values}) != len(normalized_values):
            raise ValueError("choices não aceita valores repetidos")
        normalized = {"type": method, "values": normalized_values}
    return {"key": key, "label": label, "description": description, "data_type": data_type, "generator": normalized}


def approved_definitions(definitions: object) -> dict[str, dict]:
    entries = definitions.values() if isinstance(definitions, dict) else definitions or ()
    approved = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status", "approved") != "approved":
            continue
        try:
            definition = validate_custom_definition(*(entry[name] for name in ("key", "label", "description", "data_type", "generator")))
        except (KeyError, TypeError, ValueError):
            continue
        approved[definition["key"]] = definition
    return approved


def generate_custom_value(definition: dict, rng: random.Random) -> str | int | bool | date | time:
    validated = validate_custom_definition(*(definition[name] for name in ("key", "label", "description", "data_type", "generator")))
    generator = validated["generator"]
    if generator["type"] == "unavailable":
        raise CustomGenerationUnavailable(f"Não há gerador validado para {validated['key']}")
    if generator["type"] == "integer_range":
        return rng.randint(generator["min"], generator["max"])
    value = rng.choice(generator["values"])
    if validated["data_type"] == "date":
        return date.fromisoformat(value)
    if validated["data_type"] == "time":
        return time.fromisoformat(value)
    return value
