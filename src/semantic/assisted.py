from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, time, datetime
from typing import Any, Iterable, Protocol

from src.semantic.data import LOCATION_PROFILES, OCCUPATION_CODES, OCCUPATIONS, UF_VALUES
from src.semantic.plan import SemanticPlan, semantic_plan
from src.semantic.present import validation_rules
from src.semantic.record import SemanticRecord
from src.semantic.registry import REGISTRY
from src.semantic.resolve import document_context

PROMPT_VERSION = "registro-2"
SYSTEM_PROMPT = (
    "Você preenche documentos sintéticos brasileiros com dados fictícios, plausíveis e coerentes entre si, para treinar "
    "sistemas de OCR. Preencha somente as chaves pedidas em 'campos'. Quando um campo tiver 'opcoes', use exatamente uma "
    "delas, copiada literalmente; quando tiver 'intervalo', use um número dentro dele; quando tiver 'max_caracteres', nunca "
    "ultrapasse esse tamanho; quando tiver 'formato', siga-o à risca (datas em DD/MM/AAAA, horários em HH:MM, UF com duas "
    "letras maiúsculas). Use 'documento' e 'regras' como contexto sobre o tipo de documento e suas normas de preenchimento. "
    "Textos livres (observações, averbações, teor, poderes) devem ser prosa oficial realista em português, dentro do limite "
    "de caracteres. Nomes devem ser completos, fictícios e brasileiros, sem abreviações nem títulos. Pessoas da mesma "
    "família compartilham sobrenomes de forma plausível; datas devem seguir uma ordem cronológica plausível; municípios "
    "devem ser compatíveis com a UF informada; idades devem ser compatíveis com as datas do mesmo registro. Se "
    "'ja_definidos' existir, mantenha esses valores e gere o restante de forma coerente com eles. Nunca use nomes, números "
    "de documento ou dados de pessoas reais. Nunca produza diagnósticos, causas de morte, códigos CID ou outros dados "
    "médicos. Se 'correcoes' existir, corrija os problemas descritos. Responda somente com JSON no formato "
    '{"valores": {"chave": valor}}.')
CITIES = tuple(profile["city"] for profile in LOCATION_PROFILES)
OCCUPATION_CHOICES = tuple(dict.fromkeys((*OCCUPATIONS, *OCCUPATION_CODES)))
CONNECTORS = frozenset({"da", "das", "de", "do", "dos", "e"})
NAME_WORD = re.compile(r"^[A-ZÀ-Ý][a-zà-ÿ]+(?:['-][A-Za-zÀ-ÿ][a-zà-ÿ]+)?$")
DOCUMENT_TITLES = {"dnv": "Declaração de Nascido Vivo (DNV)", "declaracao_obito": "Declaração de Óbito (DO)"}
AI_KEY_PREFIX = "ai."
_AI_KEY_SUFFIX = {"sex": "sex", "uf": "uf", "cep": "cep"}
DATE_ANSWER_FORMAT = "%d/%m/%Y"
TIME_ANSWER_FORMAT = "%H:%M"
EXCLUDED_DATA_TYPES = frozenset({"document", "cpf", "cns", "dnv_number", "digit_sequence", "signature"})
DATA_TYPE_LABELS = {
    "text": "texto livre", "person_name": "nome de pessoa", "enum": "opção única", "uf": "UF (duas letras)",
    "integer": "número inteiro", "decimal": "número decimal", "date": "data", "time": "horário",
    "cep": "CEP", "checkbox": "seleção de opção", "text_sequence": "sequência de caracteres",
    "municipality": "município", "address": "endereço", "occupation": "ocupação", "sex": "sexo",
}
LINE_HEIGHT_PX = (18.0, 42.0)
CHAR_WIDTH_RATIO = 0.55


class AssistedGenerationError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts


class RecordAssistant(Protocol):
    provider: str
    model: str

    def complete_json(self, system: str, payload: dict, *, temperature: float, seed: int) -> tuple[dict, dict]: ...


@dataclass(frozen=True)
class Slot:
    key: str
    hint: str
    choices: tuple[str, ...] = ()
    low: int | None = None
    high: int | None = None

    def describe(self) -> dict:
        entry: dict[str, Any] = {"chave": self.key, "descricao": self.hint}
        if self.choices:
            entry["opcoes"] = list(self.choices)
        if self.low is not None:
            entry["intervalo"] = [self.low, self.high]
        return entry


SLOTS: dict[str, tuple[Slot, ...]] = {
    "dnv": (
        Slot("mother.name", "nome completo da mãe"),
        Slot("mother.age", "idade da mãe em anos completos no dia do parto", low=12, high=50),
        Slot("mother.occupation", "ocupação habitual da mãe", OCCUPATION_CHOICES),
        Slot("mother.residence.municipality", "município de residência da mãe", CITIES),
        Slot("father.name", "nome completo do pai"),
        Slot("father.age", "idade do pai em anos completos no dia do parto", low=14, high=80),
        Slot("newborn.sex", "sexo do recém-nascido", ("M", "F")),
        Slot("newborn.name", "nome completo do recém-nascido, com sobrenome(s) da mãe e/ou do pai"),
        Slot("birth.place.municipality", "município onde ocorreu o parto", CITIES),
        Slot("birth.delivery_type", "tipo de parto", ("Vaginal", "Cesáreo")),
        Slot("pregnancy.weeks", "semanas de gestação no parto", low=22, high=42),
    ),
    "declaracao_obito": (
        Slot("deceased.name", "nome completo da pessoa falecida"),
        Slot("deceased.sex", "sexo da pessoa falecida", ("M", "F")),
        Slot("deceased.age", "idade em anos completos na data do óbito", low=0, high=110),
        Slot("deceased.occupation", "ocupação habitual (use 'Não se aplica' para menores de 14 anos)",
             ("Não se aplica", *OCCUPATIONS)),
        Slot("deceased.mother_name", "nome completo da mãe da pessoa falecida"),
        Slot("deceased.father_name", "nome completo do pai da pessoa falecida"),
        Slot("deceased.residence.municipality", "município de residência da pessoa falecida", CITIES),
        Slot("death.place.municipality", "município onde ocorreu o óbito", CITIES),
        Slot("certifier.name", "nome completo do médico que atesta o óbito"),
    ),
}
SCENARIOS: dict[str, tuple[tuple[str, float], ...]] = {
    "dnv": (("mãe adolescente", 1), ("mãe entre 20 e 34 anos", 6), ("mãe com 35 anos ou mais", 2), ("parto prematuro", 1),
            ("família de cidade grande", 2), ("parto em município diferente da residência", 1)),
    "declaracao_obito": (("pessoa idosa", 6), ("adulto entre 30 e 59 anos", 3), ("adulto jovem", 1), ("criança", 0.3),
                         ("óbito fora do município de residência", 1)),
}
GENERIC_SCENARIOS: tuple[tuple[str, float], ...] = (
    ("pessoa jovem", 2), ("pessoa idosa", 2), ("família em capital", 2), ("família no interior", 2), ("cenário comum", 2),
)


def _plain(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold().strip()


def _unwrap(value: Any) -> Any:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return value


def check_name(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, "o nome precisa ser um texto"
    words = value.split()
    if not 2 <= len(words) <= 6 or len(value) > 70:
        return None, "o nome precisa ter de 2 a 6 palavras e até 70 caracteres"
    if not all(word in CONNECTORS or NAME_WORD.match(word) for word in words):
        return None, "o nome só pode ter palavras iniciadas por maiúscula e conectivos (da, de, do, dos, das, e)"
    if words[0] in CONNECTORS or words[-1] in CONNECTORS:
        return None, "o nome não pode começar nem terminar com conectivo"
    return " ".join(words), None


def check_value(slot: Slot, raw: Any) -> tuple[Any, str | None]:
    value = _unwrap(raw)
    kind = REGISTRY[slot.key].kind
    if kind == "name":
        return check_name(value)
    if kind == "integer":
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, bool) or not isinstance(value, int):
            return None, "o valor precisa ser um número inteiro"
        if not slot.low <= value <= slot.high:
            return None, f"o valor precisa estar entre {slot.low} e {slot.high}"
        return value, None
    if not isinstance(value, str):
        return None, "o valor precisa ser um texto"
    if kind == "sex" and _plain(value) in {"masculino", "feminino"}:
        value = "M" if _plain(value) == "masculino" else "F"
    found = next((choice for choice in slot.choices if _plain(choice) == _plain(value)), None)
    if found is None:
        return None, "o valor precisa ser uma das opções informadas"
    return found, None


def slots_for(context: str | None, keys: Iterable[str]) -> tuple[Slot, ...]:
    entities = {REGISTRY[key].entity for key in keys if key in REGISTRY}
    return tuple(slot for slot in SLOTS.get(context or "", ()) if REGISTRY[slot.key].entity in entities)


def plan_keys(plan: SemanticPlan, legacy_keys: Iterable[str]) -> list[str]:
    return [binding.key for binding in plan.bindings.values()] + list(legacy_keys)


# --- Generic (non-registry) field slots -------------------------------------------------------------------------------

@dataclass(frozen=True)
class GenericSlot:
    field_id: str
    descricao: str
    data_type: str
    choices: tuple[str, ...] = ()
    low: float | None = None
    high: float | None = None
    max_length: int | None = None
    element_count: int | None = None
    decimals: int | None = None
    formato: str | None = None

    def describe(self) -> dict:
        entry: dict[str, Any] = {"chave": self.field_id, "descricao": self.descricao,
                                 "tipo": DATA_TYPE_LABELS.get(self.data_type, self.data_type)}
        if self.choices:
            entry["opcoes"] = list(self.choices)
        if self.low is not None or self.high is not None:
            entry["intervalo"] = [self.low, self.high]
        if self.max_length is not None:
            entry["max_caracteres"] = self.max_length
        if self.formato:
            entry["formato"] = self.formato
        return entry


def ai_binding_key(field_id: str, data_type: str) -> str:
    suffix = _AI_KEY_SUFFIX.get(data_type)
    return f"{AI_KEY_PREFIX}{field_id}.{suffix}" if suffix else f"{AI_KEY_PREFIX}{field_id}"


def _numeric_pattern(pattern: str) -> bool:
    stripped = re.sub(r"\\[A-Za-z]", "", pattern)
    return not re.search(r"[A-Za-z]", stripped)


def _excluded_from_ai(field: dict) -> bool:
    data_type = field.get("data_type", "text")
    if data_type in EXCLUDED_DATA_TYPES or field.get("always_redact"):
        return True
    validation = validation_rules(field)
    if validation.get("mask"):
        return True
    elements = field.get("elements") or []
    if data_type == "checkbox" and not elements:
        return True
    pattern = validation.get("pattern")
    if isinstance(pattern, str) and pattern and _numeric_pattern(pattern):
        return True
    return False


def bbox_capacity(bbox: Iterable[float], dpi: float) -> int:
    x1, y1, x2, y2 = bbox
    scale = 300.0 / (dpi or 300.0)
    width, height = max(0.0, x2 - x1) * scale, max(0.0, y2 - y1) * scale
    line = min(max(height, LINE_HEIGHT_PX[0]), LINE_HEIGHT_PX[1])
    return max(1, int(width / (line * CHAR_WIDTH_RATIO))) * max(1, int(height // line))


def _descricao(field: dict) -> str:
    label = str(field.get("label") or field.get("description") or field.get("field_id"))
    description = str(field.get("description") or "")
    notes = str(field.get("notes") or "")
    text = label
    if description and description != label:
        text += f" — {description}"
    if notes:
        text += f" ({notes})"
    return text[:240]


def _formato(data_type: str) -> str | None:
    return {"date": "DD/MM/AAAA", "time": "HH:MM", "uf": "duas letras maiúsculas",
           "text_sequence": "apenas letras e números maiúsculos, sem espaços", "sex": "M ou F"}.get(data_type)


def _generic_slot(field: dict, dpi: float) -> GenericSlot:
    data_type = field.get("data_type", "text")
    validation = validation_rules(field)
    elements = field.get("elements") or []
    choices: tuple[str, ...] = ()
    if data_type == "checkbox":
        choices = tuple(dict.fromkeys(str(element.get("label") or element.get("value") or element.get("element_id"))
                                      for element in elements))
    elif validation.get("allowed_values"):
        choices = tuple(str(item) for item in validation["allowed_values"])
    elif data_type == "uf":
        choices = UF_VALUES
    elif data_type == "sex":
        choices = ("M", "F")
    low = high = None
    if data_type in {"integer", "decimal"}:
        low = validation.get("min_value", validation.get("min"))
        high = validation.get("max_value", validation.get("max"))
    max_length = validation.get("max_length")
    element_count = len(elements) if data_type == "text_sequence" and elements else None
    if element_count is not None:
        max_length = element_count
    elif data_type not in {"integer", "decimal", "date", "time", "sex", "checkbox"} and field.get("bbox"):
        capacity = bbox_capacity(field["bbox"], dpi)
        max_length = capacity if max_length is None else min(max_length, capacity)
    return GenericSlot(field_id=str(field["field_id"]), descricao=_descricao(field), data_type=data_type,
                       choices=choices, low=low, high=high, max_length=max_length, element_count=element_count,
                       decimals=validation.get("decimals"), formato=_formato(data_type))


def build_generic_slots(fields: Iterable[dict], dpi: float = 300.0) -> list[GenericSlot]:
    return [_generic_slot(field, dpi) for field in fields if not _excluded_from_ai(field)]


def ai_coverage(fields: list[dict], document: str | None, dpi: float = 300.0) -> tuple[set[str], set[str]]:
    """Returns (registry_field_ids, generic_field_ids) among SELECTED fields that AI generation can fill."""
    standard = [field for field in fields if not (isinstance(field.get("semantic"), str) and field["semantic"].startswith("custom."))]
    context = document_context(document)
    plan = semantic_plan(standard, context)
    registry_keys = {slot.key for slot in SLOTS.get(context or "", ())}
    registry_ids = {field_id for field_id, binding in plan.bindings.items() if binding.key in registry_keys}
    bound_ids = set(plan.bindings)
    custom_ids = {str(field["field_id"]) for field in fields
                 if isinstance(field.get("semantic"), str) and field["semantic"].startswith("custom.")}
    unbound = [field for field in fields if str(field["field_id"]) not in bound_ids and str(field["field_id"]) not in custom_ids]
    generic_ids = {slot.field_id for slot in build_generic_slots(unbound, dpi)}
    return registry_ids, generic_ids


def validate_generic_value(slot: GenericSlot, raw: Any) -> tuple[Any, str | None]:
    value = _unwrap(raw)
    data_type = slot.data_type
    if data_type == "date":
        if not isinstance(value, str):
            return None, "a data precisa ser um texto no formato DD/MM/AAAA"
        try:
            return datetime.strptime(value.strip(), DATE_ANSWER_FORMAT).date(), None
        except ValueError:
            return None, "a data precisa estar no formato DD/MM/AAAA"
    if data_type == "time":
        if not isinstance(value, str):
            return None, "o horário precisa ser um texto no formato HH:MM"
        try:
            return datetime.strptime(value.strip(), TIME_ANSWER_FORMAT).time(), None
        except ValueError:
            return None, "o horário precisa estar no formato HH:MM"
    if data_type == "sex":
        text = _plain(str(value)).upper() if not isinstance(value, str) else value.strip().upper()
        if text in {"M", "MASCULINO"}:
            return "M", None
        if text in {"F", "FEMININO"}:
            return "F", None
        return None, "o sexo precisa ser M ou F"
    if data_type in {"integer", "decimal"}:
        if isinstance(value, bool):
            return None, "o valor precisa ser numérico"
        try:
            number = float(str(value).strip().replace(",", ".")) if isinstance(value, str) else float(value)
        except (TypeError, ValueError):
            return None, "o valor precisa ser numérico"
        if data_type == "integer" and not number.is_integer():
            return None, "o valor precisa ser um número inteiro"
        if slot.low is not None and number < float(slot.low):
            return None, f"o valor precisa ser maior ou igual a {slot.low:g}"
        if slot.high is not None and number > float(slot.high):
            return None, f"o valor precisa ser menor ou igual a {slot.high:g}"
        if data_type == "integer":
            return int(number), None
        return f"{number:.{slot.decimals if slot.decimals is not None else 2}f}".replace(".", ","), None
    if not isinstance(value, str) or not value.strip():
        return None, "o valor precisa ser um texto"
    text = value.strip()
    if slot.choices:
        found = next((choice for choice in slot.choices if _plain(choice) == _plain(text)), None)
        if found is None:
            return None, "o valor precisa ser uma das opções informadas"
        text = found
    if data_type == "uf":
        text = text.upper()
        if not re.fullmatch(r"[A-Z]{2}", text):
            return None, "a UF precisa ter duas letras"
    if data_type == "text_sequence" and slot.element_count is not None:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        if len(cleaned) != slot.element_count:
            return None, f"o texto precisa ter exatamente {slot.element_count} caracteres"
        return cleaned, None
    if slot.max_length is not None and len(text) > slot.max_length:
        return None, f"o texto precisa ter até {slot.max_length} caracteres"
    return text, None


def _display_value(value: Any) -> Any:
    if isinstance(value, date):
        return value.strftime(DATE_ANSWER_FORMAT)
    if isinstance(value, time):
        return value.strftime(TIME_ANSWER_FORMAT)
    return value


def serialize_ai_values(values: dict[str, Any]) -> dict[str, dict]:
    serialized = {}
    for field_id, value in values.items():
        if isinstance(value, date):
            serialized[field_id] = {"kind": "date", "value": value.isoformat()}
        elif isinstance(value, time):
            serialized[field_id] = {"kind": "time", "value": value.strftime(TIME_ANSWER_FORMAT)}
        else:
            serialized[field_id] = {"kind": "raw", "value": value}
    return serialized


def restore_ai_values(stored: dict[str, Any]) -> dict[str, Any]:
    restored = {}
    for field_id, entry in (stored or {}).items():
        if not isinstance(entry, dict):
            restored[field_id] = entry
            continue
        kind, value = entry.get("kind"), entry.get("value")
        if kind == "date" and isinstance(value, str):
            restored[field_id] = date.fromisoformat(value)
        elif kind == "time" and isinstance(value, str):
            restored[field_id] = datetime.strptime(value, TIME_ANSWER_FORMAT).time()
        else:
            restored[field_id] = value
    return restored


class AssistedRecords:
    def __init__(self, assistant: RecordAssistant, *, temperature: float = 0.8, attempts: int = 3,
                 model_digest: str | None = None, fields: Iterable[dict] = (), dpi: float = 300.0,
                 document_meta: dict | None = None, rules_text: str | None = None,
                 max_fields_per_request: int = 24):
        if not 1 <= attempts <= 5:
            raise ValueError("O número de tentativas da IA precisa estar entre 1 e 5")
        self.assistant, self.temperature, self.attempts, self.model_digest = assistant, float(temperature), int(attempts), model_digest
        self.fields = list(fields)
        self.dpi = float(dpi or 300.0)
        self.document_meta = document_meta or {}
        self.rules_text = rules_text
        self.max_fields_per_request = max(1, int(max_fields_per_request))
        self.documents: list[dict] = []

    @property
    def parameters(self) -> dict:
        return {"provider": self.assistant.provider, "model": self.assistant.model, "model_digest": self.model_digest,
                "prompt_version": PROMPT_VERSION, "temperature": self.temperature, "max_attempts": self.attempts}

    def _chunks(self, slots: list[GenericSlot]) -> list[list[GenericSlot]]:
        size = self.max_fields_per_request
        return [slots[index:index + size] for index in range(0, len(slots), size)]

    def _documento(self, context: str | None) -> dict:
        documento = {"nome": self.document_meta.get("nome") or DOCUMENT_TITLES.get(context or "", context),
                     "orgao": self.document_meta.get("orgao"), "descricao": self.document_meta.get("descricao")}
        if self.rules_text:
            documento["regras"] = self.rules_text
        return documento

    def _run_generic_batch(self, batch: list[GenericSlot], documento: dict, scenario: str, seed: int,
                           attempts: list[dict], ja_definidos: dict) -> tuple[dict[str, Any], dict[str, str]]:
        corrections: list[str] = []
        resolved: dict[str, Any] = {}
        errors: dict[str, str] = {}
        batch_tag = sum(ord(character) for character in (batch[0].field_id if batch else "")) + len(batch)
        for attempt in range(1, self.attempts + 1):
            pending = [slot for slot in batch if slot.field_id not in resolved]
            if not pending:
                break
            payload = {"documento": documento, "cenario_sugerido": scenario, "campos": [slot.describe() for slot in pending],
                       **({"ja_definidos": ja_definidos} if ja_definidos else {}),
                       **({"correcoes": corrections} if corrections else {})}
            entry: dict[str, Any] = {"attempt": attempt, "group": "campos", "fields": [slot.field_id for slot in pending]}
            attempts.append(entry)
            request_seed = (seed + 1000 * (batch_tag + 1) + attempt) % 2**31
            try:
                answer, meta = self.assistant.complete_json(SYSTEM_PROMPT, payload, temperature=self.temperature, seed=request_seed)
            except Exception as exc:
                entry["error"] = f"falha na inferência: {exc}"
                corrections = []
                continue
            entry.update(meta)
            entry["request_seed"] = request_seed
            proposed = answer.get("valores") if isinstance(answer.get("valores"), dict) else answer
            rejected = []
            for slot in pending:
                if not isinstance(proposed, dict) or slot.field_id not in proposed:
                    rejected.append({"key": slot.field_id, "reason": "valor ausente"})
                    errors[slot.field_id] = "valor ausente"
                    continue
                value, problem = validate_generic_value(slot, proposed[slot.field_id])
                if problem:
                    rejected.append({"key": slot.field_id, "reason": problem})
                    errors[slot.field_id] = problem
                else:
                    resolved[slot.field_id] = value
                    errors.pop(slot.field_id, None)
            entry.update({"accepted": sorted(resolved), "rejected": rejected})
            corrections = [f"{item['key']}: {item['reason']}" for item in rejected]
        return resolved, errors

    def __call__(self, index: int, seed: int, context: str | None, plan: SemanticPlan, legacy_keys: list[str]) -> SemanticRecord:
        bound_ids = set(plan.bindings)
        custom_ids = {str(field["field_id"]) for field in self.fields
                     if isinstance(field.get("semantic"), str) and field["semantic"].startswith("custom.")}
        unbound_fields = [field for field in self.fields
                          if str(field["field_id"]) not in bound_ids and str(field["field_id"]) not in custom_ids]
        registry_slots = slots_for(context, plan_keys(plan, legacy_keys))
        generic_slots = build_generic_slots(unbound_fields, self.dpi)
        if not registry_slots and not generic_slots:
            raise AssistedGenerationError("Nenhum campo selecionado admite geração assistida por IA neste modelo", [])
        rng = random.Random(f"{seed}:ai")
        labels, weights = zip(*SCENARIOS.get(context or "", GENERIC_SCENARIOS))
        scenario = rng.choices(labels, weights)[0]
        documento = self._documento(context)
        attempts: list[dict] = []
        record: SemanticRecord | None = None
        registry_accepted: dict[str, Any] = {}
        registry_rejected_total = 0
        if registry_slots:
            corrections: list[str] = []
            for attempt in range(1, self.attempts + 1):
                payload = {"documento": documento, "cenario_sugerido": scenario, "campos": [slot.describe() for slot in registry_slots],
                          **({"correcoes": corrections} if corrections else {})}
                entry: dict[str, Any] = {"attempt": attempt}
                attempts.append(entry)
                request_seed = (seed + attempt) % 2**31
                try:
                    answer, meta = self.assistant.complete_json(SYSTEM_PROMPT, payload, temperature=self.temperature, seed=request_seed)
                except Exception as exc:
                    entry["error"] = f"falha na inferência: {exc}"
                    corrections = []
                    continue
                entry.update(meta)
                entry["request_seed"] = request_seed
                proposed = answer.get("valores") if isinstance(answer.get("valores"), dict) else answer
                accepted, rejected = {}, []
                for slot in registry_slots:
                    if slot.key not in proposed:
                        rejected.append({"key": slot.key, "reason": "valor ausente"})
                        continue
                    value, problem = check_value(slot, proposed[slot.key])
                    if problem:
                        rejected.append({"key": slot.key, "reason": problem})
                    else:
                        accepted[slot.key] = value
                entry.update({"accepted": accepted, "rejected": rejected})
                if len(accepted) < math.ceil(len(registry_slots) / 2):
                    entry["error"] = "a resposta tem poucos valores válidos"
                    corrections = [f"{item['key']}: {item['reason']}" for item in rejected]
                    continue
                candidate = SemanticRecord(seed, context)
                for key, value in accepted.items():
                    candidate.require(key, (value,), "valor proposto pela IA")
                try:
                    plan.apply(candidate, legacy_keys)
                except ValueError as exc:
                    entry["error"] = f"o registro proposto viola restrições semânticas: {exc}"
                    corrections = [str(exc)[:600]]
                    continue
                record, registry_accepted, registry_rejected_total = candidate, accepted, len(rejected)
                break
            if record is None:
                self.documents.append({"index": index, "status": "rejected", "attempts": len(attempts)})
                reasons = "; ".join(entry["error"] for entry in attempts if entry.get("error"))
                raise AssistedGenerationError(
                    f"A IA não produziu um registro válido para o documento {index} após {self.attempts} tentativas: {reasons}", attempts)
        else:
            record = SemanticRecord(seed, context)
            plan.apply(record, legacy_keys)
        ja_definidos: dict[str, Any] = dict(registry_accepted)
        shown: set[str] = set()
        for field_id, binding in plan.bindings.items():
            value = record._values.get(binding.key)
            if binding.key not in shown and isinstance(value, (str, int, float, date, time)) and not isinstance(value, bool):
                shown.add(binding.key)
                ja_definidos.setdefault(field_id, _display_value(value))
        generic_accepted: dict[str, Any] = {}
        fallback_fields: dict[str, str] = {}
        for batch in self._chunks(generic_slots):
            resolved, errors = self._run_generic_batch(batch, documento, scenario, seed, attempts, ja_definidos)
            for slot in batch:
                if slot.field_id in resolved:
                    generic_accepted[slot.field_id] = resolved[slot.field_id]
                    ja_definidos[slot.field_id] = _display_value(resolved[slot.field_id])
                else:
                    fallback_fields[slot.field_id] = errors.get(slot.field_id, "sem resposta válida da IA")
        if not registry_slots and generic_slots and not any(not entry.get("error") for entry in attempts):
            self.documents.append({"index": index, "status": "rejected", "attempts": len(attempts)})
            reasons = "; ".join(entry["error"] for entry in attempts if entry.get("error"))
            raise AssistedGenerationError(f"Assistente de IA indisponível para o documento {index}: {reasons}", attempts)
        record.ai_values = generic_accepted
        record.provenance = {"mode": "ai", **self.parameters, "scenario": scenario, "ai_keys": sorted(registry_accepted),
                             "accepted": registry_accepted, "attempts": attempts,
                             "validation": {"status": "accepted", "rejected_values": registry_rejected_total},
                             "ai_fields": sorted(generic_accepted), "fallback_fields": fallback_fields}
        self.documents.append({"index": index, "status": "accepted", "attempts": len(attempts), "ai_keys": sorted(registry_accepted),
                               "ai_fields": sorted(generic_accepted), "fallback_fields": sorted(fallback_fields)})
        return record
