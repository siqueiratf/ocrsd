from __future__ import annotations

import math
import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from src.semantic.data import LOCATION_PROFILES, OCCUPATION_CODES, OCCUPATIONS
from src.semantic.plan import SemanticPlan
from src.semantic.record import SemanticRecord
from src.semantic.registry import REGISTRY

PROMPT_VERSION = "registro-1"
SYSTEM_PROMPT = (
    "Você cria cenários fictícios e coerentes para documentos sintéticos do registro civil brasileiro. Preencha somente as "
    "chaves pedidas em 'campos'. Quando um campo tiver 'opcoes', use exatamente uma delas; quando tiver 'intervalo', use um "
    "inteiro dentro dele. Nomes devem ser completos, fictícios e brasileiros, sem abreviações nem títulos. Pessoas da mesma "
    "família compartilham sobrenomes de forma plausível. Nunca produza diagnósticos, causas de morte, códigos CID ou outros "
    "dados médicos. Se 'correcoes' existir, corrija os problemas descritos. Responda somente com JSON no formato "
    '{"valores": {"chave": valor}}.')
CITIES = tuple(profile["city"] for profile in LOCATION_PROFILES)
OCCUPATION_CHOICES = tuple(dict.fromkeys((*OCCUPATIONS, *OCCUPATION_CODES)))
CONNECTORS = frozenset({"da", "das", "de", "do", "dos", "e"})
NAME_WORD = re.compile(r"^[A-ZÀ-Ý][a-zà-ÿ]+(?:['-][A-Za-zÀ-ÿ][a-zà-ÿ]+)?$")
DOCUMENT_TITLES = {"dnv": "Declaração de Nascido Vivo (DNV)", "declaracao_obito": "Declaração de Óbito (DO)"}


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


class AssistedRecords:
    def __init__(self, assistant: RecordAssistant, *, temperature: float = 0.8, attempts: int = 3,
                 model_digest: str | None = None):
        if not 1 <= attempts <= 5:
            raise ValueError("O número de tentativas da IA precisa estar entre 1 e 5")
        self.assistant, self.temperature, self.attempts, self.model_digest = assistant, float(temperature), int(attempts), model_digest
        self.documents: list[dict] = []

    @property
    def parameters(self) -> dict:
        return {"provider": self.assistant.provider, "model": self.assistant.model, "model_digest": self.model_digest,
                "prompt_version": PROMPT_VERSION, "temperature": self.temperature, "max_attempts": self.attempts}

    def __call__(self, index: int, seed: int, context: str | None, plan: SemanticPlan, legacy_keys: list[str]) -> SemanticRecord:
        slots = slots_for(context, plan_keys(plan, legacy_keys))
        if not slots:
            raise AssistedGenerationError("Nenhum campo selecionado admite geração assistida por IA neste modelo", [])
        rng = random.Random(f"{seed}:ai")
        labels, weights = zip(*SCENARIOS.get(context or "", (("cenário comum", 1),)))
        scenario = rng.choices(labels, weights)[0]
        attempts: list[dict] = []
        corrections: list[str] = []
        for attempt in range(1, self.attempts + 1):
            payload = {"documento": DOCUMENT_TITLES.get(context or "", context), "cenario_sugerido": scenario,
                       "campos": [slot.describe() for slot in slots], **({"correcoes": corrections} if corrections else {})}
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
            for slot in slots:
                if slot.key not in proposed:
                    rejected.append({"key": slot.key, "reason": "valor ausente"})
                    continue
                value, problem = check_value(slot, proposed[slot.key])
                if problem:
                    rejected.append({"key": slot.key, "value": _unwrap(proposed[slot.key]), "reason": problem})
                else:
                    accepted[slot.key] = value
            entry.update({"accepted": accepted, "rejected": rejected})
            if len(accepted) < math.ceil(len(slots) / 2):
                entry["error"] = "a resposta tem poucos valores válidos"
                corrections = [f"{item['key']}: {item['reason']}" for item in rejected]
                continue
            record = SemanticRecord(seed, context)
            for key, value in accepted.items():
                record.require(key, (value,), "valor proposto pela IA")
            try:
                plan.apply(record, legacy_keys)
            except ValueError as exc:
                entry["error"] = f"o registro proposto viola restrições semânticas: {exc}"
                corrections = [str(exc)[:600]]
                continue
            record.provenance = {"mode": "ai", **self.parameters, "scenario": scenario, "ai_keys": sorted(accepted),
                                 "accepted": accepted, "attempts": attempts,
                                 "validation": {"status": "accepted", "rejected_values": len(rejected)}}
            self.documents.append({"index": index, "status": "accepted", "attempts": len(attempts), "ai_keys": sorted(accepted)})
            return record
        self.documents.append({"index": index, "status": "rejected", "attempts": len(attempts)})
        reasons = "; ".join(entry["error"] for entry in attempts if entry.get("error"))
        raise AssistedGenerationError(
            f"A IA não produziu um registro válido para o documento {index} após {self.attempts} tentativas: {reasons}", attempts)
