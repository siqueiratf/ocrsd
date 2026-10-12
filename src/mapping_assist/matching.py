from __future__ import annotations

import functools
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable

from src.semantic.custom import validate_custom_definition
from src.semantic.registry import CONTEXTS, REGISTRY
from src.semantic.resolve import document_context, excluded_terms, normalize, resolve_field

if TYPE_CHECKING:
    from src.mapping_assist.analysis import FieldCandidate

MIN_SCORE, MIN_MARGIN, AMBIGUOUS_FLOOR = 0.5, 0.2, 0.35
ROMANS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
SECTIONS = {
    "dnv": {"I": ("newborn", "birth.date", "birth.time"), "II": ("birth.establishment", "birth.place"), "III": ("mother",),
            "IV": ("guardian", "father"), "V": ("mother.obstetric", "pregnancy"), "VI": (), "VII": ("filling",),
            "VIII": ("registration",)},
    "declaracao_obito": {"I": ("deceased", "death.date", "death.time"), "II": ("deceased.residence",), "III": ("death.place",),
                         "IV": (), "V": (), "VI": ("certifier",), "VII": (), "VIII": ("death_registration",), "IX": ()},
}
SECTION_TITLES = {
    "dnv": (("recem nascido", "I"), ("ocorrencia", "II"), ("responsavel legal", "IV"), ("pai", "IV"), ("mae", "III"),
            ("parturiente", "III"), ("gestacao", "V"), ("parto", "V"), ("anomalia", "VI"), ("preenchimento", "VII"),
            ("cartorio", "VIII")),
    "declaracao_obito": (("identificacao", "I"), ("residencia", "II"), ("causas externas", "VII"), ("ocorrencia", "III"),
                         ("fetal", "IV"), ("menor", "IV"), ("condicoes", "V"), ("causas", "V"), ("medico", "VI"),
                         ("cartorio", "VIII"), ("localid", "IX"), ("declarante", "IX")),
}
NEW_PREFIXES = {
    "dnv": {"I": "newborn", "II": "birth.place", "III": "mother", "IV": "guardian", "V": "pregnancy", "VI": "newborn.anomaly",
            "VII": "filling", "VIII": "registration"},
    "declaracao_obito": {"I": "deceased", "II": "deceased.residence", "III": "death.place", "IV": "death.infant",
                         "V": "death.cause", "VI": "certifier", "VII": "death.external", "VIII": "death_registration",
                         "IX": "death.declarant"},
}
DEFAULT_PREFIX = {"dnv": "newborn", "declaracao_obito": "death", "certidao_nascimento": "registration"}
ANCHORS = {"dnv": "birth.date", "certidao_nascimento": "birth.date", "declaracao_obito": "death.date"}
IMPLIED = {"death": {"obito", "morte"}, "deceased": {"falecido", "falecida"}, "cause": {"causa", "causas"},
           "newborn": {"recem", "nascido", "rn"}, "mother": {"mae", "parturiente"}, "certifier": {"medico"},
           "residence": {"residencia"}, "place": {"ocorrencia", "local"}, "registration": {"cartorio"}}
STOPWORDS = frozenset("a o as os e de da do das dos em no na nos nas por para se ou com sem que ao aos um uma etc".split())
MEDICAL = ("causa", "causas", "cid", "doenca", "diagnostico", "morbido", "morbidos", "necropsia", "morte", "anomalia",
           "anomalias", "apgar", "peso", "perimetro", "comprimento", "gravidez", "parto", "partos", "gestacao", "gestacoes",
           "fetal", "fetais", "aborto", "abortos", "cesarea", "cesareas", "apresentacao", "induzido", "menstruacao",
           "assistencia medica", "lesao", "lesoes", "acidente", "suicidio", "homicidio", "violencia", "cirurgia", "tratamento")
MEDICAL_SECTIONS = {"declaracao_obito": {"IV", "V", "VII"}, "dnv": {"VI"}}
MEDICAL_GENERATOR = "requer regra validada (ex.: CID-10)"
DIGIT_LENGTHS = {"cpf": 11, "cns": 15, "cep": 8, "ibge": 7, "cnes": 7, "dnv_number": 11, "matricula": 32}
FREE_DIGIT_KINDS = frozenset({"integer", "house_number", "document_number"})
KIND_NAMES = {"checkbox": "caixas de seleção", "digit_sequence": "caixas de algarismos", "text": "texto"}


@dataclass(frozen=True)
class Match:
    candidate_id: str
    key: str | None
    status: str
    method: str
    score: float | None
    alternatives: tuple[tuple[str, float], ...]
    reason: str
    proposal: dict | None = None


def _features(text: str) -> Counter:
    plain = normalize(text)
    words = plain.split()
    found = Counter(f"w:{word}" for word in words)
    found.update(f"b:{first}_{second}" for first, second in zip(words, words[1:]))
    padded = f" {plain} "
    found.update(f"c:{padded[i:i + n]}" for n in (3, 4, 5) for i in range(len(padded) - n + 1))
    return found


class NgramIndex:
    def __init__(self, documents: Iterable[tuple[str, str]]):
        counted = [(key, _features(text)) for key, text in documents]
        frequency = Counter(gram for _, features in counted for gram in features)
        total = len(counted)
        self.idf = {gram: math.log((1 + total) / (1 + count)) + 1 for gram, count in frequency.items()}
        self.unseen = math.log(1 + total) + 1
        self.vectors = [(key, self.vector(features)) for key, features in counted]

    def vector(self, features: Counter) -> dict[str, float]:
        weights = {gram: (1 + math.log(count)) * self.idf.get(gram, self.unseen) for gram, count in features.items()}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        return {gram: value / norm for gram, value in weights.items()}

    def scores(self, text: str, allowed: Iterable[str]) -> list[tuple[str, float]]:
        allowed, query, best = set(allowed), self.vector(_features(text)), {}
        for key, vector in self.vectors:
            if key in allowed:
                score = sum(value * vector.get(gram, 0.0) for gram, value in query.items())
                best[key] = max(best.get(key, 0.0), score)
        return sorted(best.items(), key=lambda item: (-item[1], item[0]))


@functools.lru_cache(maxsize=1)
def registry_index() -> NgramIndex:
    return NgramIndex((key, text) for key, field in REGISTRY.items()
                      for text in (field.label, field.description, *field.synonyms, f"{field.group} {field.label}"))


def _has(text: str, phrases: Iterable[str]) -> str | None:
    padded = f" {text} "
    return next((phrase for phrase in phrases if f" {phrase} " in padded), None)


def section_of(document: str | None, section: str | None, title: str | None = None) -> str | None:
    tokens = re.split(r"[^A-Za-z0-9]+", str(section or "").upper())
    numeral = next((token for token in tokens if token in ROMANS), None)
    if numeral:
        return numeral
    text = normalize(f"{section or ''} {title or ''}")
    return next((value for phrase, value in SECTION_TITLES.get(document or "", ()) if _has(text, (phrase,))), None)


def _within(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + ".")


def key_section(document: str | None, key: str) -> str | None:
    found = [(len(prefix), name) for name, prefixes in SECTIONS.get(document or "", {}).items()
             for prefix in prefixes if _within(key, prefix)]
    return max(found)[1] if found else None


def kind_fits(semantic_kind: str, kind: str, count: int) -> bool:
    if kind == "checkbox":
        return semantic_kind == "sex"
    if kind != "digit_sequence":
        return True
    if semantic_kind == "date":
        return count in {0, 6, 8}
    if semantic_kind == "time":
        return count in {0, 4}
    if semantic_kind in DIGIT_LENGTHS:
        return count in {0, DIGIT_LENGTHS[semantic_kind]}
    return semantic_kind in FREE_DIGIT_KINDS


def allowed_keys(document: str | None, section: str | None, kind: str = "text", count: int = 0) -> list[str]:
    keys = [key for key, field in REGISTRY.items() if document not in CONTEXTS or document in field.contexts]
    if section and section in SECTIONS.get(document or "", {}):
        keys = [key for key in keys if key_section(document, key) == section]
    return [key for key in keys if kind_fits(REGISTRY[key].kind, kind, count)]


def _options(candidate: FieldCandidate) -> list[str]:
    return [str(element.get("label") or element.get("code") or "").strip() for element in candidate.elements
            if str(element.get("label") or element.get("code") or "").strip()]


def guess_data_type(candidate: FieldCandidate) -> str:
    if candidate.kind in {"checkbox", "digit_sequence"}:
        return candidate.kind
    text = normalize(candidate.label)
    for phrases, data_type in ((("data",), "date"), (("hora", "horario"), "time"), (("nome",), "person_name")):
        if _has(text, phrases):
            return data_type
    if _has(text, ("municipio", "cidade")) and not _has(text, ("uf",)):
        return "municipality"
    return "uf" if text in {"uf", "unidade da federacao"} else "text"


def _rules_field(candidate: FieldCandidate, section: str | None) -> dict:
    text, count = normalize(candidate.label), len(candidate.elements)
    data_type = None
    if candidate.kind == "checkbox":
        data_type = "checkbox"
    elif candidate.kind == "digit_sequence":
        data_type = ("date" if count in {6, 8} and _has(text, ("data",)) else
                     "time" if count == 4 and _has(text, ("hora", "horario")) else "digit_sequence")
    field = {"label": candidate.label, "bloco": section or candidate.section}
    return {**field, "data_type": data_type} if data_type else field


def _medical(candidate: FieldCandidate, document: str | None, section: str | None) -> bool:
    return bool(_has(normalize(candidate.label), MEDICAL)) or section in MEDICAL_SECTIONS.get(document or "", set())


def _slug(text: str, drop: set[str] = frozenset(), limit: int = 4) -> str:
    words = [word for word in normalize(re.sub(r"\([^)]*\)", " ", text)).split() if word not in STOPWORDS]
    kept = [word for word in words if word not in drop] or words
    return "_".join(kept[:limit])


def _semantic_kind(candidate: FieldCandidate, data_type: str, medical: bool) -> str:
    text, count = normalize(candidate.label), len(candidate.elements)
    if medical:
        return "text"
    if candidate.kind == "checkbox":
        return "boolean" if {normalize(option) for option in _options(candidate)} <= {"sim", "nao", "ignorado"} and _options(candidate) else "text"
    if candidate.kind == "digit_sequence":
        if _has(text, ("data",)) and count in {6, 8}:
            return "date"
        if _has(text, ("hora",)) and count == 4:
            return "time"
        return "integer" if count <= 3 else "document_number"
    return {"date": "date", "time": "time", "person_name": "name", "municipality": "municipality", "uf": "uf"}.get(
        data_type, "occupation" if _has(text, ("ocupacao", "profissao")) else "integer" if _has(text, ("idade",)) else "text")


def _generator(candidate: FieldCandidate, kind: str, medical: bool, document: str | None) -> str:
    options = ", ".join(_options(candidate))
    if medical:
        return MEDICAL_GENERATOR + (f"; opções detectadas: {options}" if options else "")
    if candidate.kind == "checkbox":
        return f"lista de opções detectadas: {options}" if options else "lista de opções a definir pelo revisor"
    count = len(candidate.elements)
    return {"date": f"data coerente com {ANCHORS.get(document or '', 'a data principal do documento')}",
            "time": "hora no formato HH:MM", "name": "nome completo sintético",
            "municipality": "município sintético coerente com a UF", "uf": "UF sintética coerente com o município",
            "occupation": "ocupação sintética da lista CBO", "boolean": "sim ou não",
            "integer": f"inteiro de até {count or 3} algarismos em faixa a validar",
            "document_number": f"sequência sintética de {count} algarismos"}.get(
        kind, "texto sintético; requer lista de valores ou regra validada")


def new_proposal(candidate: FieldCandidate, document: str | None, section: str | None, reason: str) -> dict:
    medical = _medical(candidate, document, section)
    prefix = NEW_PREFIXES.get(document or "", {}).get(section or "") or DEFAULT_PREFIX.get(document or "", "custom")
    drop = set().union(*(IMPLIED.get(part, set()) for part in prefix.split(".")))
    label = re.sub(r"[\s:?]+$", "", candidate.label.strip()) or f"Campo {candidate.id}"
    tail = label.rsplit(" - ", 1)[-1]
    key = f"{prefix}.{_slug(tail, drop) or _slug(label, drop) or 'campo_' + re.sub(r'[^a-z0-9]+', '_', candidate.id.lower())}"
    key = key if key not in REGISTRY else f"{key}_novo"
    data_type = guess_data_type(candidate)
    kind = _semantic_kind(candidate, data_type, medical)
    where = f"seção {section}" if section else "documento"
    title = f" ({candidate.section_title})" if candidate.section_title else ""
    return {"key": key, "label": label[:1].upper() + label[1:], "kind": kind, "data_type": data_type,
            "description": f"{label} — campo detectado na {where}{title}, sem chave no registro semântico",
            "group": candidate.section_title or (f"Seção {section}" if section else "Campos novos"),
            "generator": _generator(candidate, kind, medical, document), "reason": reason}


def _alternatives(ranked: list[tuple[str, float]], first: str | None = None) -> tuple[tuple[str, float], ...]:
    scores = dict(ranked)
    ordered = ([(first, scores.get(first, 1.0))] if first else []) + [item for item in ranked if item[0] != first and item[1] >= 0.05]
    return tuple((key, round(score, 3)) for key, score in ordered[:3])


def match_one(candidate: FieldCandidate, document: str | None, min_score: float = MIN_SCORE,
              min_margin: float = MIN_MARGIN) -> Match:
    section = section_of(document, candidate.section, candidate.section_title)
    text, count = normalize(candidate.label), len(candidate.elements)

    def new(reason: str, alternatives: tuple = (), score: float | None = None) -> Match:
        return Match(candidate.id, None, "new", "none", score, alternatives, reason, new_proposal(candidate, document, section, reason))

    if not text:
        return new("Rótulo vazio: não há texto para comparar com o registro semântico.")
    allowed = allowed_keys(document, section, candidate.kind, count)
    ranked = registry_index().scores(candidate.label, allowed) if allowed else []
    binding = resolve_field(_rules_field(candidate, section), document)
    where = f" na seção {section}" if section else ""
    tied = [key for key, score in ranked if ranked[0][1] >= min_score and score >= ranked[0][1] - 0.02]
    if binding and binding.key in allowed and section is None and len(tied) > 1 and binding.key in tied:
        names = ", ".join(f"“{REGISTRY[key].label}”" for key in tied[:3])
        return Match(candidate.id, None, "ambiguous", "rules", None, _alternatives(ranked, binding.key),
                     f"Rótulo genérico sem seção identificada: pode ser {names}; revisar.")
    if binding and binding.key in allowed:
        return Match(candidate.id, binding.key, "matched", "rules", 1.0, _alternatives(ranked, binding.key),
                     f"Regra semântica: “{candidate.label}”{where} corresponde a “{REGISTRY[binding.key].label}”.")
    if binding and binding.key in allowed_keys(document, None, candidate.kind, count):
        return Match(candidate.id, None, "ambiguous", "rules", None, _alternatives(ranked, binding.key),
                     f"A regra indica “{REGISTRY[binding.key].label}”, que não pertence à seção {section}; revisar.")
    if section and candidate.kind == "digit_sequence" and text in {"data", "hora", "horario"}:
        expected = "date" if text == "data" else "time"
        choices = [key for key in allowed if REGISTRY[key].kind == expected]
        if len(choices) == 1:
            key = choices[0]
            return Match(candidate.id, key, "matched", "rules", 1.0, _alternatives(ranked, key),
                         f"A seção {section} e {count} caixas identificam “{REGISTRY[key].label}”.")
    if term := _has(text, excluded_terms(document)):
        return new(f"O termo “{term}” é clínico, estatístico ou administrativo e não tem chave no registro semântico.")
    if not allowed:
        if section and not SECTIONS.get(document or "", {}).get(section, True):
            return new(f"A seção {section} não tem chaves no registro semântico deste documento.")
        return new(f"Nenhuma chave do registro é compatível com {KIND_NAMES.get(candidate.kind, candidate.kind)}{where}.")
    top, best = ranked[0]
    margin = best - (ranked[1][1] if len(ranked) > 1 else 0.0)
    alternatives = _alternatives(ranked)
    if best >= min_score and margin >= min_margin:
        return Match(candidate.id, top, "matched", "ngram", round(best, 3), alternatives,
                     f"Semelhança textual {best:.2f} com “{REGISTRY[top].label}” (margem {margin:.2f}).")
    if best >= min(min_score, AMBIGUOUS_FLOOR):
        detail = (f"margem {margin:.2f} abaixo de {min_margin:.2f}" if best >= min_score
                  else f"semelhança {best:.2f} abaixo do limiar {min_score:.2f}")
        names = ", ".join(f"“{REGISTRY[key].label}”" for key, _ in alternatives[:2])
        return Match(candidate.id, None, "ambiguous", "ngram", round(best, 3), alternatives,
                     f"Correspondência incerta ({detail}): revisar entre {names}.")
    if candidate.kind == "checkbox":
        return new(f"Caixas de seleção só se vinculam a chaves de sexo; “{candidate.label}” não corresponde a nenhuma.",
                   alternatives, round(best, 3))
    return new(f"Nenhuma chave acima do limiar: a melhor (“{REGISTRY[top].label}”) tem semelhança {best:.2f}.",
               alternatives, round(best, 3))


def _assisted(matches: list[Match], candidates: list[FieldCandidate], document: str | None, provider: Any) -> list[Match]:
    by_id = {candidate.id: candidate for candidate in candidates}
    ambiguous = [match for match in matches if match.status == "ambiguous" and match.alternatives]
    new = [match for match in matches if match.status == "new"]
    if not ambiguous and not new:
        return matches
    try:
        status = provider.status()
        available, name, detail = status.available, status.provider, status.detail
    except Exception as exc:
        available, name, detail = False, "assistente", str(exc) or type(exc).__name__
    if not available:
        if name == "none":
            return matches
        note = f" Assistente indisponível: {detail}."
        return [replace(match, reason=match.reason + note) if match.status == "ambiguous" else match for match in matches]
    updates: dict[str, Match] = {}
    if ambiguous:
        allowed = {match.candidate_id: match.alternatives for match in ambiguous}
        try:
            choices, failure = provider.disambiguate(document, [by_id[key] for key in allowed], allowed), None
        except Exception as exc:
            choices, failure = {}, str(exc) or type(exc).__name__
        for match in ambiguous:
            choice = choices.get(match.candidate_id)
            if not isinstance(choice, tuple) or len(choice) != 2:
                note = f"falhou ({failure})" if failure else "não respondeu"
                updates[match.candidate_id] = replace(match, reason=f"{match.reason} Assistente {note}.")
                continue
            key, why = choice
            scores = dict(match.alternatives)
            if key is None:
                candidate = by_id[match.candidate_id]
                reason = f"Assistente: nenhuma chave existente se aplica — {why}"
                updates[match.candidate_id] = Match(match.candidate_id, None, "new", "assistant", match.score, match.alternatives,
                                                    reason, new_proposal(candidate, document, section_of(
                                                        document, candidate.section, candidate.section_title), reason))
            elif key in scores:
                updates[match.candidate_id] = Match(match.candidate_id, key, "matched", "assistant", scores[key],
                                                    match.alternatives, f"Assistente escolheu “{REGISTRY[key].label}”: {why}")
    pending = [match for match in new if match.proposal] + [match for match in updates.values() if match.status == "new"]
    if pending:
        try:
            extra = provider.propose_categories(document, [by_id[match.candidate_id] for match in pending])
        except Exception:
            extra = {}
        for match in pending:
            suggestion = extra.get(match.candidate_id)
            if not isinstance(suggestion, dict) or not suggestion:
                continue
            proposal = dict(match.proposal or {})
            medical = proposal.get("generator", "").startswith(MEDICAL_GENERATOR)
            for field in ("key", "label", "description", "kind", "group", "data_type", "generator"):
                value = suggestion.get(field)
                if isinstance(value, str) and value.strip() and not (medical and field in {"generator", "kind", "data_type"}):
                    proposal[field] = value.strip()
            proposal["assistant"] = True
            updates[match.candidate_id] = replace(updates.get(match.candidate_id, match), proposal=proposal)
    return [updates.get(match.candidate_id, match) for match in matches]


def _custom_definitions(definitions: Any) -> list[dict]:
    entries = definitions.values() if isinstance(definitions, dict) else definitions or ()
    approved = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("status", "approved") != "approved":
            continue
        try:
            definition = validate_custom_definition(*(entry[name] for name in
                                                      ("key", "label", "description", "data_type", "generator")))
        except (KeyError, TypeError, ValueError):
            continue
        approved.append(definition)
    return approved


def _custom_kind_fits(candidate: FieldCandidate, definition: dict) -> bool:
    data_type = definition["data_type"]
    if candidate.kind == "text":
        return True
    if candidate.kind == "digit_sequence":
        return data_type == "date" and len(candidate.elements) in {6, 8} or \
            data_type == "time" and len(candidate.elements) == 4 or \
            data_type == "integer" and bool(candidate.elements)
    if candidate.kind == "checkbox" and len(candidate.elements) >= 2:
        generator = definition["generator"]
        if generator["type"] != "choices" or data_type not in {"text", "integer", "boolean"}:
            return False
        options = {normalize(element.get(name)) for element in candidate.elements for name in ("label", "code")
                   if element.get(name) is not None}
        values = generator["values"]
        if data_type == "boolean":
            return set(values) == {True, False} and bool(options & {"sim", "yes"}) and bool(options & {"nao", "no"})
        return all(normalize(value) in options for value in values)
    return False


def _match_custom(candidate: FieldCandidate, match: Match, definitions: list[dict]) -> Match:
    if match.status == "matched" or not definitions:
        return match
    label = normalize(candidate.label)
    significant = [word for word in label.split() if word not in STOPWORDS]
    if not label or len(significant) < 2 and len(label) < 5:
        return match
    applicable = [entry for entry in definitions if _custom_kind_fits(candidate, entry)]
    if not applicable:
        return match
    index = NgramIndex((entry["key"], entry["label"]) for entry in applicable)
    ranked = index.scores(label, (entry["key"] for entry in applicable))
    if not ranked:
        return match
    labels = {entry["key"]: normalize(entry["label"]) for entry in applicable}
    exact = [key for key, _ in ranked if label == labels[key]]
    if len(exact) == 1:
        key, score = exact[0], 1.0
    elif len(exact) > 1:
        return match
    else:
        key, score = ranked[0]
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)
        overlap = set(significant) & set(labels[key].split())
        if len(significant) < 2 or score < 0.9 or margin < 0.15 or not overlap:
            return match
    return Match(candidate.id, key, "matched", "custom", round(score, 3), ((key, round(score, 3)),),
                 f"Rótulo associado à definição semântica aprovada “{key}”; revisar antes de publicar.")


def match_candidates(analysis_or_candidates: Any, document: str | None, provider: Any = None, *,
                     min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN,
                     custom_definitions: Any = None) -> list[Match]:
    candidates = list(getattr(analysis_or_candidates, "candidates", analysis_or_candidates))
    document = document_context(document)
    matches = [match_one(candidate, document, min_score, min_margin) for candidate in candidates]
    definitions = _custom_definitions(custom_definitions)
    if definitions:
        matches = [_match_custom(candidate, match, definitions) for candidate, match in zip(candidates, matches)]
    return _assisted(matches, candidates, document, provider) if provider is not None else matches
