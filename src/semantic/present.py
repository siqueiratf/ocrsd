from __future__ import annotations

import re
from datetime import date, time

from src.field_rules import DATE_FORMATS, DATE_PARTS, SEX_FORMATS, SEX_VALUES, format_date, format_dnv, sex_text
from src.semantic.resolve import Binding, normalize

CATEGORICAL_TYPES = frozenset({"enum", "uf", "sex"})
IDENTIFIER_KINDS = frozenset({"cpf", "cns", "cep", "matricula", "dnv", "digits"})
SUFFIX_KINDS = {"sex": "sex", "cpf": "cpf", "cns": "cns", "cep": "cep", "matricula": "matricula", "dnv_number": "dnv",
                "uf": "uf", "ibge": "digits", "cnes": "digits", "number": "digits", "document_number": "digits", "occupation_cbo": "digits"}
TYPE_KINDS = {
    "date": {"date"}, "time": {"time"}, "sex": {"sex"}, "dnv_number": {"dnv"}, "cep": {"cep"}, "cns": {"cns"},
    "uf": {"uf"}, "integer": {"int", "digits", "date"}, "decimal": {"int"}, "document": {"cpf", "cns", "digits"},
    "person_name": {"text"}, "municipality": {"text"}, "occupation": {"text"}, "address": {"text"}, "signature": {"text"},
}
SEX_PREFIXES = {"M": "masc", "F": "fem"}
BOOL_WORDS = {True: ("sim", "s", "yes", "true", "1", "x"), False: ("nao", "n", "no", "false", "0")}
VALIDATION_KEYS = frozenset({"allowed_values", "digit_count", "pattern", "min", "max", "min_value", "max_value",
                             "allow_digits", "format", "mask"})


def validation_rules(field: dict) -> dict:
    validation = dict(field.get("validation") or {})
    for key, value in field.items():
        if key in VALIDATION_KEYS:
            validation.setdefault(key, value)
    return validation


def _boxes(field: dict) -> int:
    count = len(field.get("elements") or []) or validation_rules(field).get("digit_count") or 0
    return count if isinstance(count, int) and not isinstance(count, bool) else 0


def value_kind(binding: Binding, value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, time):
        return "time"
    if isinstance(value, date):
        return "date"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    kind = SUFFIX_KINDS.get(binding.key.rsplit(".", 1)[-1], "text")
    return "sex" if kind == "sex" and value not in SEX_VALUES else kind


def matches_pattern(pattern: object, text: str) -> bool:
    try:
        return isinstance(pattern, str) and re.fullmatch(pattern, text) is not None
    except re.error:
        return False


def _pick(field: dict, candidates: list[str]) -> str:
    pattern = validation_rules(field).get("pattern")
    return next((candidate for candidate in candidates if matches_pattern(pattern, candidate)), candidates[0])


def format_matricula(digits: str) -> str:
    return (f"{digits[:6]}.{digits[6:8]}.{digits[8:10]}.{digits[10:14]}.{digits[14:15]}.{digits[15:20]}."
            f"{digits[20:23]}.{digits[23:30]}-{digits[30:]}")


def _identifier(kind: str, digits: str) -> list[str]:
    if kind == "cpf":
        return [f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}", digits]
    if kind == "cep":
        return [f"{digits[:5]}-{digits[5:]}", digits]
    if kind == "cns":
        return [digits, f"{digits[:3]} {digits[3:7]} {digits[7:11]} {digits[11:]}"]
    if kind == "matricula":
        return [format_matricula(digits), digits]
    if kind == "dnv":
        return [digits, format_dnv(digits, "agrupado")]
    return [digits]


def _date_part(field: dict, value: date, part: str) -> str:
    number = {"day": value.day, "month": value.month, "year": value.year}[part]
    padded = f"{number:0{4 if part == 'year' else 2}d}"
    boxes = _boxes(field)
    if field.get("data_type") == "digit_sequence" and boxes:
        return padded.zfill(boxes)[-boxes:]
    return _pick(field, [padded, str(number)])


def _sex(field: dict, code: str) -> str:
    validation = validation_rules(field)
    allowed = [str(item) for item in validation.get("allowed_values") or []]
    if field.get("data_type") == "sex":
        return sex_text(code, validation.get("format"))
    if allowed:
        return next((item for item in allowed if _sex_matches(normalize(item), code)), SEX_VALUES[code])
    if field.get("data_type") == "checkbox":
        return ", ".join(str(item.get("label") or item.get("value") or "") for item in _sex_elements(field, code))
    return _pick(field, [sex_text(code, validation.get("format") if validation.get("format") in SEX_FORMATS else None), code])


def _sex_matches(text: str, code: str) -> bool:
    return text.startswith(SEX_PREFIXES[code]) or text == code.lower()


def _bool(field: dict, value: bool) -> str:
    allowed = [str(item) for item in validation_rules(field).get("allowed_values") or []]
    return next((item for item in allowed if normalize(item) in BOOL_WORDS[value]), "Sim" if value else "Não")


def _sex_elements(field: dict, code: str) -> list[dict]:
    return [element for element in field.get("elements") or []
            if any(_sex_matches(normalize(element.get(name)), code) for name in ("value", "label") if element.get(name))]


def present_text(field: dict, binding: Binding, value) -> str:
    data_type = field.get("data_type") or "text"
    validation = validation_rules(field)
    kind = value_kind(binding, value)
    boxes = _boxes(field)
    sequence = data_type == "digit_sequence"
    if kind == "date":
        if binding.part:
            return _date_part(field, value, binding.part)
        if sequence:
            text = value.strftime("%d%m%y" if boxes == 6 else "%d%m%Y")
            return text[:boxes] if boxes else text
        fmt = binding.format or validation.get("format")
        text = format_date(value, fmt if fmt in DATE_FORMATS else None)
        return _pick(field, [text, format_date(value, "EXTENSO_PALAVRAS")]) if fmt == "DD de mês de AAAA" else text
    if kind == "time":
        if sequence:
            return value.strftime("%H%M")[:boxes or None]
        return _pick(field, [value.strftime("%H:%M"), value.strftime("%Hh%M"), value.strftime("%H%M")])
    if kind == "int":
        if sequence and boxes:
            return f"{value:0{boxes}d}"
        return _pick(field, [str(value), f"{value:02d}"])
    if kind == "sex":
        return _sex(field, value)
    if kind == "bool":
        return _bool(field, value)
    text = str(value)
    if kind in IDENTIFIER_KINDS and text.isdigit():
        if sequence:
            return text.zfill(boxes) if boxes else text
        if kind == "dnv" and data_type == "dnv_number":
            return format_dnv(text, validation.get("format"))
        return _pick(field, _identifier(kind, text))
    allowed = [str(item) for item in validation.get("allowed_values") or []]
    return next((item for item in allowed if normalize(item) == normalize(text)), text)


DNV_CHOICES = {
    "birth.pregnancy_type": (("Única", "Dupla", "Tripla ou mais", "Ignorado"), ("1", "2", "3", "9"), "tipo_gravidez"),
    "birth.delivery_type": (("Vaginal", "Cesáreo", "Ignorado"), ("1", "2", "9"), "tipo_parto"),
    "birth.cesarean_before_labor": (("Sim", "Não", "Não se aplica", "Ignorado"), ("1", "2", "3", "9"), "cesarea_ocorreu_antes_do_trabalho_de_parto_iniciar"),
    "newborn.anomaly_status": (("Sim", "Não", "Ignorado"), ("1", "2", "9"), "anomalia"),
}


def checkbox_choice(field: dict, binding: Binding, value) -> list[dict] | None:
    elements = field.get("elements") or []
    if len(elements) == 1 and isinstance(value, bool):
        return elements if value else []
    if binding.key in DNV_CHOICES:
        labels, codes, prefix = DNV_CHOICES[binding.key]
        target = normalize(value)
        index = next((index for index, label in enumerate(labels) if normalize(label) == target), None)
        if index is None:
            return None
        elements = field.get("elements") or []
        matches = [element for element in elements if any(normalize(element.get(name)) in {target, codes[index], normalize(codes[index] + " " + labels[index])}
                   for name in ("label", "value"))]
        if not matches and {str(element.get("element_id")) for element in elements} == {f"{prefix}_{i}" for i in range(len(labels))}:
            matches = [element for element in elements if str(element.get("element_id")) == f"{prefix}_{index}"]
        return matches[:1] or None
    if value_kind(binding, value) == "sex":
        matches = _sex_elements(field, value)
    elif isinstance(value, bool):
        matches = [element for element in field.get("elements") or []
                   if any(normalize(element.get(name)) in BOOL_WORDS[value] for name in ("value", "label"))]
    elif isinstance(value, str) and normalize(value):
        target = normalize(value)
        matches = [element for element in field.get("elements") or []
                   if target in {normalize(element.get("value")), normalize(element.get("label"))}]
    else:
        matches = []
    if not matches:
        return None
    return matches[:1] if field.get("selection_mode", "single") == "single" else matches


def present_elements(field: dict, binding: Binding, value, ordered_elements: list[dict]) -> list[tuple[dict, str]]:
    parts = [element for element in ordered_elements if element.get("part") in DATE_PARTS]
    if parts and value_kind(binding, value) == "date" and not binding.part:
        return [(element, value.strftime(DATE_PARTS[element["part"]][1])) for element in parts]
    text = present_text({**field, "elements": ordered_elements}, binding, value)
    return list(zip(ordered_elements, text))


def kind_compatible(field: dict, binding: Binding, value) -> bool:
    if value is None or value == "":
        return False
    data_type = field.get("data_type") or "text"
    kind = value_kind(binding, value)
    if data_type == "checkbox":
        return checkbox_choice(field, binding, value) is not None
    if binding.part and kind != "date" or kind == "int" and value < 0:
        return False
    boxes = _boxes(field)
    if data_type == "digit_sequence":
        if kind == "date":
            return bool(binding.part) or boxes in {0, 6, 8}
        if kind == "time":
            return boxes in {0, 4}
        if kind == "int":
            return not boxes or len(str(value)) <= boxes
        return kind in IDENTIFIER_KINDS and str(value).isdigit() and (not boxes or len(str(value)) <= boxes)
    text = present_text(field, binding, value)
    if data_type == "text_sequence":
        return kind not in {"date", "time"} and (not boxes or len(text) <= boxes)
    if kind not in TYPE_KINDS.get(data_type, {kind}) or (kind == "date" and data_type == "integer" and not binding.part):
        return False
    validation = validation_rules(field)
    allowed = validation.get("allowed_values") or []
    if allowed and normalize(text) not in {normalize(item) for item in allowed}:
        return data_type not in CATEGORICAL_TYPES and kind in IDENTIFIER_KINDS and matches_pattern(validation.get("pattern"), text)
    return True
