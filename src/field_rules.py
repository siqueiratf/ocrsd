from __future__ import annotations

import random
import re
import string
from datetime import date

FIELD_TYPES = (
    "text", "person_name", "enum", "uf", "integer", "decimal", "date", "time", "cep", "cns",
    "checkbox", "digit_sequence", "text_sequence", "municipality", "address", "occupation",
    "document", "signature", "sex", "dnv_number",
)
COMPOSITE_TYPES = frozenset({"checkbox", "digit_sequence", "text_sequence"})
TEXT_TYPES = frozenset({"text", "person_name", "enum", "uf", "municipality", "address", "occupation", "signature", "sex"})
SEPARATED_DATE = "SEPARADO"
DATE_FORMATS = {
    "DD/MM/AAAA": "%d/%m/%Y",
    "DD-MM-AAAA": "%d-%m-%Y",
    "DD/MM/AA": "%d/%m/%y",
    "AAAA-MM-DD": "%Y-%m-%d",
    "DDMMAAAA": "%d%m%Y",
    "DDMMYYYY": "%d%m%Y",
    "DD de mês de AAAA": None,
    SEPARATED_DATE: "%d/%m/%Y",
}
DATE_FORMAT_LABELS = {"DD/MM/AAAA": "DD/MM/AAAA · 15/09/2026", "DD-MM-AAAA": "DD-MM-AAAA · 15-09-2026",
                      "DD de mês de AAAA": "Por extenso · 15 de setembro de 2026", "DD/MM/AA": "DD/MM/AA · 15/09/26",
                      "AAAA-MM-DD": "AAAA-MM-DD · 2026-09-15", "DDMMAAAA": "DDMMAAAA · 15092026",
                      SEPARATED_DATE: "Dia, mês e ano em áreas separadas"}
DATE_PARTS = {"day": ("Dia", "%d"), "month": ("Mês", "%m"), "year": ("Ano", "%Y")}
SEX_FORMATS = {"completo": "Por extenso · Masculino ou Feminino", "abreviado": "Abreviado · M ou F"}
SEX_VALUES = {"M": "Masculino", "F": "Feminino"}
DNV_FORMATS = {"agrupado": "Agrupado · 00-00000000-0", "digitos": "Somente algarismos · 00000000000",
               "caixas": "Um algarismo por caixa"}
DNV_TEST_PREFIX = "00"
DNV_PATTERN = re.compile(r"^(\d{2})-(\d{8})-(\d)$")
MONTHS = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro",
          "outubro", "novembro", "dezembro")
DOCUMENT_KINDS = {"cpf": "CPF", "rg": "RG", "cns": "Cartão Nacional de Saúde"}
MASK_TOKENS = {"#": string.digits, "A": string.ascii_uppercase, "a": string.ascii_lowercase,
               "*": string.ascii_uppercase + string.digits}
MAX_MASK_LENGTH = 60


def format_date(value: date, fmt: str | None) -> str:
    if fmt == "DD de mês de AAAA":
        return f"{value.day:02d} de {MONTHS[value.month - 1]} de {value.year}"
    return value.strftime(DATE_FORMATS.get(fmt or "DD/MM/AAAA") or "%d/%m/%Y")


def sex_text(code: str, fmt: str | None) -> str:
    return code if fmt == "abreviado" else SEX_VALUES[code]


def format_dnv(digits: str, fmt: str | None = "agrupado") -> str:
    return digits if fmt in {"digitos", "caixas"} else f"{digits[:2]}-{digits[2:10]}-{digits[10:]}"


def dnv_metadata(digits: str) -> dict:
    return {"kind": "dnv", "prefix": digits[:2], "sequence": digits[2:10], "check_digit": digits[10:], "official": False,
            "synthetic": True, "check_digit_rule": "não calculado: o manual oficial do SINASC não publica o algoritmo"}


def dnv_structure_errors(value: str) -> list[str]:
    if not DNV_PATTERN.fullmatch(value):
        return ["O número da DNV precisa ter 11 algarismos no formato XX-XXXXXXXX-X"]
    return []


def masked_value(mask: str, rng: random.Random) -> str:
    return "".join(rng.choice(MASK_TOKENS[character]) if character in MASK_TOKENS else character for character in mask)


def cpf(rng: random.Random) -> str:
    digits = [rng.randrange(10) for _ in range(9)]
    for length in (9, 10):
        total = sum(value * weight for value, weight in zip(digits, range(length + 1, 1, -1)))
        digits.append(0 if total % 11 < 2 else 11 - total % 11)
    text = "".join(map(str, digits))
    return f"{text[:3]}.{text[3:6]}.{text[6:9]}-{text[9:]}"


def document_number(kind: str, rng: random.Random) -> str:
    if kind == "rg":
        return masked_value("##.###.###-#", rng)
    if kind == "cns":
        return masked_value("###############", rng)
    return cpf(rng)


def rule_errors(data_type: str, rules: object) -> list[str]:
    if rules in (None, {}):
        return []
    if not isinstance(rules, dict):
        return ["As validações precisam ser um objeto"]
    errors = []
    allowed = rules.get("allowed_values")
    if allowed is not None and (not isinstance(allowed, list) or len(allowed) > 500
                                or any(not isinstance(item, (str, int, float)) or len(str(item)) > 160 for item in allowed)):
        errors.append("Os valores permitidos precisam ser uma lista de até 500 itens curtos")
    for low, high in (("min_value", "max_value"), ("min", "max")):
        values = [rules.get(low), rules.get(high)]
        if any(value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))) for value in values):
            errors.append("Os limites numéricos precisam ser números")
        elif None not in values and values[0] > values[1]:
            errors.append("O valor mínimo não pode ser maior que o máximo")
    mask = rules.get("mask")
    if mask is not None and (not isinstance(mask, str) or not 0 < len(mask) <= MAX_MASK_LENGTH
                             or not any(character in MASK_TOKENS for character in mask)):
        errors.append("A máscara precisa ter até 60 caracteres e usar #, A, a ou *")
    if data_type == "date" and rules.get("format") is not None and rules.get("format") not in DATE_FORMATS:
        errors.append("Formato de data não suportado")
    blank = rules.get("blank_probability")
    if blank is not None and (isinstance(blank, bool) or not isinstance(blank, (int, float)) or not 0 <= blank <= 0.95):
        errors.append("A probabilidade de campo vazio precisa estar entre 0 e 95%")
    decimals = rules.get("decimals")
    if decimals is not None and (isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 6):
        errors.append("As casas decimais precisam estar entre 0 e 6")
    if rules.get("document_kind") is not None and rules.get("document_kind") not in DOCUMENT_KINDS:
        errors.append("Tipo de documento não suportado")
    pattern = rules.get("pattern")
    if pattern is not None:
        try:
            if not isinstance(pattern, str) or len(pattern) > 300:
                raise re.error("invalid")
            re.compile(pattern)
        except re.error:
            errors.append("A expressão regular é inválida")
    if data_type == "sex" and rules.get("format") is not None and rules.get("format") not in SEX_FORMATS:
        errors.append("Representação de sexo não suportada")
    if data_type == "dnv_number":
        if rules.get("format") is not None and rules.get("format") not in DNV_FORMATS:
            errors.append("Formato do número da DNV não suportado")
        prefix = rules.get("prefix")
        if prefix is not None and (not isinstance(prefix, str) or not re.fullmatch(r"\d{2}", prefix)):
            errors.append("O prefixo da DNV precisa ter dois algarismos")
    semantic = rules.get("semantic_key")
    if semantic is not None and (not isinstance(semantic, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", semantic)):
        errors.append("O vínculo semântico aceita até 60 letras, números, “_” ou “-”")
    for key in ("min_length", "max_length", "digit_count", "max_edit_distance"):
        value = rules.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10000):
            errors.append("Os limites de tamanho precisam ser inteiros não negativos")
    return errors
