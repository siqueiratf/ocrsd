"""Content classification for field values.

Chooses a small, generic "content class" for a field based on its
`data_type`, semantic key and the rendered value's shape. Used to pick a
writing style (font/neural pen) per content, independently from the
document type: the same class ("digits", "date", ...) should get the same
treatment everywhere, the actual style being decided by presets.

This module is deliberately data-driven (regexes over key/value) rather
than a per-document lookup table, so it keeps working for new mappings
without edits here.
"""
from __future__ import annotations

import re

CONTENT_CLASSES = (
    "digits", "date", "time", "cpf_cnpj", "name", "place",
    "short_text", "long_text", "checkbox",
)

_KEY_PATTERNS = (
    (re.compile(r"cpf|cnpj|\bcns\b|rg\b|rne\b"), "cpf_cnpj"),
    (re.compile(r"data|nascimento|emissao|validade"), "date"),
    (re.compile(r"\bhora\b|horario"), "time"),
    (re.compile(r"nome|filiac|pai\b|mae\b|conjuge|testemunha|declarante|requerente|socio"), "name"),
    (re.compile(r"endereco|logradouro|municipio|bairro|cidade|uf\b|naturalidade|cartorio"), "place"),
    (re.compile(r"numero|codigo|matricula|protocolo|registro|folha|livro|termo|ddd|cep"), "digits"),
    (re.compile(r"observ|descri|finalidade|objeto|ato\b|texto"), "long_text"),
)

_DIGIT_TYPES = {"digit_sequence"}
_DATE_TYPES = {"date"}
_TIME_TYPES = {"time"}
_TEXT_SEQUENCE_TYPES = {"text_sequence"}

_DATE_VALUE = re.compile(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$")
_TIME_VALUE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_CPF_VALUE = re.compile(r"^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$")
_CNPJ_VALUE = re.compile(r"^\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}$")
_DIGITS_VALUE = re.compile(r"^[\d.\-/\s]+$")


def classify_content(data_type: str | None, semantic_key: str | None, value: str) -> str:
    """Classify a single field value into one of CONTENT_CLASSES.

    `data_type` is the mapping's field data type (e.g. "digit_sequence",
    "date", "text"); `semantic_key` is the semantic binding key when known
    (e.g. "pessoa.cpf", "endereco.municipio"); `value` is the (already
    generated) text that will be rendered.
    """
    data_type = data_type or "text"
    value = value or ""
    key = (semantic_key or "").lower()

    if data_type == "checkbox":
        return "checkbox"

    for pattern, label in _KEY_PATTERNS:
        if pattern.search(key):
            return label

    if data_type in _DATE_TYPES or _DATE_VALUE.match(value):
        return "date"
    if data_type in _TIME_TYPES or _TIME_VALUE.match(value):
        return "time"
    if _CPF_VALUE.match(value) or _CNPJ_VALUE.match(value):
        return "cpf_cnpj"
    if data_type in _DIGIT_TYPES or (value and _DIGITS_VALUE.match(value)):
        return "digits"

    words = value.split()
    if data_type in _TEXT_SEQUENCE_TYPES or len(value) > 40 or len(words) > 6:
        return "long_text"
    return "short_text"
