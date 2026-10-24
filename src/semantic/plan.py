from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field as dataclass_field
from datetime import date, time
from typing import Any, Callable, Iterable

from src.semantic.present import (IDENTIFIER_KINDS, kind_compatible, matches_pattern, present_text, validation_rules,
                                  value_kind)
from src.semantic.record import SemanticConstraintError, SemanticRecord
from src.semantic.registry import REGISTRY
from src.semantic.resolve import Binding, document_context, normalize, resolve_field

FEASIBILITY_SEEDS = (0, 1, 2)
IDENTIFIER_KEYS = frozenset({"cpf", "cns", "matricula", "dnv_number", "document_number"})
DOMAIN_RULES = frozenset({"allowed_values", "min", "max", "min_value", "max_value"})
MASK_CLASSES = {"#": "[0-9]", "A": "[A-Z]", "a": "[a-z]", "*": "[A-Z0-9]"}
SAMPLES = {
    "date": tuple(date(year, month, day) for year in (1980, 2025) for month in range(1, 13) for day in (1, 9, 10, 28)) + (date(2024, 2, 29),),
    "time": (time(0, 0), time(9, 5), time(12, 34), time(23, 59)),
}


@dataclass(frozen=True)
class Rule:
    key: str
    accepts: Callable[[Any], bool]
    rule: str
    field: str

    @property
    def reason(self) -> str:
        return f"{self.rule} do campo {self.field}"


@dataclass(frozen=True)
class Decision:
    binding: Binding | None = None
    rules: tuple[Rule, ...] = ()
    issue: str | None = None
    invalid: bool = False
    name: str = ""


@dataclass(frozen=True)
class SemanticPlan:
    document: str | None
    decisions: dict[str, Decision] = dataclass_field(default_factory=dict)
    issues: tuple[str, ...] = ()

    @property
    def bindings(self) -> dict[str, Binding]:
        return {field_id: decision.binding for field_id, decision in self.decisions.items() if decision.binding and not decision.issue}

    def apply(self, record: SemanticRecord, extra_keys: Iterable[str] = ()) -> dict[str, Binding]:
        if self.issues:
            raise ValueError(self.issues[0])
        bindings = self.bindings
        for field_id in bindings:
            for rule in self.decisions[field_id].rules:
                record.require(rule.key, rule.accepts, rule.reason)
        try:
            record.resolve([binding.key for binding in bindings.values()] + list(extra_keys))
        except SemanticConstraintError as exc:
            raise ValueError(str(exc)) from exc
        return bindings


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounds(rules: dict) -> tuple[float | None, float | None]:
    low = rules.get("min") if rules.get("min") is not None else rules.get("min_value")
    high = rules.get("max") if rules.get("max") is not None else rules.get("max_value")
    return _number(low), _number(high)


def _span(low: float | None, high: float | None) -> str:
    return f"intervalo {'' if low is None else f'{low:g}'}–{'' if high is None else f'{high:g}'}"


def _format_rule(field: dict, binding: Binding, name: str) -> Rule:
    rules = validation_rules(field)
    stripped = {**{key: value for key, value in field.items() if key not in DOMAIN_RULES},
                "validation": {key: value for key, value in (field.get("validation") or {}).items() if key not in DOMAIN_RULES}}
    data_type = field.get("data_type") or "text"
    allowed = ({normalize(item) for item in rules.get("allowed_values") or []}
               if REGISTRY[binding.key].kind in IDENTIFIER_KEYS else set())
    mask = re.compile("".join(MASK_CLASSES.get(character, re.escape(character)) for character in rules["mask"])) if rules.get("mask") else None
    max_length, pattern = rules.get("max_length"), rules.get("pattern")
    prefix = str(rules.get("prefix") or "") if data_type == "dnv_number" else ""

    def accepts(value) -> bool:
        if value is None:
            return True
        if not kind_compatible(stripped, binding, value):
            return False
        if data_type == "checkbox":
            return True
        text, kind = present_text(field, binding, value), value_kind(binding, value)
        return ((not allowed or normalize(text) in allowed) and (mask is None or mask.fullmatch(text) is not None)
                and (max_length is None or len(text) <= max_length)
                and (not pattern or kind not in IDENTIFIER_KINDS or matches_pattern(pattern, text))
                and (not prefix or kind != "dnv" or str(value).startswith(prefix)))
    return Rule(binding.key, accepts, "formato", name)


def _domain_rules(field: dict, binding: Binding, name: str) -> tuple[Rule, ...]:
    rules = validation_rules(field)
    kind = REGISTRY[binding.key].kind
    found = []
    allowed = rules.get("allowed_values") or []
    if allowed and kind not in IDENTIFIER_KEYS and field.get("data_type") != "checkbox":
        numbers = [_number(item) for item in allowed]
        if kind == "integer" and None not in numbers:
            found.append(Rule(binding.key, lambda value, accepted=frozenset(numbers): value is None or value in accepted,
                              "valores permitidos", name))
        else:
            texts = frozenset(normalize(item) for item in allowed)
            found.append(Rule(binding.key, lambda value: value is None or normalize(present_text(field, binding, value)) in texts,
                              "valores permitidos", name))
    low, high = _bounds(rules)
    if low is not None or high is not None:
        def within(number) -> bool:
            return (low is None or number >= low) and (high is None or number <= high)
        if kind == "integer":
            found.append(Rule(binding.key, lambda value: value is None or within(value), _span(low, high), name))
        elif kind == "date" and binding.part:
            found.append(Rule(binding.key, lambda value: value is None or within(getattr(value, binding.part)), _span(low, high), name))
    return tuple(found)


def _feasible(rules: tuple[Rule, ...], document: str | None) -> str | None:
    if not rules:
        return None
    for seed in FEASIBILITY_SEEDS:
        record = SemanticRecord(seed, document)
        for rule in rules:
            record.require(rule.key, rule.accepts, rule.reason)
        try:
            record.resolve(rule.key for rule in rules)
        except SemanticConstraintError as exc:
            return str(exc)
    return None


@functools.lru_cache(maxsize=4096)
def _decide(encoded: str, document: str | None) -> Decision:
    field = json.loads(encoded)
    name = str(field.get("label") or field.get("description") or field.get("field_id") or "")
    explicit = field.get("semantic") not in (None, "", False)
    try:
        binding = resolve_field(field, document)
    except ValueError as exc:
        return Decision(issue=str(exc), invalid=True, name=name)
    if binding is None:
        return Decision()
    entry = REGISTRY[binding.key]
    shape = _format_rule(field, binding, name)
    if entry.kind in SAMPLES:
        fits, formats = all(shape.accepts(value) for value in SAMPLES[entry.kind]), ()
    else:
        formats = (shape,)
        fits = _feasible(formats, document) is None
    if not fits:
        return Decision(issue=f"o significado “{entry.label}” ({binding.key}) não cabe no formato do campo",
                        name=name) if explicit else Decision()
    domain = _domain_rules(field, binding, name)
    if domain and _feasible(formats + domain, document):
        rules = ", ".join(rule.rule for rule in domain)
        return Decision(binding, formats + domain,
                        f"nenhum valor coerente de “{entry.label}” ({binding.key}) atende às regras do campo ({rules})", name=name)
    return Decision(binding, formats + domain, name=name)


@functools.lru_cache(maxsize=256)
def _plan(encoded: tuple[str, ...], document: str | None) -> SemanticPlan:
    decisions = {str(json.loads(item).get("field_id", "")): _decide(item, document) for item in encoded}
    issues = [f"{decision.name}: {decision.issue}" for decision in decisions.values() if decision.issue]
    rules = tuple(rule for decision in decisions.values() if decision.binding and not decision.issue for rule in decision.rules)
    if not issues and len({rule.field for rule in rules}) > 1 and (failure := _feasible(rules, document)):
        issues.append(f"As regras dos campos são incompatíveis entre si: {failure}")
    return SemanticPlan(document, decisions, tuple(issues))


def semantic_plan(fields: Iterable[dict], document: str | None) -> SemanticPlan:
    return _plan(tuple(json.dumps(item, sort_keys=True, ensure_ascii=False, default=str) for item in fields), document_context(document))
