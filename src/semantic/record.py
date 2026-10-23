from __future__ import annotations

import inspect
import random
from datetime import date, time
from typing import Any, Callable, Iterable, Sequence

from src.semantic.constraints import violations
from src.semantic.generators import GENERATORS
from src.semantic.registry import ENTITY_KEYS, REGISTRY, semantic_field


class SemanticConstraintError(ValueError):
    def __init__(self, violations: list[str], attempts: int = 0):
        self.violations = list(violations)
        prefix = (f"Não foi possível gerar um registro sintético coerente após {attempts} tentativas" if attempts
                  else "O registro sintético não atende às restrições semânticas")
        super().__init__(f"{prefix}: " + "; ".join(self.violations))


def _json_value(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return value.strftime("%H:%M")
    return value


def stored_value(key: str, value: Any) -> Any:
    kind = semantic_field(key).kind
    if value is None:
        return None
    if kind == "date":
        if isinstance(value, date):
            return value
        if not isinstance(value, str):
            raise ValueError(f"{key} precisa ser uma data AAAA-MM-DD")
        return date.fromisoformat(value)
    if kind == "time":
        if isinstance(value, time):
            return value
        if not isinstance(value, str) or len(value) != 5:
            raise ValueError(f"{key} precisa ser um horário HH:MM")
        return time.fromisoformat(value)
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} precisa ser um número inteiro")
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{key} precisa ser verdadeiro ou falso")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{key} precisa ser um texto")
    return value


class SemanticRecord:
    def __init__(self, seed: int, context: str | None = None, name_sex_association: float = 0.95, max_attempts: int = 25):
        if not 0 <= name_sex_association <= 1:
            raise ValueError("A associação estatística entre nome e sexo precisa estar entre 0 e 1")
        if max_attempts < 1:
            raise ValueError("O número máximo de tentativas precisa ser positivo")
        self.seed = int(seed)
        self.context = context
        self.name_sex_association = float(name_sex_association)
        self.max_attempts = int(max_attempts)
        self._values: dict[str, Any] = {}
        self._attempts: dict[str, int] = {}
        self._deps: dict[str, set[str]] = {}
        self._stack: list[str] = []
        self._requires: dict[str, list[tuple[Any, str]]] = {}
        self._links: dict[str, Any] = {}
        self._frozen: set[str] = set()
        self.provenance: dict | None = None

    @classmethod
    def from_values(cls, values: dict[str, Any], context: str | None, seed: int = 0,
                    allow_partial: bool = False) -> SemanticRecord:
        record = cls(seed, context)
        problems = []
        for key, value in values.items():
            if key.startswith("custom."):
                continue
            if key not in REGISTRY:
                problems.append(f"Chave semântica desconhecida: {key}")
                continue
            try:
                record._values[key] = stored_value(key, value)
            except ValueError as exc:
                problems.append(str(exc))
        if problems:
            raise SemanticConstraintError(problems)
        record._frozen = ({semantic_field(key).entity for key in record._values} if allow_partial else
                          {entity for entity, keys in ENTITY_KEYS.items() if keys and all(key in record._values for key in keys)})
        partial = {semantic_field(key).entity for key in record._values} - record._frozen
        if partial:
            raise SemanticConstraintError([f"Registro incompleto para a entidade {entity}: faltam "
                                           + ", ".join(key for key in ENTITY_KEYS[entity] if key not in record._values)
                                           for entity in sorted(partial)])
        return record

    @property
    def frozen(self) -> bool:
        return bool(self._frozen)

    def value(self, key: str) -> Any:
        entity = semantic_field(key).entity
        if self._stack and self._stack[-1] != entity:
            self._deps[self._stack[-1]].add(entity)
        if key not in self._values:
            self._generate(entity)
        return self._values[key]

    def require(self, key: str, allowed, reason: str) -> None:
        semantic_field(key)
        if isinstance(allowed, (str, bytes)):
            allowed = (allowed,)
        elif not callable(allowed):
            allowed = tuple(allowed)
        self._requires.setdefault(key, []).append((allowed, reason))

    def allows(self, key: str, value: Any) -> bool:
        for allowed, _ in self._requires.get(key, ()):
            try:
                if not (allowed(value) if callable(allowed) else value in allowed):
                    return False
            except (TypeError, ValueError, AttributeError):
                return False
        return True

    def pick(self, rng: random.Random, project: str | Callable[[Any], dict[str, Any]], options: Sequence,
             weights: Sequence[float] | None = None) -> Any:
        project = (lambda option, key=project: {key: option}) if isinstance(project, str) else project
        if len(options) and any(key in self._requires for key in project(options[0])):
            kept = [index for index, option in enumerate(options)
                    if all(self.allows(key, value) for key, value in project(option).items())]
            if kept:
                weights = [weights[index] for index in kept] if weights else None
                options = [options[index] for index in kept]
        if weights:
            return rng.choices(options, weights)[0]
        return options[rng.randrange(len(options))]

    def draw(self, key: str, rng: random.Random, sampler: Callable[[], Any], tries: int = 64) -> Any:
        rules = self._requires.get(key)
        if not rules:
            return sampler()
        finite = [allowed for allowed, _ in rules if not callable(allowed)]
        if finite:
            candidates = sorted({repr(value): value for value in finite[0] if self.allows(key, value)}.items())
            if candidates:
                return rng.choice(candidates)[1]
        value = sampler()
        for _ in range(tries - 1):
            if self.allows(key, value):
                break
            value = sampler()
        return value

    def link(self, name: str, factory):
        if name not in self._links:
            accepts_rng = bool(inspect.signature(factory).parameters) if callable(factory) else False
            self._links[name] = factory(random.Random(f"{self.seed}:link:{name}")) if accepts_rng else factory()
        return self._links[name]

    def resolve(self, keys: Iterable[str]) -> dict:
        keys = list(dict.fromkeys(keys))
        for key in keys:
            semantic_field(key)
        targets = keys + [key for key in self._requires if key not in keys]
        failures: list[tuple[str, set[str]]] = []
        if self._frozen:
            for key in targets:
                self.value(key)
            if failures := self._failures():
                raise SemanticConstraintError([message for message, _ in failures], 0)
            return {key: self._values[key] for key in keys}
        for attempt in range(self.max_attempts):
            for key in targets:
                self.value(key)
            failures = self._failures()
            if not failures:
                return {key: self._values[key] for key in keys}
            entities = set().union(*(entities for _, entities in failures))
            if attempt:
                entities |= self._upstream(entities)
            for entity in sorted(entities):
                self._attempts[entity] = self._attempts.get(entity, 0) + 1
            self._discard(entities)
        raise SemanticConstraintError([message for message, _ in failures], self.max_attempts)

    def snapshot(self) -> dict:
        return {key: _json_value(value) for key, value in sorted(self._values.items())}

    def _failures(self) -> list[tuple[str, set[str]]]:
        found = [(message, set(constraint.regenerate)) for constraint, message in violations(self._values)]
        for key, rules in self._requires.items():
            if key not in self._values:
                continue
            value = self._values[key]
            for allowed, reason in rules:
                try:
                    accepted = allowed(value) if callable(allowed) else value in allowed
                except (TypeError, ValueError, AttributeError):
                    accepted = False
                if not accepted:
                    found.append((f"Requisito “{reason}” não atendido para {key} (valor gerado: {_json_value(value)!r})",
                                  {semantic_field(key).entity}))
        return found

    def _generate(self, entity: str) -> None:
        if entity in self._frozen:
            raise RuntimeError(f"A entidade {entity} do registro armazenado não pode ser gerada novamente")
        if entity in self._stack:
            raise RuntimeError(f"Dependência circular entre entidades semânticas: {' -> '.join(self._stack + [entity])}")
        attempt = self._attempts.get(entity, 0)
        rng = random.Random(f"{self.seed}:{entity}:{attempt}")
        self._deps[entity] = set()
        self._stack.append(entity)
        try:
            produced = GENERATORS[entity](self, rng)
        finally:
            self._stack.pop()
        missing = set(ENTITY_KEYS[entity]) - set(produced)
        if missing:
            raise RuntimeError(f"O gerador de {entity} não produziu: {', '.join(sorted(missing))}")
        self._values.update({key: produced[key] for key in ENTITY_KEYS[entity]})

    def _upstream(self, entities: set[str]) -> set[str]:
        found, pending = set(), list(entities)
        while pending:
            for dependency in self._deps.get(pending.pop(), ()):
                if dependency not in found:
                    found.add(dependency)
                    pending.append(dependency)
        return found

    def _discard(self, entities: set[str]) -> None:
        pending = set(entities)
        while extra := {entity for entity, deps in self._deps.items() if deps & pending and entity not in pending}:
            pending |= extra
        for entity in pending:
            self._deps.pop(entity, None)
            for key in ENTITY_KEYS[entity]:
                self._values.pop(key, None)
