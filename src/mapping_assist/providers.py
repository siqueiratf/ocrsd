from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

from src.field_rules import FIELD_TYPES
from src.semantic.custom import CUSTOM_KEY
from src.semantic.registry import KINDS, REGISTRY, VERSION

if TYPE_CHECKING:
    from src.mapping_assist.analysis import FieldCandidate

PROMPT_VERSION = "2"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
NULL_KEYS = {"", "null", "none", "nenhuma", "new", "nova"}
DISAMBIGUATE_PROMPT = (
    "Você classifica rótulos de campos de formulários brasileiros. Para cada item, escolha exatamente uma chave da lista "
    "'options' cujo significado corresponda ao campo, ou null quando nenhuma corresponder. Use a seção, o título da seção e "
    "os vizinhos apenas como contexto. Nunca invente chaves fora de 'options'. Responda somente com JSON no formato "
    '{"answers": [{"id": "...", "key": "chave ou null", "reason": "justificativa curta em português"}]}.')
PROPOSE_PROMPT = (
    "Você sugere categorias novas para campos de formulários brasileiros que não têm chave no registro semântico. Para cada "
    "item proponha uma chave no formato custom.<entidade>.<nome> em snake_case (ex.: custom.death.cause.part_i), rótulo e "
    "descrição em português, "
    f"kind entre {sorted(KINDS)}, data_type entre {list(FIELD_TYPES)} e um gerador descritivo. Nunca invente conteúdo médico: "
    "para causas, diagnósticos, CID ou dados clínicos use o gerador 'requer regra validada (ex.: CID-10)'. Responda somente "
    'com JSON no formato {"proposals": [{"id": "...", "key": "...", "label": "...", "description": "...", "kind": "...", '
    '"group": "...", "data_type": "...", "generator": "..."}]}.')


class AssistantError(RuntimeError):
    pass


class BudgetExceeded(AssistantError):
    pass


@dataclass(frozen=True)
class ProviderStatus:
    available: bool
    provider: str
    model: str | None
    detail: str


class MappingAssistant(Protocol):
    def status(self) -> ProviderStatus: ...

    def disambiguate(self, document: str | None, items: Sequence[FieldCandidate],
                     candidates: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, tuple[str | None, str]]: ...

    def propose_categories(self, document: str | None, items: Sequence[FieldCandidate]) -> dict[str, dict]: ...


class NullAssistant:
    def __init__(self, detail: str = "Assistente de IA desativado"):
        self.detail = detail

    def status(self) -> ProviderStatus:
        return ProviderStatus(False, "none", None, self.detail)

    def disambiguate(self, document, items, candidates) -> dict[str, tuple[str | None, str]]:
        return {}

    def propose_categories(self, document, items) -> dict[str, dict]:
        return {}


def is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname
    if host == "localhost":
        return True
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_json_object(content: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", str(content or ""), flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise AssistantError("resposta sem objeto JSON")
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise AssistantError("resposta com JSON inválido") from exc
    if not isinstance(value, dict):
        raise AssistantError("resposta JSON não é um objeto")
    return value


def _item(candidate: FieldCandidate) -> dict:
    options = [str(element.get("label") or element.get("code") or "") for element in candidate.elements if candidate.kind == "checkbox"]
    return {"id": candidate.id, "label": candidate.label, "section": candidate.section, "section_title": candidate.section_title,
            "neighbors": list(candidate.neighbors[:6]), "kind": candidate.kind, "boxes": len(candidate.elements),
            **({"options": [option for option in options if option]} if options else {})}


def _text(value: Any, limit: int) -> str | None:
    return value.strip()[:limit] if isinstance(value, str) and value.strip() else None


class OpenAICompatibleAssistant:
    provider = "openai"

    def __init__(self, base_url: str, model: str, *, timeout: float = 60.0, api_key: str | None = None,
                 allow_remote: bool = False, cache_dir: Path | str | None = None, connect_timeout: float = 2.0,
                 batch_size: int = 20, failure_limit: int = 3, cooldown: float = 300.0,
                 clock: Callable[[], float] = time.monotonic):
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("Endereço do assistente de IA inválido")
        if not is_loopback(base_url):
            if not allow_remote:
                raise ValueError("O assistente de IA só aceita endereços locais (loopback); habilite OCRSD_AI_ALLOW_REMOTE para outro host")
            if parts.scheme != "https":
                raise ValueError("Hosts remotos do assistente de IA exigem https")
        self.base_url, self.model, self.timeout, self.api_key = base_url.rstrip("/"), model, float(timeout), api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.connect_timeout, self.batch_size = connect_timeout, max(1, min(int(batch_size), 20))
        self.failure_limit, self.cooldown, self.clock = failure_limit, cooldown, clock
        self.failures, self.open_until, self.requests = 0, 0.0, 0
        self.deadline: float | None = None
        self.schema_unsupported = False
        self._lock = threading.Lock()

    def _paused(self) -> bool:
        with self._lock:
            return self.clock() < self.open_until

    def _record(self, success: bool) -> None:
        with self._lock:
            if success:
                self.failures = 0
                return
            self.failures += 1
            if self.failures >= self.failure_limit:
                self.failures, self.open_until = 0, self.clock() + self.cooldown

    def _http(self, method: str, path: str, body: dict | None = None, timeout: float | None = None, base: str | None = None) -> dict:
        limit = float(timeout or self.timeout)
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise BudgetExceeded("orçamento de tempo do assistente esgotado")
            limit = min(limit, remaining)
        parts = urlsplit(base or self.base_url)
        factory = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
        connection = factory(parts.hostname, parts.port, timeout=min(self.connect_timeout, limit))
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        expired, opened = threading.Event(), []

        def stop() -> None:
            expired.set()
            for sock in opened:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        watchdog = threading.Timer(limit, stop)
        watchdog.daemon = True
        self.requests += 1
        try:
            watchdog.start()
            connection.connect()
            opened.append(connection.sock)
            if expired.is_set():
                raise TimeoutError
            connection.sock.settimeout(limit)
            connection.request(method, f"{parts.path.rstrip('/')}{path}",
                               json.dumps(body).encode() if body is not None else None, headers)
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE_BYTES + 1)
            if expired.is_set():
                raise TimeoutError
            if response.status != 200:
                raise AssistantError(f"o servidor rejeitou a requisição (HTTP {response.status})")
            if len(data) > MAX_RESPONSE_BYTES:
                raise AssistantError("resposta grande demais")
            value = json.loads(data)
            if not isinstance(value, dict):
                raise AssistantError("resposta do servidor não é um objeto JSON")
            return value
        except TimeoutError as exc:
            raise AssistantError("tempo esgotado") from exc
        except ConnectionRefusedError as exc:
            raise AssistantError("conexão recusada pelo servidor") from exc
        except (OSError, http.client.HTTPException) as exc:
            if expired.is_set():
                raise AssistantError("tempo esgotado") from exc
            raise AssistantError(f"servidor indisponível ({type(exc).__name__})") from exc
        except ValueError as exc:
            raise AssistantError("resposta inválida do servidor") from exc
        finally:
            watchdog.cancel()
            connection.close()

    def _cache_path(self, messages: list[dict]) -> Path | None:
        if not self.cache_dir:
            return None
        digest = hashlib.sha256(json.dumps([self.provider, self.base_url, self.model, PROMPT_VERSION, VERSION, messages], sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _chat(self, messages: list[dict]) -> dict:
        path = self._cache_path(messages)
        if path and path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if self._paused():
            raise AssistantError("assistente pausado após falhas consecutivas")
        body = {"model": self.model, "messages": messages, "temperature": 0, "stream": False,
                "response_format": {"type": "json_object"}}
        error: AssistantError | None = None
        for _ in range(2):
            try:
                reply = self._http("POST", "/chat/completions", body)
                choices = reply.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise AssistantError("resposta sem escolhas")
                answer = parse_json_object((choices[0].get("message") or {}).get("content"))
                break
            except BudgetExceeded:
                if error is None:
                    raise
                self._record(False)
                raise error
            except AssistantError as exc:
                error = exc
        else:
            self._record(False)
            raise error or AssistantError("falha desconhecida")
        self._record(True)
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
                    json.dump(answer, handle, ensure_ascii=False)
                os.replace(handle.name, path)
            except OSError:
                pass
        return answer

    def _request_chat(self, body: dict) -> tuple[dict, dict]:
        reply = self._http("POST", "/chat/completions", body)
        choices = reply.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AssistantError("resposta sem escolhas")
        answer = parse_json_object((choices[0].get("message") or {}).get("content"))
        return answer, reply

    def _schema_format(self, campos: list[dict]) -> dict:
        properties: dict[str, dict] = {}
        required: list[str] = []
        for campo in campos:
            key = str(campo.get("chave"))
            required.append(key)
            if campo.get("opcoes"):
                properties[key] = {"type": "string", "enum": [str(item) for item in campo["opcoes"]]}
            elif campo.get("intervalo") is not None:
                low, high = campo["intervalo"]
                schema: dict[str, Any] = {"type": "number" if campo.get("tipo") == "número decimal" else "integer"}
                if low is not None:
                    schema["minimum"] = low
                if high is not None:
                    schema["maximum"] = high
                properties[key] = schema
            elif campo.get("max_caracteres") is not None:
                properties[key] = {"type": "string", "maxLength": int(campo["max_caracteres"])}
            else:
                properties[key] = {"type": "string"}
        return {"type": "json_schema", "json_schema": {"name": "registro", "strict": True, "schema": {
            "type": "object", "properties": {"valores": {"type": "object", "properties": properties,
                                                          "required": required, "additionalProperties": False}},
            "required": ["valores"], "additionalProperties": False}}}

    def complete_json(self, system: str, payload: dict, *, temperature: float, seed: int) -> tuple[dict, dict]:
        if self._paused():
            raise AssistantError("assistente pausado após falhas consecutivas")
        campos = payload.get("campos") if isinstance(payload.get("campos"), list) else None
        use_schema = bool(campos) and not self.schema_unsupported
        body = {"model": self.model, "messages": self._messages(system, payload), "temperature": float(temperature),
                "seed": int(seed), "stream": False,
                "response_format": self._schema_format(campos) if use_schema else {"type": "json_object"}}
        started = time.monotonic()
        try:
            answer, reply = self._request_chat(body)
        except AssistantError:
            if not use_schema:
                self._record(False)
                raise
            self.schema_unsupported = True
            body["response_format"] = {"type": "json_object"}
            try:
                answer, reply = self._request_chat(body)
            except AssistantError:
                self._record(False)
                raise
        self._record(True)
        usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
        return answer, {"model": str(reply.get("model") or self.model), "seconds": round(time.monotonic() - started, 3),
                        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}

    def model_digest(self) -> str | None:
        if not self.base_url.endswith("/v1"):
            return None
        try:
            reply = self._http("GET", "/api/tags", timeout=self.connect_timeout, base=self.base_url[:-3])
        except AssistantError:
            return None
        for item in reply.get("models") or ():
            if isinstance(item, dict) and self.model in (item.get("name"), item.get("model")):
                return str(item.get("digest") or "") or None
        return None

    def _messages(self, system: str, payload: dict) -> list[dict]:
        suffix = " /no_think" if self.model.lower().startswith("qwen3") else ""
        return [{"role": "system", "content": system + suffix},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]

    def status(self) -> ProviderStatus:
        if self._paused():
            return ProviderStatus(False, self.provider, self.model, "Assistente pausado após falhas consecutivas")
        try:
            reply = self._http("GET", "/models", timeout=self.connect_timeout)
        except AssistantError as exc:
            self._record(False)
            return ProviderStatus(False, self.provider, self.model, f"Assistente indisponível: {exc}")
        listed = reply.get("data")
        if not isinstance(listed, list):
            return ProviderStatus(False, self.provider, self.model, "Servidor não informou os modelos disponíveis")
        models = [item["id"] for item in listed if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if self.model not in models:
            return ProviderStatus(False, self.provider, self.model, f"Modelo {self.model} não encontrado no servidor")
        return ProviderStatus(True, self.provider, self.model, f"Assistente local disponível ({self.model})")

    def _batches(self, items: Sequence[FieldCandidate]) -> list[Sequence[FieldCandidate]]:
        return [items[start:start + self.batch_size] for start in range(0, len(items), self.batch_size)]

    def disambiguate(self, document: str | None, items: Sequence[FieldCandidate],
                     candidates: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, tuple[str | None, str]]:
        found: dict[str, tuple[str | None, str]] = {}
        errors: list[str] = []
        pending = [item for item in items if candidates.get(item.id)]
        for batch in self._batches(pending):
            allowed = {item.id: {str(key) for key, _ in candidates[item.id]} for item in batch}
            payload = {"document": document, "items": [
                {**_item(item), "options": [{"key": key, "label": REGISTRY[key].label, "description": REGISTRY[key].description,
                                             "score": round(float(score), 3)}
                                            for key, score in candidates[item.id] if key in REGISTRY]} for item in batch]}
            try:
                answer = self._chat(self._messages(DISAMBIGUATE_PROMPT, payload))
            except AssistantError as exc:
                errors.append(str(exc))
                if isinstance(exc, BudgetExceeded):
                    break
                continue
            for row in answer.get("answers") or []:
                if not isinstance(row, dict) or str(row.get("id")) not in allowed or str(row.get("id")) in found:
                    continue
                identifier, key = str(row["id"]), row.get("key")
                key = None if key is None or (isinstance(key, str) and key.strip().lower() in NULL_KEYS) else key
                if key is not None and (not isinstance(key, str) or key not in allowed[identifier]):
                    continue
                found[identifier] = (key, _text(row.get("reason"), 240) or "sem justificativa")
        if errors and not found:
            raise AssistantError(errors[-1])
        return found

    def propose_categories(self, document: str | None, items: Sequence[FieldCandidate]) -> dict[str, dict]:
        found: dict[str, dict] = {}
        errors: list[str] = []
        for batch in self._batches(list(items)):
            identifiers = {item.id for item in batch}
            try:
                answer = self._chat(self._messages(PROPOSE_PROMPT, {"document": document, "items": [_item(item) for item in batch]}))
            except AssistantError as exc:
                errors.append(str(exc))
                if isinstance(exc, BudgetExceeded):
                    break
                continue
            for row in answer.get("proposals") or []:
                if not isinstance(row, dict) or str(row.get("id")) not in identifiers or str(row.get("id")) in found:
                    continue
                proposal = {name: value for name in ("label", "description", "group", "generator")
                            if (value := _text(row.get(name), 240))}
                key = _text(row.get("key"), 80)
                if key and CUSTOM_KEY.fullmatch(key) and key not in REGISTRY:
                    proposal["key"] = key
                if row.get("kind") in KINDS:
                    proposal["kind"] = row["kind"]
                if row.get("data_type") in FIELD_TYPES:
                    proposal["data_type"] = row["data_type"]
                if proposal:
                    found[str(row["id"])] = proposal
        if errors and not found:
            raise AssistantError(errors[-1])
        return found


def assistant_from_settings(settings: Any, cache_dir: Path | str | None) -> MappingAssistant:
    if getattr(settings, "ai_provider", "none") != "openai":
        return NullAssistant()
    secret = getattr(settings, "ai_api_key", None)
    try:
        return OpenAICompatibleAssistant(settings.ai_base_url, settings.ai_model, timeout=settings.ai_timeout,
                                         api_key=secret.get_secret_value() if secret else None,
                                         allow_remote=settings.ai_allow_remote, cache_dir=cache_dir)
    except ValueError as exc:
        return NullAssistant(str(exc))
