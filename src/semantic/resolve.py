from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from src.semantic.registry import CONTEXTS, REGISTRY

DATE_PARTS = ("day", "month", "year")
EXTENSO = "DD de mês de AAAA"
_ADDRESS = ("street", "number", "complement", "neighborhood", "cep", "municipality", "uf", "ibge")
KEYS = frozenset(REGISTRY)
EXCLUDED = (
    "apgar", "peso", "comprimento", "perimetro", "raca", "cor", "escolaridade", "serie", "situacao conjugal", "estado civil",
    "funcao", "tipo de documento", "tipo de gravidez", "tipo de parto", "apresentacao", "induzido", "cesarea ocorreu",
    "assistido", "anomalia", "anomalias", "metodo", "cbo", "averbacao", "averbacoes", "anotacao", "anotacoes",
    "livro", "folha", "termo", "causa", "causas", "cid", "assinatura", "observacao", "observacoes", "selo",
)
CONTEXT_EXCLUDED = {
    "declaracao_obito": ("filhos", "gestacao", "gestacoes", "semanas", "gravidez", "parto", "partos", "aborto", "abortos",
                         "fetal", "fetais", "nascidos vivos", "necropsia", "morte", "doenca", "diagnostico", "obstetrica",
                         "puerperio", "gestante"),
}
ROLES = (
    ("filling", ("responsavel pelo preenchimento", "preenchimento", "preenchedor")),
    ("guardian", ("responsavel legal",)),
    ("certifier", ("medico", "medica", "crm", "atestante")),
    ("death", ("obito",)),
    ("deceased", ("falecido", "falecida")),
    ("occurrence", ("ocorrencia", "local de nascimento", "local municipio de nascimento", "local municipio nascimento")),
    ("registration", ("cartorio", "registro")),
    ("residence", ("residencia",)),
    ("newborn", ("recem nascido", "rn", "registrado", "registrada")),
    ("mother", ("parturiente", "mae", "genitora", "materna", "materno")),
    ("father", ("pai", "genitor", "paterna", "paterno")),
)
ATTRIBUTES = (
    ("twin_name_matricula", ("nome e matricula dos gemeos", "nome matricula gemeos")),
    ("twins", ("gemeos",)),
    ("weeks", ("semanas de gestacao",)),
    ("last_menstrual_period", ("ultima menstruacao", "dum")),
    ("previous_pregnancies", ("gestacoes anteriores",)),
    ("vaginal_births", ("partos vaginais",)),
    ("cesareans", ("cesareas",)),
    ("live_births", ("nascidos vivos",)),
    ("fetal_losses", ("perdas fetais", "abortos")),
    ("prenatal_visits", ("consultas de pre natal", "consultas pre natal")),
    ("prenatal_start_month", ("iniciou o pre natal", "inicio do pre natal", "mes de gestacao")),
    ("parents", ("filiacao",)),
    ("grandparents", ("avos",)),
    ("matricula", ("matricula",)),
    ("dnv_number", ("dnv", "declaracao de nascido vivo")),
    ("crm", ("crm",)),
    ("cnes", ("cnes",)),
    ("ibge", ("ibge", "codigo do municipio")),
    ("cpf", ("cpf",)),
    ("cns", ("cartao sus", "cns", "cartao nacional de saude")),
    ("cep", ("cep",)),
    ("issuer", ("orgao emissor",)),
    ("document_number", ("numero do documento", "n do documento")),
    ("naturalidade", ("naturalidade",)),
    ("occupation", ("ocupacao", "profissao")),
    ("complement", ("complemento",)),
    ("neighborhood", ("bairro", "distrito")),
    ("municipality", ("municipio", "cidade")),
    ("uf", ("uf", "unidade da federacao")),
    ("street", ("logradouro", "endereco", "rua", "avenida")),
    ("time", ("hora", "horario")),
    ("date", ("data",)),
    ("age", ("idade",)),
    ("sex", ("sexo",)),
    ("establishment", ("estabelecimento", "hospital", "unidade de saude")),
    ("office", ("cartorio",)),
    ("name", ("nome", "nomes")),
    ("number", ("numero", "n")),
    ("registration_number", ("registro",)),
)
ROLE_FREE = {
    "weeks": "pregnancy.weeks", "last_menstrual_period": "pregnancy.last_menstrual_period",
    "previous_pregnancies": "mother.obstetric.previous_pregnancies", "vaginal_births": "mother.obstetric.vaginal_births",
    "cesareans": "mother.obstetric.cesareans", "live_births": "mother.obstetric.live_births",
    "fetal_losses": "mother.obstetric.fetal_losses", "prenatal_visits": "pregnancy.prenatal_visits",
    "prenatal_start_month": "pregnancy.prenatal_start_month", "parents": "parents.names",
    "grandparents": "grandparents.names", "matricula": "registration.matricula", "dnv_number": "registration.dnv_number",
    "crm": "certifier.registration", "twins": "birth.twins", "twin_name_matricula": "twin.name_matricula",
}
TYPE_ATTRIBUTES = {"sex": "sex", "dnv_number": "dnv_number", "cep": "cep", "cns": "cns", "occupation": "occupation",
                   "uf": "uf", "municipality": "municipality", "person_name": "name", "date": "date", "time": "time"}
PERSON_ATTRIBUTES = {
    "newborn": ("name", "sex", "cpf", "naturalidade"),
    "mother": ("name", "age", "cpf", "cns", "occupation", "naturalidade"),
    "father": ("name", "age"),
    "guardian": ("name", "age"),
    "deceased": ("name", "sex", "age", "cpf", "occupation"),
}
BLOCK_ROLES = {
    "dnv": {"I": "newborn", "II": "occurrence", "III": "mother", "IV": "guardian", "V": "pregnancy", "VI": None,
            "VII": "filling", "VIII": "registration"},
    "declaracao_obito": {"I": "deceased", "II": "residence", "III": "occurrence", "IV": None, "V": None, "VI": "certifier",
                         "VII": None, "VIII": "registration", "IX": None},
}
DEFAULT_ROLES = {"certidao_nascimento": ("newborn", None), "declaracao_obito": ("deceased", None), "dnv": ("newborn", {"sex"})}


@dataclass(frozen=True)
class Binding:
    key: str
    part: str | None = None
    format: str | None = None
    source: str = "inferred"


def normalize(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    plain = "".join(character for character in decomposed if not unicodedata.combining(character)).lower()
    return re.sub(r"[^a-z0-9]+", " ", plain).strip()


def _has(text: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any(f" {phrase} " in padded for phrase in phrases)


def excluded_terms(document: str | None) -> tuple[str, ...]:
    return EXCLUDED + CONTEXT_EXCLUDED.get(document_context(document) or "", ())


def _block(field: dict) -> str:
    return str(field.get("bloco") or "").strip().upper()


def document_context(document: str | None) -> str | None:
    text = normalize(document)
    if not text:
        return None
    if text.startswith("dnv") or "nascido vivo" in text:
        return "dnv"
    if text == "do" or "obito" in text:
        return "declaracao_obito"
    if "certidao" in text and "nascimento" in text:
        return "certidao_nascimento"
    return text.replace(" ", "_")


def explicit_binding(semantic: object) -> Binding:
    spec = {"key": semantic} if isinstance(semantic, str) else semantic
    if not isinstance(spec, dict) or not isinstance(spec.get("key"), str):
        raise ValueError("O vínculo semântico precisa ser uma chave ou um objeto com “key”")
    if spec["key"] not in KEYS:
        raise ValueError(f"Chave semântica desconhecida: {spec['key']}")
    if spec.get("part") not in (None, *DATE_PARTS):
        raise ValueError("A parte da data precisa ser day, month ou year")
    if spec.get("format") is not None and not isinstance(spec["format"], str):
        raise ValueError("O formato de apresentação precisa ser um texto")
    return Binding(spec["key"], spec.get("part"), spec.get("format"), "explicit")


def _attribute(text: str, field: dict) -> str | None:
    if _has(text, ("data",)) and _has(text, ("hora", "horario")):
        validation = field.get("validation") or {}
        if field.get("data_type") == "time":
            return "time"
        if field.get("data_type") == "date" or validation.get("format") in {"DDMMYYYY", "DDMMAAAA"}:
            return "date"
        return "ambiguous"
    if _has(text, ("municipio", "cidade")) and _has(text, ("uf", "unidade da federacao")):
        return "summary"
    return next((name for name, phrases in ATTRIBUTES if _has(text, phrases)), None)


def _date_part(text: str) -> str | None:
    found = [part for part, word in zip(DATE_PARTS, ("dia", "mes", "ano")) if _has(text, (word,))]
    return found[0] if len(found) == 1 and not _has(text, ("data",)) else None


def _role(text: str, field: dict, document: str | None, attribute: str) -> str | None:
    role = next((name for name, phrases in ROLES if _has(text, phrases)), None)
    if role is None:
        role = BLOCK_ROLES.get(document or "", {}).get(_block(field))
    if role is None and document in DEFAULT_ROLES:
        default, attributes = DEFAULT_ROLES[document]
        role = default if attributes is None or attribute in attributes else None
    return role


def _event_date(role: str | None, text: str, document: str | None, kind: str) -> str | None:
    birth = _has(text, ("nascimento", "nascer"))
    death = document == "declaracao_obito"
    if role == "deceased" and death and kind == "time" and not birth:
        return "death.time"
    if role in {"mother", "father", "deceased"}:
        return f"{role}.birth_date" if birth and kind == "date" else None
    if role in {"newborn", "occurrence"} and birth and not death:
        return f"birth.{kind}"
    if role == "death" or (role == "occurrence" and death):
        return f"death.{kind}"
    if role in {"filling", "registration"} and kind == "date":
        return "death_registration.date" if death and role == "registration" else f"{role}.date"
    return None


def _key(role: str | None, attribute: str, text: str, document: str | None) -> str | None:
    death = document == "declaracao_obito"
    if attribute in ROLE_FREE:
        return ROLE_FREE[attribute]
    if attribute in {"date", "time"}:
        return _event_date(role, text, document, attribute)
    if role in {"mother", "father"} and death:
        return f"deceased.{role}_name" if attribute == "name" else None
    if attribute in PERSON_ATTRIBUTES.get(role or "", ()):
        return f"{role}.{attribute}"
    if role in {"filling", "certifier"} and attribute == "name":
        return "filling.responsible.name" if role == "filling" else "certifier.name"
    if role == "filling" and attribute in {"document_number", "issuer"}:
        return f"filling.responsible.{attribute}"
    if role == "registration":
        prefix = "death_registration" if death else "registration"
        return {"office": f"{prefix}.office.name", "municipality": f"{prefix}.office.municipality",
                "uf": f"{prefix}.office.uf", "summary": f"{prefix}.place.summary",
                "number": f"{prefix}.number", "registration_number": f"{prefix}.number"}.get(attribute)
    if attribute in {"establishment", "cnes"} and role in {None, "newborn", "occurrence", "death"}:
        if death or role == "death":
            return "death.place.establishment.name" if attribute == "establishment" else None
        return f"birth.establishment.{'name' if attribute == 'establishment' else 'cnes'}"
    if role == "death" or (role == "occurrence" and death):
        return f"death.place.{attribute}" if attribute in {"municipality", "uf", "ibge"} else None
    if role == "occurrence":
        return f"birth.place.{attribute}"
    if role in {"residence", "mother", "deceased"} and attribute in _ADDRESS:
        owner = "deceased" if role == "deceased" or (role == "residence" and death) else "mother"
        return f"{owner}.residence.{attribute}"
    return None


def resolve_field(field: dict, document: str | None = None) -> Binding | None:
    if field.get("semantic") not in (None, ""):
        return explicit_binding(field["semantic"])
    document = document_context(document)
    label, description = str(field.get("label") or ""), str(field.get("description") or "")
    text = normalize(" ".join((str(field.get("field_id") or ""), label, description)))
    blocks = BLOCK_ROLES.get(document or "", {})
    if not text or _has(text, excluded_terms(document)) or (_block(field) in blocks and blocks[_block(field)] is None):
        return None
    segments = [normalize(value.rsplit(" - ", 1)[1]) for value in (description, label) if " - " in value]
    attribute = next((found for found in (_attribute(segment, field) for segment in segments) if found), None)
    attribute = attribute or _attribute(text, field)
    part = None
    if attribute is None and (part := _date_part(text)):
        attribute = "date"
    attribute = attribute or TYPE_ATTRIBUTES.get(str(field.get("data_type")))
    if attribute in {None, "ambiguous"}:
        return None
    if field.get("data_type") == "checkbox" and attribute != "sex":
        return None
    key = _key(_role(text, field, document, attribute), attribute, text, document)
    if key not in KEYS or (document in CONTEXTS and document not in REGISTRY[key].contexts):
        return None
    if field.get("data_type") == "date":
        part = None
    extenso = attribute == "date" and not part and "por extenso" in normalize(f"{text} {field.get('notes') or ''}")
    return Binding(key, part, EXTENSO if extenso else None)
