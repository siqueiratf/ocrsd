from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Mapping

from src.semantic.data import LOCATION_PROFILES, OCCUPATION_CODES
from src.semantic.registry import REGISTRY

ADDRESS_PARTS = {"municipality": "city", "uf": "uf", "ibge": "ibge", "cep": "cep"}
ADDRESS_PREFIXES = ("birth.place", "mother.residence", "deceased.residence", "death.place", "registration.office",
                    "death_registration.office")
SEXES = frozenset({"M", "F"})
REFERENCE_END = date(2025, 12, 31)
EVENT_DATES = ("birth.date", "filling.date", "registration.date", "death.date", "death_registration.date")


@dataclass(frozen=True)
class Constraint:
    name: str
    keys: tuple[str, ...]
    regenerate: tuple[str, ...]
    check: Callable[[Mapping[str, Any]], str | None]
    partial: bool = False

    def applies(self, values: Mapping[str, Any]) -> bool:
        return (any if self.partial else all)(key in values for key in self.keys)

    def violation(self, values: Mapping[str, Any]) -> str | None:
        if not self.applies(values):
            return None
        message = self.check(values)
        return f"Restrição “{self.name}” ({', '.join(self.keys)}): {message}" if message else None


def completed_years(born: date, on: date) -> int:
    return on.year - born.year - ((on.month, on.day) < (born.month, born.day))


def _age_rule(person: str, event: str, title: str) -> Callable[[Mapping[str, Any]], str | None]:
    def check(values: Mapping[str, Any]) -> str | None:
        born, on, age = values[f"{person}.birth_date"], values[event], values[f"{person}.age"]
        if born > on:
            return f"a data de nascimento de {title} ({born.isoformat()}) é posterior a {event} ({on.isoformat()})"
        if age != completed_years(born, on):
            return f"a idade de {title} ({age}) difere dos anos completos em {event} ({completed_years(born, on)})"
        return None
    return check


def _mother_age(values: Mapping[str, Any]) -> str | None:
    if values["mother.birth_date"] >= values["birth.date"]:
        return "a mãe precisa ter nascido antes do nascimento do recém-nascido"
    return _age_rule("mother", "birth.date", "mãe")(values)


def _between(key: str, low: int, high: int, title: str) -> Constraint:
    return Constraint(f"{title}_entre_{low}_e_{high}", (key,), (REGISTRY[key].entity,),
                      lambda values: None if low <= values[key] <= high else f"{key} = {values[key]} fora do intervalo {low}–{high}")


def _filling_window(values: Mapping[str, Any]) -> str | None:
    delay = (values["filling.date"] - values["birth.date"]).days
    return None if 0 <= delay <= 60 else f"o preenchimento ocorreu {delay} dias após o nascimento (permitido: 0 a 60)"


def _last_menstrual_period(values: Mapping[str, Any]) -> str | None:
    expected = values["birth.date"] - timedelta(days=7 * values["pregnancy.weeks"])
    difference = (values["pregnancy.last_menstrual_period"] - expected).days
    return None if abs(difference) <= 6 else f"a DUM difere {difference} dias do esperado para {values['pregnancy.weeks']} semanas"


def _obstetric(values: Mapping[str, Any]) -> str | None:
    prefix = "mother.obstetric."
    pregnancies, vaginal, cesareans, live, losses = (values[prefix + name] for name in (
        "previous_pregnancies", "vaginal_births", "cesareans", "live_births", "fetal_losses"))
    if min(pregnancies, vaginal, cesareans, live, losses) < 0:
        return "os contadores obstétricos não podem ser negativos"
    return None


def _address(prefix: str) -> Constraint:
    keys = tuple(f"{prefix}.{part}" for part in ADDRESS_PARTS if f"{prefix}.{part}" in REGISTRY)

    def check(values: Mapping[str, Any]) -> str | None:
        present = {key: values[key] for key in keys if key in values}
        if any(all(profile[ADDRESS_PARTS[key.rsplit(".", 1)[1]]] == value for key, value in present.items())
               for profile in LOCATION_PROFILES):
            return None
        return "município, UF, código IBGE e CEP não pertencem à mesma localidade: " + ", ".join(
            f"{key}={value}" for key, value in present.items())
    return Constraint(f"endereco_coerente:{prefix}", keys, (REGISTRY[keys[0]].entity,), check, partial=True)


def _sex(key: str) -> Constraint:
    return Constraint(f"sexo_valido:{key}", (key,), (REGISTRY[key].entity,),
                      lambda values: None if values[key] in SEXES else f"{key} precisa ser “M” ou “F”, não {values[key]!r}")


def _parents(values: Mapping[str, Any]) -> str | None:
    expected = f"{values['mother.name']} e {values['father.name']}"
    return None if values["parents.names"] == expected else f"a filiação deveria ser “{expected}”"


def _grandparents(values: Mapping[str, Any]) -> str | None:
    pairs = [pair.split(" e ") for pair in values["grandparents.names"].split("; ")]
    names = [name for pair in pairs for name in pair]
    if len(pairs) != 2 or any(len(pair) != 2 for pair in pairs) or not all(name.strip() for name in names):
        return "os avós precisam seguir o formato “avó e avô maternos; avó e avô paternos”"
    if {values["mother.name"], values["father.name"]} & set(names):
        return "a mãe ou o pai aparecem entre os próprios avós do registrado"
    return None


def _naturalidade(values: Mapping[str, Any]) -> str | None:
    allowed = {values["birth.place.municipality"], values["mother.residence.municipality"]}
    if values["newborn.naturalidade"] in allowed:
        return None
    return f"a naturalidade ({values['newborn.naturalidade']}) precisa ser o município do nascimento ou da residência da mãe"


def _matricula(values: Mapping[str, Any]) -> str | None:
    matricula = values["registration.matricula"]
    if len(matricula) != 32 or not matricula.isdigit():
        return "a matrícula precisa ter 32 algarismos"
    if matricula[10:14] != f"{values['registration.date'].year:04d}":
        return f"o ano da matrícula ({matricula[10:14]}) difere do ano do registro ({values['registration.date'].year})"
    if matricula[14] != "1":
        return "a matrícula de nascimento usa o livro A (tipo 1)"
    if values["registration.number"] != str(int(matricula[23:30])):
        return "o número do registro precisa coincidir com o termo da matrícula"
    return None


def _until_reference(key: str) -> Constraint:
    return Constraint(f"data_ate_referencia:{key}", (key,), (REGISTRY[key].entity,),
                      lambda values: None if values[key] <= REFERENCE_END
                      else f"{key} ({values[key].isoformat()}) é posterior à data de referência {REFERENCE_END.isoformat()}")


def surnames_of(name: str | None) -> set[str]:
    return set(str(name or "").split()[1:])


def _guardian(values: Mapping[str, Any]) -> str | None:
    name, age = values["guardian.name"], values["guardian.age"]
    if (name is None) != (age is None):
        return "nome e idade do responsável legal precisam estar ambos preenchidos ou ambos vazios"
    if age is not None and age < 18:
        return f"o responsável legal tem {age} anos"
    return None


def _twin(values: Mapping[str, Any]) -> str | None:
    twins, twin = values["birth.twins"], values["twin.name_matricula"]
    if bool(twins) != (twin is not None):
        return "o nome e a matrícula do gêmeo só existem quando o nascimento é gemelar"
    if twin is not None and not surnames_of(twin.split(",")[0]) & surnames_of(values["newborn.name"]):
        return "o gêmeo precisa compartilhar os sobrenomes do recém-nascido"
    return None


def _summary(key: str, parts: tuple[str, ...], template: str) -> Constraint:
    def check(values: Mapping[str, Any]) -> str | None:
        expected = template.format(*(values[part] for part in parts))
        return None if values[key] == expected else f"{key} deveria ser “{expected}”"
    return Constraint(f"resumo_coerente:{key}", (key, *parts), (REGISTRY[key].entity,), check)


CONSTRAINTS: tuple[Constraint, ...] = (
    Constraint("gravidez_e_gemelaridade", ("birth.twins", "birth.pregnancy_type"), ("birth",),
               lambda v: None if v["birth.pregnancy_type"] == ("Dupla" if v["birth.twins"] else "Única") else "tipo de gravidez difere da gemelaridade"),
    Constraint("cesarea_e_via_de_parto", ("birth.delivery_type", "birth.cesarean_before_labor"), ("birth",),
               lambda v: None if v["birth.cesarean_before_labor"] == ("Não se aplica" if v["birth.delivery_type"] == "Vaginal" else "Ignorado") else "indicação de cesárea incompatível com a via de parto"),
    Constraint("anomalia_sem_diagnostico", ("newborn.anomaly_status", "newborn.anomaly_description"), ("newborn",),
               lambda v: None if v["newborn.anomaly_status"] == "Ignorado" and v["newborn.anomaly_description"] is None else "diagnóstico não validado"),
    Constraint("ocupacao_e_cbo", ("mother.occupation", "mother.occupation_cbo"), ("mother",),
               lambda v: None if v["mother.occupation_cbo"] == OCCUPATION_CODES.get(v["mother.occupation"]) else "CBO não corresponde à ocupação documentada"),
    Constraint("idade_da_mae_no_nascimento", ("mother.birth_date", "mother.age", "birth.date"), ("mother",), _mother_age),
    Constraint("idade_do_pai_no_nascimento", ("father.birth_date", "father.age", "birth.date"), ("father",),
               _age_rule("father", "birth.date", "pai")),
    Constraint("responsavel_legal_coerente", ("guardian.name", "guardian.age"), ("guardian",), _guardian),
    Constraint("responsavel_legal_diferente_da_mae", ("guardian.name", "mother.name"), ("guardian",),
               lambda values: None if values["guardian.name"] != values["mother.name"]
               else "o responsável legal não pode ser a própria mãe"),
    Constraint("gemeos_coerentes", ("birth.twins", "twin.name_matricula", "newborn.name"), ("twin",), _twin),
    *(_until_reference(key) for key in EVENT_DATES),
    Constraint("registro_do_obito_apos_obito", ("death_registration.date", "death.date"), ("death_registration",),
               lambda values: None if values["death_registration.date"] >= values["death.date"] else "o registro é anterior ao óbito"),
    Constraint("registro_apos_nascimento", ("registration.date", "birth.date"), ("registration",),
               lambda values: None if values["registration.date"] >= values["birth.date"] else "o registro é anterior ao nascimento"),
    Constraint("preenchimento_ate_60_dias", ("filling.date", "birth.date"), ("filling",), _filling_window),
    Constraint("preenchimento_antes_do_registro", ("filling.date", "registration.date"), ("registration",),
               lambda values: None if values["filling.date"] <= values["registration.date"]
               else "o preenchimento da DNV é posterior ao registro civil"),
    _between("pregnancy.weeks", 22, 42, "semanas_de_gestacao"),
    Constraint("dum_coerente", ("pregnancy.last_menstrual_period", "pregnancy.weeks", "birth.date"), ("pregnancy",),
               _last_menstrual_period),
    _between("pregnancy.prenatal_start_month", 1, 9, "mes_inicio_pre_natal"),
    _between("pregnancy.prenatal_visits", 0, 20, "consultas_pre_natal"),
    Constraint("historico_obstetrico", tuple(f"mother.obstetric.{name}" for name in (
        "previous_pregnancies", "vaginal_births", "cesareans", "live_births", "fetal_losses")), ("mother",), _obstetric),
    *(_address(prefix) for prefix in ADDRESS_PREFIXES),
    Constraint("idade_do_falecido_no_obito", ("deceased.birth_date", "deceased.age", "death.date"), ("deceased",),
               _age_rule("deceased", "death.date", "falecido")),
    _sex("newborn.sex"),
    _sex("deceased.sex"),
    Constraint("filiacao_coerente", ("parents.names", "mother.name", "father.name"), ("parents",), _parents),
    Constraint("avos_coerentes", ("grandparents.names", "mother.name", "father.name"), ("grandparents",), _grandparents),
    Constraint("naturalidade_legal", ("newborn.naturalidade", "birth.place.municipality", "mother.residence.municipality"),
               ("newborn",), _naturalidade),
    Constraint("matricula_coerente", ("registration.matricula", "registration.date", "registration.number"), ("registration",),
               _matricula),
    _summary("birth.place.summary", ("birth.establishment.name", "birth.place.municipality", "birth.place.uf"), "{}, {} - {}"),
    _summary("registration.place.summary", ("registration.office.municipality", "registration.office.uf"), "{} - {}"),
    _summary("death_registration.place.summary", ("death_registration.office.municipality", "death_registration.office.uf"), "{} - {}"),
)


def violations(values: Mapping[str, Any]) -> list[tuple[Constraint, str]]:
    return [(constraint, message) for constraint in CONSTRAINTS if (message := constraint.violation(values))]
