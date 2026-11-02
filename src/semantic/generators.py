from __future__ import annotations

import hashlib
import math
import random
import re
from datetime import date, time, timedelta
from typing import Any, Callable

from src.field_rules import DNV_TEST_PREFIX, cpf, format_date
from src.semantic.constraints import REFERENCE_END, completed_years
from src.semantic.data import (ADULT_ESTABLISHMENT_NAMES, COMPLEMENTS, ESTABLISHMENT_NAMES, FEMALE_ASSOCIATED_FIRST_NAMES,
                               ISSUERS, LOCATION_PROFILES, MALE_ASSOCIATED_FIRST_NAMES, OCCUPATIONS, OCCUPATION_CODES,
                               PROFILES_BY_IBGE, REGISTRY_OFFICE_NAMES, SURNAMES)
from src.semantic.present import format_matricula
from src.semantic.cause_chains import CAUSE_CHAINS, PART2_POOL, format_interval

EVENT_START, EVENT_END = date(1980, 1, 1), REFERENCE_END
MINUTES = tuple(time(minute // 60, minute % 60) for minute in range(24 * 60))

RACES = ("Branca", "Preta", "Amarela", "Parda", "Indígena")
RACE_WEIGHTS = (43, 10, 1, 45, 1)
SCHOOLING_LEVELS = ("Sem escolaridade", "Fundamental I (1ª a 4ª Série)", "Fundamental II (5ª a 8ª Série)",
                    "Médio (antigo 2º grau)", "Superior incompleto", "Superior completo")
SCHOOLING_SERIES_RANGE = {SCHOOLING_LEVELS[1]: (1, 4), SCHOOLING_LEVELS[2]: (5, 8), SCHOOLING_LEVELS[3]: (1, 3)}
# Occupations whose title legally/practically requires a completed degree (M4):
# schooling is derived from occupation for these, not drawn independently.
OCCUPATION_MIN_SCHOOLING = {
    "Engenheira": "Superior completo", "Advogada": "Superior completo", "Professora": "Superior completo",
    "Enfermeiro": "Superior completo", "Técnico em informática": "Médio (antigo 2º grau)",
    "Auxiliar administrativo": "Médio (antigo 2º grau)",
}


def _schooling_for_age(rng: random.Random, age: int, occupation: str | None = None) -> tuple[str, int | None]:
    minimum = OCCUPATION_MIN_SCHOOLING.get(occupation or "")
    if minimum == "Superior completo":
        return "Superior completo", None
    if minimum == "Médio (antigo 2º grau)":
        level = rng.choices(SCHOOLING_LEVELS[3:], (60, 20, 20))[0]
        series_range = SCHOOLING_SERIES_RANGE.get(level)
        return level, rng.randint(*series_range) if series_range else None
    if age < 5:
        return SCHOOLING_LEVELS[0], None
    if age < 7:
        levels, weights = SCHOOLING_LEVELS[:2], (60, 40)
    elif age < 15:
        levels, weights = SCHOOLING_LEVELS[:3], (10, 40, 50)
    else:
        levels, weights = SCHOOLING_LEVELS, (8, 15, 22, 30, 10, 15)
    level = rng.choices(levels, weights[:len(levels)])[0]
    series_range = SCHOOLING_SERIES_RANGE.get(level)
    series = rng.randint(*series_range) if series_range else None
    return level, series


_CAUSE_LINE_LETTERS = ("a", "b", "c", "d")
PLACE_DIGIT = {"1": "9", "2": "9", "3": "0", "4": "4"}  # DO field 20 category -> W-Y 4th-character place digit
# Only W00-Y34 use the 4th character for the place of occurrence; V01-V99 uses it for the
# victim's role/accident mechanism, so the table already stores the complete V code (R3).
_PLACE_DIGIT_CID = re.compile(r"^[WXY][0-9]{2}$")
_UNIT_MINUTES = {"minutos": 1, "horas": 60, "dias": 1440, "semanas": 10080, "meses": 43200, "anos": 525600}
# Scenarios where Part II comorbidities are medically plausible as mere contributors (R2/m-e):
# falls at any age may have chronic comorbidities; natural disease deaths may too. Homicide,
# suicide and traffic deaths get Part II only in the rare case already modelled by the low
# "Não natural" weight below — restricted further to falls so gunshot/stabbing deaths don't
# carry an implausible "Hipertensão arterial" line.
_PART2_ELIGIBLE_EXTERNAL = {"external_fall"}


def _place_coded_cid(base_cid: str, category: str) -> str:
    if _PLACE_DIGIT_CID.fullmatch(base_cid):
        return f"{base_cid}.{PLACE_DIGIT.get(category, '9')}"
    return base_cid


def _ordered_intervals(rng: random.Random, part1: list) -> dict[int, int]:
    """Sample each line's interval value so they are non-decreasing from ``a`` to ``d`` in
    minutes, by construction (R1): walk from ``d`` (longest) down to ``a`` (shortest), capping
    each line's range at the minutes of the line already sampled below it. Lines sharing a
    `group` id (near-simultaneous events) reuse one sampled value."""
    shared: dict[int, int] = {}
    values: dict[int, int] = {}
    cap_minutes: int | None = None
    for index in reversed(range(len(part1))):
        line = part1[index]
        if line is None:
            continue
        if line.group and line.group in shared:
            value = shared[line.group]
        else:
            hi = line.hi
            if cap_minutes is not None:
                hi = max(line.lo, min(line.hi, cap_minutes // _UNIT_MINUTES[line.unit]))
            value = rng.randint(line.lo, hi)
            if line.group:
                shared[line.group] = value
        values[index] = value
        line_minutes = value * _UNIT_MINUTES[line.unit]
        cap_minutes = line_minutes if cap_minutes is None else min(cap_minutes, line_minutes)
    return values


def _render_chain(rng: random.Random, chain, category: str, scenario: str | None) -> dict[str, Any]:
    """Render a chosen CauseChain's Part I/II into the death.cause.* fields.

    Text variants are sampled per record (variety); interval numbers are sampled so they stay
    non-decreasing from ``a`` to ``d`` (R1, see `_ordered_intervals`). W00-Y34 external-cause
    codes get the 4th "place of occurrence" character appended from the DO's own field 20
    (`category`); V01-V99 codes are stored complete in the table (R3). Part II never repeats a
    Part I code/category and is restricted to scenarios where a comorbidity is plausible (R2/m-e).
    """
    part1 = list(chain.part1) + [None] * (4 - len(chain.part1))
    interval_values = _ordered_intervals(rng, part1)
    values: dict[str, Any] = {"death.cause.chain_id": chain.chain_id}
    used_cids = set()
    for index, (letter, line) in enumerate(zip(_CAUSE_LINE_LETTERS, part1)):
        if line is None:
            values[f"death.cause.{letter}_text"] = None
            values[f"death.cause.{letter}_cid"] = None
            values[f"death.cause.{letter}_interval"] = None
            continue
        values[f"death.cause.{letter}_text"] = rng.choice(line.text_variants)
        values[f"death.cause.{letter}_cid"] = _place_coded_cid(line.cid, category)
        values[f"death.cause.{letter}_interval"] = format_interval(line.unit, interval_values[index])
        base = line.cid.split(".")[0]
        used_cids.add(base)  # base ICD-10 category, so E11.2 also excludes E11.9 (same disease)
        if base == "I11":
            used_cids.add("I10")  # hypertensive heart disease already implies the hypertension
    part2_eligible = chain.manner == "Natural" or scenario in _PART2_ELIGIBLE_EXTERNAL
    pool = [line for line in PART2_POOL if line.cid.split(".")[0] not in used_cids] if part2_eligible else []
    rng.shuffle(pool)
    count_options, count_weights = ((0, 1), (90, 10)) if chain.manner == "Não natural" else ((0, 1, 2), (35, 45, 20))
    count = rng.choices(count_options, count_weights)[0] if part2_eligible else 0
    chosen = (pool[:count] + [None, None])[:2]
    values["death.cause.ii1_text"] = chosen[0].text_variants[0] if chosen[0] else None
    values["death.cause.ii1_cid"] = chosen[0].cid if chosen[0] else None
    values["death.cause.ii2_text"] = chosen[1].text_variants[0] if chosen[1] else None
    values["death.cause.ii2_cid"] = chosen[1].cid if chosen[1] else None
    return values


def _weights(low: int, high: int, bands: tuple[tuple[int, float], ...]) -> tuple[float, ...]:
    return tuple(next(weight for limit, weight in bands if value <= limit) for value in range(low, high + 1))


MOTHER_AGES = range(10, 61)
MOTHER_AGE_WEIGHTS = _weights(10, 60, ((12, 0.02), (14, 0.3), (17, 3), (19, 5), (24, 9), (29, 10), (34, 9), (39, 5),
                                       (44, 1.5), (49, 0.2), (60, 0.01)))
FATHER_AGES = range(12, 91)
FATHER_AGE_WEIGHTS = _weights(12, 90, ((15, 0.05), (19, 1.5), (24, 6), (29, 9), (34, 9), (39, 7), (44, 4), (49, 2),
                                       (59, 0.6), (69, 0.1), (90, 0.01)))
DECEASED_AGES = range(0, 111)
DECEASED_AGE_WEIGHTS = _weights(0, 110, ((0, 3), (14, 0.15), (29, 0.4), (49, 0.8), (64, 1.6), (79, 3), (94, 2.2), (110, 0.15)))
SPOUSE_AGES = range(16, 96)
SPOUSE_AGE_WEIGHTS = _weights(16, 95, ((19, 1), (24, 7), (29, 9), (34, 8), (44, 6), (59, 3), (74, 1.2), (95, 0.3)))
WEEKS = range(22, 43)
WEEK_WEIGHTS = _weights(22, 42, ((27, 0.3), (31, 0.8), (33, 1.5), (36, 4), (37, 10), (38, 22), (39, 28), (40, 20), (41, 8),
                                 (42, 1)))
PRENATAL_MONTHS = range(1, 10)
PRENATAL_MONTH_WEIGHTS = (20, 30, 22, 12, 7, 4, 2, 2, 1)
PREGNANCY_COUNTS = range(0, 9)
PREGNANCY_COUNT_WEIGHTS = (20, 22, 18, 14, 10, 7, 4, 3, 2)


class DateSpan:
    def __init__(self, start: date, end: date):
        if end < start:
            raise ValueError(f"Intervalo de datas invertido: {start.isoformat()} a {end.isoformat()}")
        self.start, self.size = start, (end - start).days + 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> date:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.start + timedelta(days=index)


def shift_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def birth_dates_for_age(age: int, on: date) -> DateSpan:
    return DateSpan(shift_years(on, age + 1) + timedelta(days=1), shift_years(on, age))


def stable_digits(width: int, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{int(digest, 16) % 10 ** width:0{width}d}"


def digits(rng: random.Random, width: int) -> str:
    return "".join(str(rng.randrange(10)) for _ in range(width))


def cpf_digits(rng: random.Random) -> str:
    return re.sub(r"\D", "", cpf(rng))


def cns_digits(rng: random.Random) -> str:
    for _ in range(128):
        body = rng.choice("789") + digits(rng, 13)
        check = -sum(int(digit) * weight for digit, weight in zip(body, range(15, 1, -1))) % 11
        if check < 10:
            return body + str(check)
    raise ValueError("Não foi possível produzir um CNS válido após 128 tentativas")


def first_name(record, rng: random.Random, sex: str) -> str:
    associated = rng.random() < record.name_sex_association
    female = (sex == "F") == associated
    return rng.choice(FEMALE_ASSOCIATED_FIRST_NAMES if female else MALE_ASSOCIATED_FIRST_NAMES)


def surnames(rng: random.Random, maternal: str | None = None, paternal: str | None = None,
             inherit: float = 0.97) -> list[str]:
    chosen = [maternal if maternal and rng.random() < inherit else rng.choice(SURNAMES),
              paternal if paternal and rng.random() < inherit else rng.choice(SURNAMES)]
    return list(dict.fromkeys(chosen))


def person_name(record, rng: random.Random, sex: str, family: list[str] | None = None) -> str:
    return " ".join([first_name(record, rng, sex), *(family or surnames(rng))])


def family_names(name: str) -> list[str]:
    return name.split()[1:] or [name]


def parent_names(record, rng: random.Random, child: str) -> tuple[str, str]:
    family = family_names(child)
    mother = person_name(record, rng, "F", surnames(rng, paternal=family[0]))
    father = person_name(record, rng, "M", surnames(rng, paternal=family[-1]))
    return mother, father


def profile_of(record, prefix: str) -> dict:
    return PROFILES_BY_IBGE[record.value(f"{prefix}.ibge")]


def favoring(favored: dict, weight: float) -> list[float]:
    return [weight if profile is favored else (1 - weight) / (len(LOCATION_PROFILES) - 1) for profile in LOCATION_PROFILES]


def pick_profile(record, rng: random.Random, prefix: str, weights: list[float] | None = None,
                 parts: tuple[str, ...] = ("street", "neighborhood", "cep", "municipality", "uf", "ibge")) -> dict:
    fields = {"street": "street", "neighborhood": "neighborhood", "cep": "cep", "municipality": "city", "uf": "uf", "ibge": "ibge"}
    return record.pick(rng, lambda profile: {f"{prefix}.{part}": profile[fields[part]] for part in parts},
                       LOCATION_PROFILES, weights)


def address(prefix: str, profile: dict, number: str, parts: tuple[str, ...]) -> dict[str, Any]:
    fields = {"street": profile["street"], "number": number, "neighborhood": profile["neighborhood"], "cep": profile["cep"],
              "municipality": profile["city"], "uf": profile["uf"], "ibge": profile["ibge"]}
    return {f"{prefix}.{part}": fields[part] for part in parts}


def house_number(record, rng: random.Random, key: str) -> str:
    return record.draw(key, rng, lambda: str(rng.choice((rng.randint(1, 300), rng.randint(1, 3000)))))


def age_and_birth_date(record, rng: random.Random, person: str, on: date, ages: range,
                       weights: tuple[float, ...]) -> tuple[int, date]:
    age = record.pick(rng, f"{person}.age", ages, weights)
    born = record.pick(rng, f"{person}.birth_date", birth_dates_for_age(age, on))
    return completed_years(born, on), born


def age_and_birth_date_in_range(record, rng: random.Random, person: str, on: date, lo: int, hi: int) -> tuple[int, date]:
    """Like age_and_birth_date, but drawing uniformly from [lo, hi] (a chosen chain's own
    age gate) instead of the general population pyramid — used so the DO's deceased age is
    always inside the cause-of-death chain's plausible range, by construction."""
    return age_and_birth_date(record, rng, person, on, range(lo, hi + 1), tuple(1.0 for _ in range(lo, hi + 1)))


def gestational_history(rng: random.Random, pick: Callable, age: int) -> dict[str, int]:
    prefix = "mother.obstetric."
    limit = max(0, (age - 13) // 2 + 1)
    weights = [weight if count <= limit else weight * 0.001 for count, weight in zip(PREGNANCY_COUNTS, PREGNANCY_COUNT_WEIGHTS)]
    gestacoes = pick(rng, prefix + "previous_pregnancies", PREGNANCY_COUNTS, weights)
    perdas = pick(rng, prefix + "fetal_losses", range(0, min(gestacoes, 3) + 1))
    births = max(0, gestacoes - perdas)
    cesareas = pick(rng, prefix + "cesareans", range(0, births + 1))
    nascidos = pick(rng, prefix + "live_births", (births, births - 1) if births else (0,), (3, 1) if births else None)
    return {prefix + "previous_pregnancies": gestacoes, prefix + "vaginal_births": births - cesareas,
            prefix + "cesareans": cesareas, prefix + "live_births": nascidos, prefix + "fetal_losses": perdas}


def birth(record, rng: random.Random) -> dict[str, Any]:
    day = record.pick(rng, "birth.date", DateSpan(EVENT_START, EVENT_END))
    moment = record.pick(rng, "birth.time", MINUTES)
    profile = pick_profile(record, rng, "birth.place")
    establishment = record.pick(rng, "birth.establishment.name", ESTABLISHMENT_NAMES)
    values = {
        "birth.date": day, "birth.time": moment, "birth.establishment.name": establishment,
        "birth.occurrence_type": "Hospital",
        "birth.establishment.cnes": stable_digits(7, "cnes", establishment, profile["ibge"]),
        **address("birth.place", profile, house_number(record, rng, "birth.place.number"),
                  ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge")),
        "birth.place.summary": f"{establishment}, {profile['city']} - {profile['uf']}",
    }
    twins = record.pick(rng, "birth.twins", (False, True), (98, 2))
    delivery = record.pick(rng, "birth.delivery_type", ("Vaginal", "Cesáreo"))
    return {**values, "birth.twins": twins, "birth.pregnancy_type": "Dupla" if twins else "Única",
            "birth.delivery_type": delivery, "birth.cesarean_before_labor": "Não se aplica" if delivery == "Vaginal" else "Ignorado",
            "birth.presentation": "Cefálica", "birth.labor_induced": "Não", "birth.attendant_role": "Médico"}


def mother(record, rng: random.Random) -> dict[str, Any]:
    on = record.value("birth.date")
    age, born = age_and_birth_date(record, rng, "mother", on, MOTHER_AGES, MOTHER_AGE_WEIGHTS)
    name = record.draw("mother.name", rng, lambda: person_name(record, rng, "F"))
    residence = pick_profile(record, rng, "mother.residence", favoring(profile_of(record, "birth.place"), 0.7))
    cities = [profile["city"] for profile in LOCATION_PROFILES]
    naturalidade = record.pick(rng, "mother.naturalidade", cities,
                               [0.55 if city == residence["city"] else 0.45 / (len(cities) - 1) for city in cities])
    occupation = record.pick(rng, lambda value: {"mother.occupation": value, "mother.occupation_cbo": OCCUPATION_CODES.get(value)},
                             (*OCCUPATIONS, *OCCUPATION_CODES))
    schooling = ("Fundamental I (1ª a 4ª série)" if age < 12 else
                 record.pick(rng, "mother.schooling", ("Fundamental II (5ª a 8ª série)", "Médio (antigo 2º grau)",
                                                         "Superior incompleto", "Superior completo")) if age >= 18 else
                 record.pick(rng, "mother.schooling", ("Fundamental I (1ª a 4ª série)", "Fundamental II (5ª a 8ª série)")))
    marital_status = "Solteira(o)" if age < 18 else record.pick(rng, "mother.marital_status", ("Solteira(o)", "Casada(o)", "União estável"))
    return {
        "mother.name": name, "mother.birth_date": born, "mother.age": age,
        "mother.schooling": schooling, "mother.marital_status": marital_status,
        "mother.race": record.pick(rng, "mother.race", ("Branca", "Preta", "Amarela", "Parda", "Indígena"), (38, 10, 1, 49, 2)),
        "mother.cpf": record.draw("mother.cpf", rng, lambda: cpf_digits(rng)),
        "mother.cns": record.draw("mother.cns", rng, lambda: cns_digits(rng)),
        "mother.occupation": occupation, "mother.occupation_cbo": OCCUPATION_CODES.get(occupation),
        "mother.naturalidade": naturalidade,
        **address("mother.residence", residence, house_number(record, rng, "mother.residence.number"),
                  ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge")),
        "mother.residence.complement": record.pick(rng, "mother.residence.complement", COMPLEMENTS),
        **gestational_history(rng, record.pick, age),
    }


def father(record, rng: random.Random) -> dict[str, Any]:
    age, born = age_and_birth_date(record, rng, "father", record.value("birth.date"), FATHER_AGES, FATHER_AGE_WEIGHTS)
    return {"father.name": record.draw("father.name", rng, lambda: person_name(record, rng, "M")),
            "father.birth_date": born, "father.age": age}


def newborn(record, rng: random.Random) -> dict[str, Any]:
    maternal, paternal = family_names(record.value("mother.name"))[-1], family_names(record.value("father.name"))[-1]
    sex = record.pick(rng, "newborn.sex", ("M", "F"), (51, 49))
    family = rng.choices(([maternal, paternal], [maternal]), (85, 15))[0]
    cities = (record.value("birth.place.municipality"), record.value("mother.residence.municipality"))
    weeks = record.value("pregnancy.weeks")
    weight = max(300, min(6000, int(rng.gauss(3150 + (weeks - 39) * 170, 360))))
    length = max(350, min(600, int(rng.gauss(500 + (weeks - 39) * 9, 15))))
    head = max(25, min(40, round(rng.gauss(34 + (weeks - 39) * 0.4, 1.4))))
    apgar_1 = record.pick(rng, "newborn.apgar_1min", range(5, 11), (1, 2, 4, 13, 35, 45))
    values = {
        "newborn.anomaly_status": "Ignorado", "newborn.anomaly_description": None,
        "newborn.race": record.pick(rng, "newborn.race", ("Branca", "Preta", "Amarela", "Parda", "Indígena"), (38, 10, 1, 49, 2)),
        "newborn.weight_g": weight, "newborn.length_tenths_cm": length, "newborn.head_circumference_cm": head,
        "newborn.apgar_1min": apgar_1, "newborn.apgar_5min": record.pick(rng, "newborn.apgar_5min", range(apgar_1, 11)),
        "newborn.name": record.draw("newborn.name", rng, lambda: person_name(record, rng, sex, list(dict.fromkeys(family)))),
        "newborn.sex": sex, "newborn.cpf": record.draw("newborn.cpf", rng, lambda: cpf_digits(rng)),
        "newborn.naturalidade": record.pick(rng, "newborn.naturalidade", cities, (60, 40)),
    }
    values["newborn.naturalidade.uf"] = next(profile["uf"] for profile in LOCATION_PROFILES
                                             if profile["city"] == values["newborn.naturalidade"])
    return values


def guardian(record, rng: random.Random) -> dict[str, Any]:
    if rng.random() >= (0.9 if record.value("mother.age") < 18 else 0.05):
        return {"guardian.name": None, "guardian.age": None}
    center = record.value("mother.age") + 26
    mother_name = record.value("mother.name")
    family = family_names(mother_name)
    weights = [math.exp(-((age - center) / 8) ** 2) + 0.001 for age in range(18, 91)]
    sex = rng.choice(("F", "M"))

    def sample() -> str:
        name = mother_name
        for _ in range(128):
            name = person_name(record, rng, sex, surnames(rng, paternal=rng.choice(family), inherit=0.7))
            if name != mother_name:
                return name
        raise ValueError("Não foi possível gerar um responsável distinto após 128 tentativas")
    return {"guardian.name": record.draw("guardian.name", rng, sample),
            "guardian.age": record.pick(rng, "guardian.age", range(18, 91), weights)}


def twin(record, rng: random.Random) -> dict[str, Any]:
    if not record.value("birth.twins"):
        return {"twin.name_matricula": None}
    newborn_name, matricula = record.value("newborn.name"), record.value("registration.matricula")
    sex = rng.choice(("F", "M"))
    name = newborn_name
    for _ in range(128):
        name = person_name(record, rng, sex, family_names(newborn_name))
        if name != newborn_name:
            break
    else:
        raise ValueError("Não foi possível gerar um gêmeo distinto após 128 tentativas")
    sibling = matricula[:23] + f"{int(matricula[23:30]) % 9999999 + 1:07d}" + digits(rng, 2)
    return {"twin.name_matricula": f"{name}, {format_matricula(sibling)}"}


def parents(record, rng: random.Random) -> dict[str, Any]:
    return {"parents.names": f"{record.value('mother.name')} e {record.value('father.name')}"}


def grandparents(record, rng: random.Random) -> dict[str, Any]:
    pairs = [parent_names(record, rng, record.value(key)) for key in ("mother.name", "father.name")]
    return {"grandparents.names": "; ".join(f"{grandmother} e {grandfather}" for grandmother, grandfather in pairs)}


def pregnancy(record, rng: random.Random) -> dict[str, Any]:
    on = record.value("birth.date")
    weeks = record.pick(rng, "pregnancy.weeks", WEEKS, WEEK_WEIGHTS)
    known = record.pick(rng, "pregnancy.dum_known", (True, False), (70, 30))
    period = (record.pick(rng, "pregnancy.last_menstrual_period",
                          DateSpan(on - timedelta(days=weeks * 7 + 6), on - timedelta(days=weeks * 7))) if known else None)
    start = record.pick(rng, "pregnancy.prenatal_start_month", PRENATAL_MONTHS,
                        [weight if (month - 1) * 4.3 < weeks else weight * 0.001
                         for month, weight in zip(PRENATAL_MONTHS, PRENATAL_MONTH_WEIGHTS)])
    expected = max(0.0, (weeks - (start - 1) * 4.3) / 4.3) * 1.6
    visits = record.pick(rng, "pregnancy.prenatal_visits", range(0, 21),
                         [math.exp(-((count - expected) / 2.5) ** 2) + 0.002 for count in range(0, 21)])
    return {"pregnancy.weeks": weeks, "pregnancy.dum_known": known,
            "pregnancy.reported_weeks": None if known else weeks,
            "pregnancy.estimation_method": None if known else record.pick(rng, "pregnancy.estimation_method", ("Exame físico", "Outro método")),
            "pregnancy.last_menstrual_period": period,
            "pregnancy.prenatal_start_month": start, "pregnancy.prenatal_visits": visits}


def filling(record, rng: random.Random) -> dict[str, Any]:
    on = record.value("birth.date")
    span = DateSpan(on, min(on + timedelta(days=60), EVENT_END))
    day = record.pick(rng, "filling.date", span, [10 if index <= 1 else 3 if index <= 7 else 0.2 for index in range(len(span))])
    sex = rng.choice(("F", "M"))
    return {
        "filling.date": day, "filling.responsible.name": record.draw("filling.responsible.name", rng,
                                                                     lambda: person_name(record, rng, sex)),
        "filling.responsible.role": "Médico", "filling.responsible.document_type": "CRM",
        "filling.responsible.document_number": record.draw("filling.responsible.document_number", rng,
                                                           lambda: f"{rng.randint(100000, 999999):06d}"),
        "filling.responsible.issuer": "CRM",
    }


def registration(record, rng: random.Random) -> dict[str, Any]:
    start = max(record.value("birth.date"), record.value("filling.date"))
    span = DateSpan(start, min(start + timedelta(days=3 * 365), EVENT_END))
    day = record.pick(rng, "registration.date", span,
                      [1 if index <= 15 else 0.3 if index <= 90 else 0.005 for index in range(len(span))])
    born_in, lives_in = profile_of(record, "birth.place"), profile_of(record, "mother.residence")
    weights = [0.6 * (profile is born_in) + 0.35 * (profile is lives_in) + 0.05 / len(LOCATION_PROFILES)
               for profile in LOCATION_PROFILES]
    office = pick_profile(record, rng, "registration.office", weights, ("municipality", "uf"))
    name = record.pick(rng, "registration.office.name", REGISTRY_OFFICE_NAMES)
    term = f"{rng.randint(1, 250000):07d}"
    matricula = "".join((stable_digits(6, "cns", name, office["ibge"]), "01", "55", f"{day.year:04d}", "1",
                         f"{rng.randint(1, 400):05d}", f"{rng.randint(1, 300):03d}", term, digits(rng, 2)))
    return {
        "registration.date": day, "registration.number": str(int(term)), "registration.matricula": matricula,
        "registration.dnv_number": record.draw("registration.dnv_number", rng,
                                               lambda: DNV_TEST_PREFIX + digits(rng, 8) + digits(rng, 1)),
        "registration.office.name": name, "registration.office.municipality": office["city"],
        "registration.office.uf": office["uf"], "registration.place.summary": f"{office['city']} - {office['uf']}",
    }


EXTERNAL_SOURCE_BY_SCENARIO = {
    "external_traffic": "Boletim de Ocorrência", "external_fall": "Boletim de Ocorrência",
    "external_homicide": "Boletim de Ocorrência", "external_suicide": "Família",
}
EXTERNAL_DESCRIPTIONS = {
    ("external_traffic", "hospital"): "Vítima de acidente de trânsito, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("external_traffic", "public"): "Vítima de acidente de trânsito, com óbito no local, em via pública.",
    ("external_fall", "hospital"): "Vítima de queda, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("external_fall", "home_own"): "Vítima de queda no próprio domicílio.",
    ("external_fall", "home_other"): "Vítima de queda em domicílio.",
    ("external_fall", "public"): "Vítima de queda em via pública.",
    ("external_homicide", "public"): "Vítima de agressão física, com óbito no local, em via pública.",
    ("external_homicide", "home_own"): "Vítima de agressão física no próprio domicílio.",
    ("external_homicide", "home_other"): "Vítima de agressão física em domicílio.",
    ("external_homicide", "hospital"): "Vítima de agressão física, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("external_suicide", "home_own"): "Vítima de lesão autoprovocada intencionalmente, encontrada no próprio domicílio.",
    ("external_suicide", "home_other"): "Vítima de lesão autoprovocada intencionalmente, encontrada em domicílio.",
    ("external_suicide", "hospital"): ("Vítima de lesão autoprovocada intencionalmente, socorrida e encaminhada "
                                       "ao hospital, onde veio a óbito."),
    ("external_suicide", "public"): "Vítima de lesão autoprovocada intencionalmente em via pública.",
}
UF_DDD = {"AC": "68", "AL": "82", "AP": "96", "AM": "92", "BA": "71", "CE": "85", "DF": "61", "ES": "27", "GO": "62",
         "MA": "98", "MT": "65", "MS": "67", "MG": "31", "PA": "91", "PB": "83", "PR": "41", "PE": "81", "PI": "86",
         "RJ": "21", "RN": "84", "RS": "51", "RO": "69", "RR": "95", "SC": "48", "SP": "11", "SE": "79", "TO": "63"}


def _place_group(category: str, own_home: bool) -> str:
    if category in ("1", "2"):
        return "hospital"
    if category == "3":
        return "home_own" if own_home else "home_other"
    return "public"


def _own_residence_draw(rng: random.Random) -> bool:
    return rng.random() < 0.85


def death(record, rng: random.Random) -> dict[str, Any]:
    from src.semantic.cause_chains import chains_for_scenario
    from src.semantic.death_scenarios import ADULT_SCENARIOS, SCENARIOS_BY_ID, pick_place, pick_scenario

    is_do = record.context == "declaracao_obito"
    # The template is "Versão 01/10" (2010); restrict DO deaths to the period that version
    # covers (M5) — this also keeps every DO death inside the Cartão Nacional de Saúde era
    # (CNS exists since 1996) without a separate date cutoff for field 3.
    event_start = date(2011, 1, 1) if is_do else EVENT_START
    profile = pick_profile(record, rng, "death.place", parts=("municipality", "uf", "ibge"))

    scenario = chain = None
    category, hospital, external, manner = "1", True, False, "Natural"
    if is_do:
        # When the AI (or any caller) required a specific deceased.age, the scenario/chain must
        # be picked consistently with it (R5): never draw a scenario/chain pair whose age gate
        # excludes the required age, so deceased() (which draws age from the chosen chain's own
        # range) can always satisfy the requirement on the first try.
        required_ages = record.required_values("deceased.age") if hasattr(record, "required_values") else None
        if required_ages:
            target_age = min(required_ages)
            eligible = [(sid, c) for sid in SCENARIOS_BY_ID for c in chains_for_scenario(sid)
                       if c.min_age <= target_age <= c.max_age]
        else:
            eligible = None
        if eligible:
            pair_weights = [SCENARIOS_BY_ID[sid].weight * c.weight for sid, c in eligible]
            scenario, chain = rng.choices(eligible, pair_weights)[0]
        else:
            scenario = record.draw("death.scenario", rng, lambda: pick_scenario(rng).scenario_id)
            # Scenario -> chain FIRST (never crossing scenario or manner — B1/B2), then place
            # consistent with the chosen chain's setting (hospital-only chains force a hospital
            # place; everything else follows the scenario's own place distribution — M1).
            candidates = chains_for_scenario(scenario)
            chain = rng.choices(candidates, weights=[c.weight for c in candidates])[0]
        picked = SCENARIOS_BY_ID[scenario]
        manner = picked.manner
        external = manner == "Não natural"
        if chain.setting == "hospital":
            category = rng.choices(["1", "2"], (90, 10))[0]
        else:
            category = pick_place(rng, picked)
        hospital = category in ("1", "2")

    # Whether a category-"3" (domicílio) death happened in the deceased's OWN residence (~85%,
    # field 22 blank, field 51 says "próprio domicílio") or a different address (~15%, field 22
    # filled with a profile street, field 51 says "em domicílio") — R4. Shared via record.link so
    # deceased() (which renders field 22) and death() (which renders field 51's text) agree.
    own_home = category == "3" and record.link("death.own_residence", _own_residence_draw)

    values: dict[str, Any] = {
        "death.date": record.pick(rng, "death.date", DateSpan(event_start, EVENT_END)),
        "death.time": record.pick(rng, "death.time", MINUTES),
        "death.type": "2",
        "death.place.category": category,
        "death.cause.manner": manner,
        "death.scenario": scenario,
        **address("death.place", profile, "", ("municipality", "uf", "ibge")),
    }

    if is_do:
        # Certifier role / necropsy / medical assistance are one deterministic function of
        # (manner, hospital, assistance, chain.course) — never drawn independently — so external
        # deaths are always IML + necropsy "Sim" (CPP art. 162), natural home deaths without
        # assistance go to the SVO (necropsy "Sim"), and the IML is never used for a natural
        # death, nor the SVO for a hospital death (B4). A home death from a chronic, long-
        # diagnosed chain (neoplasms, Alzheimer, DPOC...) almost always had medical assistance
        # (R2); a home death from a sudden chain (IAM, AVC, HSA...) plausibly did not.
        # certifier() mirrors this same rule to pick the role.
        if external:
            values["death.medical_assistance"] = "1" if hospital else rng.choices(["2", "1"], (92, 8))[0]
            values["death.autopsy"] = "1"
        elif hospital:
            values["death.medical_assistance"] = rng.choices(["1", "2"], (95, 5))[0]
            values["death.autopsy"] = rng.choices(["2", "1"], (97, 3))[0]  # m-d: rare outside the SVO/IML
        elif chain.course == "chronic":
            values["death.medical_assistance"] = rng.choices(["1", "2"], (90, 10))[0]
            values["death.autopsy"] = "1" if values["death.medical_assistance"] == "2" else rng.choices(["2", "1"], (97, 3))[0]
        else:
            values["death.medical_assistance"] = rng.choices(["1", "2"], (30, 70))[0]
            values["death.autopsy"] = "1" if values["death.medical_assistance"] == "2" else rng.choices(["2", "1"], (70, 30))[0]
    else:
        values["death.medical_assistance"] = "1"
        values["death.autopsy"] = "2"

    values["death.place.establishment.name"] = (
        record.pick(rng, "death.place.establishment.name", ADULT_ESTABLISHMENT_NAMES if is_do else ESTABLISHMENT_NAMES)
        if hospital else None)
    values["death.place.establishment.cnes"] = (
        record.draw("death.place.establishment.cnes", rng, lambda: digits(rng, 7)) if hospital else None)

    if external:
        work_hour = values["death.time"].hour
        is_work_accident = (scenario in ("external_traffic", "external_fall") and 7 <= work_hour <= 18
                            and rng.random() < 0.12)
        place_group = _place_group(category, own_home)
        values.update({
            "death.external.type": chain.external_type,
            "death.external.work_accident": "1" if is_work_accident else "2",
            "death.external.source": "Hospital" if hospital else EXTERNAL_SOURCE_BY_SCENARIO[scenario],
            "death.external.description": EXTERNAL_DESCRIPTIONS[(scenario, place_group)],
            "death.external.place.municipality": profile["city"], "death.external.place.uf": profile["uf"],
        })
    else:
        values.update({
            "death.external.type": None, "death.external.work_accident": None, "death.external.source": None,
            "death.external.description": None, "death.external.place.municipality": None, "death.external.place.uf": None,
        })

    if is_do:
        values.update(_render_chain(rng, chain, category, scenario))
    else:
        values.update({f"death.cause.{letter}_{suffix}": None
                       for letter in _CAUSE_LINE_LETTERS for suffix in ("text", "cid", "interval")})
        values.update({"death.cause.chain_id": None, "death.cause.ii1_text": None, "death.cause.ii1_cid": None,
                       "death.cause.ii2_text": None, "death.cause.ii2_cid": None})
    return values


MARITAL_TO_DO = {
    "SOLTEIRO(A)": "Solteiro", "CASADO(A)": "Casado", "VIÚVO(A)": "Viúvo",
    "DIVORCIADO(A)": "Separado judicialmente/divorciado",
    "SEPARADO(A) JUDICIALMENTE": "Separado judicialmente/divorciado", "UNIÃO ESTÁVEL": "União estável",
}


def _DECEASED_MARITAL_WEIGHTS(age: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if age < 18:
        return ("SOLTEIRO(A)",), (1,)
    options = ("SOLTEIRO(A)", "CASADO(A)", "UNIÃO ESTÁVEL", "DIVORCIADO(A)", "SEPARADO(A) JUDICIALMENTE", "VIÚVO(A)")
    weights = (max(1.0, 30 - age / 3), 35, 10, 10, 4, max(0.5, (age - 40) * 1.2))
    return options, weights


def _occurrence_address(record, rng: random.Random, category: str, own_home: bool, profile: dict) -> dict[str, Any]:
    """Field 22/23/24 ("endereço da ocorrência, se fora do estabelecimento OU DA RESIDÊNCIA").

    When the death happened at the deceased's own home (category "3", ~85% of home deaths —
    `own_home`, shared with death()/field 51 via record.link), the DO form wants this block
    BLANK: field 22 is for an occurrence outside the establishment OR the residence, so a death
    in the deceased's own residence never fills it (R4). The remaining ~15% of home deaths (a
    different address, e.g. a relative's home) and every via-pública death (category "4") use
    the SAME address generator as the residence — the death municipality's own profile
    street/CEP/neighborhood (R4) — instead of a 5-item generic street list.
    """
    if category not in ("3", "4") or own_home:
        return {"deceased.occurrence.address": None, "deceased.occurrence.number": None,
               "deceased.occurrence.complement": None, "deceased.occurrence.cep": None,
               "deceased.occurrence.neighborhood": None}
    return {
        "deceased.occurrence.address": profile["street"],
        "deceased.occurrence.number": house_number(record, rng, "deceased.occurrence.number"),
        "deceased.occurrence.complement": (record.pick(rng, "deceased.occurrence.complement", COMPLEMENTS)
                                           if rng.random() < 0.3 else None),
        "deceased.occurrence.cep": profile["cep"], "deceased.occurrence.neighborhood": profile["neighborhood"],
    }


def deceased(record, rng: random.Random) -> dict[str, Any]:
    is_do = record.context == "declaracao_obito"
    on = record.value("death.date")
    if is_do:
        from src.semantic.cause_chains import CAUSE_CHAINS
        chain_id = record.value("death.cause.chain_id")
        chain = next(c for c in CAUSE_CHAINS if c.chain_id == chain_id)
        age, born = age_and_birth_date_in_range(record, rng, "deceased", on, chain.min_age, chain.max_age)
        sex = chain.sex or record.pick(rng, "deceased.sex", ("M", "F"), (52, 48))
    else:
        age, born = age_and_birth_date(record, rng, "deceased", on, DECEASED_AGES, DECEASED_AGE_WEIGHTS)
        sex = record.pick(rng, "deceased.sex", ("M", "F"), (52, 48))
    name = record.draw("deceased.name", rng, lambda: person_name(record, rng, sex))
    mother_name, father_name = parent_names(record, rng, name)
    category = record.value("death.place.category") if is_do else None
    # Home deaths (category "3") must be in the deceased's own residence city (M1) — a death at
    # home is, by definition, in the municipality the person lived in.
    residence_bias = 1.0 if (is_do and category == "3") else 0.7
    residence = pick_profile(record, rng, "deceased.residence", favoring(profile_of(record, "death.place"), residence_bias))
    birthplace = rng.choice(LOCATION_PROFILES)
    # "Estudante" is implausible for an elderly DO death (M4) — exclude it from the pool once
    # the person is well past typical student age in the DO context.
    occupation_pool = tuple(o for o in OCCUPATIONS if o != "Estudante") if (is_do and age >= 65) else OCCUPATIONS
    occupation = "Não se aplica" if age < 14 else record.pick(rng, "deceased.occupation", occupation_pool)
    residence_fields = address("deceased.residence", residence, house_number(record, rng, "deceased.residence.number"),
                               ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge"))
    values = {
        "deceased.name": name, "deceased.sex": sex, "deceased.birth_date": born, "deceased.age": age,
        "deceased.cpf": record.draw("deceased.cpf", rng, lambda: cpf_digits(rng)),
        "deceased.naturalidade": f"{birthplace['city']} - {birthplace['uf']}",
        "deceased.naturalidade.municipality": birthplace["city"], "deceased.naturalidade.uf": birthplace["uf"],
        "deceased.mother_name": record.draw("deceased.mother_name", rng, lambda: mother_name),
        "deceased.father_name": record.draw("deceased.father_name", rng, lambda: father_name),
        "deceased.parents_names": f"{mother_name}; {father_name}",
        "deceased.occupation": occupation, "deceased.occupation_cbo": OCCUPATION_CODES.get(occupation),
        **residence_fields,
    }
    if is_do:
        death_profile = profile_of(record, "death.place")
        own_home = category == "3" and record.link("death.own_residence", _own_residence_draw)
        values.update(_occurrence_address(record, rng, category, own_home, death_profile))
    else:
        values.update({"deceased.occurrence.address": None, "deceased.occurrence.number": None,
                       "deceased.occurrence.complement": None, "deceased.occurrence.cep": None,
                       "deceased.occurrence.neighborhood": None})
    # The Cartão Nacional de Saúde only exists from 1996+: for the DO this is moot (deaths are
    # already restricted to >= 2011, M5), but the generic path keeps the old unconditional draw.
    values["deceased.cns"] = (record.draw("deceased.cns", rng, lambda: cns_digits(rng))
                              if (not is_do or on >= date(1997, 1, 1)) else None)
    values["deceased.race"] = record.pick(rng, "deceased.race", RACES, RACE_WEIGHTS)
    schooling, series = _schooling_for_age(rng, age, occupation if is_do else None)
    values["deceased.schooling"] = schooling
    values["deceased.schooling_series"] = series
    marital_options, marital_weights = _DECEASED_MARITAL_WEIGHTS(age)
    marital = record.pick(rng, "deceased.marital_status", marital_options, marital_weights)
    values["deceased.marital_status_do"] = MARITAL_TO_DO.get(marital, "Ignorada")
    married_like = marital in {"CASADO(A)", "VIÚVO(A)", "UNIÃO ESTÁVEL"}
    spouse_sex = "F" if sex == "M" else "M"
    values["deceased.marital_status"] = marital
    values["deceased.spouse_name"] = (record.draw("deceased.spouse_name", rng, lambda: person_name(record, rng, spouse_sex))
                                      if married_like else None)
    values["deceased.children_summary"] = record.draw("deceased.children_summary", rng,
                                                      lambda: _children_summary(record, rng, age, name))
    return values


def _children_summary(record, rng: random.Random, age: int, parent: str) -> str:
    if age < 16 or rng.random() < 0.3:
        return "Não"
    ages = sorted((rng.randint(max(0, age - 50), age - 16) for _ in range(rng.choices((1, 2, 3), (4, 4, 2))[0])), reverse=True)
    return "; ".join(f"{person_name(record, rng, rng.choice(('F', 'M')), [surnames(rng)[0], parent.split()[-1]])}, "
                     f"{years} {'ano' if years == 1 else 'anos'}" for years in ages)


def death_registration(record, rng: random.Random) -> dict[str, Any]:
    on = record.value("death.date")
    span = DateSpan(on, min(on + timedelta(days=365), EVENT_END))
    day = record.pick(rng, "death_registration.date", span,
                      [1 if index <= 15 else 0.3 if index <= 90 else 0.01 for index in range(len(span))])
    office = pick_profile(record, rng, "death_registration.office", favoring(profile_of(record, "death.place"), 0.8),
                          ("municipality", "uf"))
    values = {
        "death_registration.date": day,
        "death_registration.number": record.draw("death_registration.number", rng, lambda: str(rng.randint(1, 250000))),
        "death_registration.office.name": record.pick(rng, "death_registration.office.name", REGISTRY_OFFICE_NAMES),
        "death_registration.office.municipality": office["city"], "death_registration.office.uf": office["uf"],
        "death_registration.place.summary": f"{office['city']} - {office['uf']}",
    }
    sex = rng.choice(("F", "M"))
    values["death_registration.declarant_name"] = record.draw("death_registration.declarant_name", rng,
                                                               lambda: person_name(record, rng, sex))
    values["death_registration.matricula"] = build_matricula(rng, day.year, "4")
    values["death_registration.witness1_name"] = record.draw("death_registration.witness1_name", rng,
                                                              lambda: person_name(record, rng, rng.choice(("F", "M"))))
    values["death_registration.witness2_name"] = record.draw("death_registration.witness2_name", rng,
                                                              lambda: person_name(record, rng, rng.choice(("F", "M"))))
    return values


def certifier(record, rng: random.Random) -> dict[str, Any]:
    sex = rng.choice(("F", "M"))
    is_do = record.context == "declaracao_obito"
    if is_do:
        # Mirrors the exact same facts death() used to derive necropsy (B4): external deaths
        # always go to the IML; a natural death at home without medical assistance goes to the
        # SVO; everything else (hospital, or home WITH assistance) is a regular assistente/
        # substituto. This never produces IML for a natural death, nor SVO for a hospital one.
        manner = record.value("death.cause.manner")
        hospital = record.value("death.place.category") in ("1", "2")
        assistance = record.value("death.medical_assistance")
        if manner == "Não natural":
            role = "3"
        elif not hospital and assistance == "2":
            role = "4"
        elif hospital:
            role = rng.choices(["1", "2"], (85, 15))[0]
        else:
            role = rng.choices(["1", "2"], (70, 30))[0]
    else:
        role = rng.choices(["1", "2", "3", "4"], (80, 10, 5, 5))[0]
    iml_svo = role in ("3", "4")
    profile = profile_of(record, "death.place") if iml_svo else None
    ddd = UF_DDD.get(record.value("death.place.uf"), "61")
    return {
        "certifier.name": record.draw("certifier.name", rng, lambda: person_name(record, rng, sex)),
        "certifier.registration": record.draw("certifier.registration", rng, lambda: str(rng.randint(1000, 999999))),
        "certifier.role": role,
        "certifier.contact": record.draw("certifier.contact", rng,
                                         lambda: f"({ddd}) 9{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"),
        "certifier.iml_svo_municipality": profile["city"] if iml_svo else None,
        "certifier.iml_svo_uf": profile["uf"] if iml_svo else None,
    }


def build_matricula(rng: random.Random, year: int, livro: str) -> str:
    term = f"{rng.randint(1, 250000):07d}"
    return "".join((digits(rng, 6), "01", "55", f"{year:04d}", livro, f"{rng.randint(1, 400):05d}",
                    f"{rng.randint(1, 300):03d}", term, digits(rng, 2)))


def burial(record, rng: random.Random) -> dict[str, Any]:
    profile = pick_profile(record, rng, "burial", favoring(profile_of(record, "death.place"), 0.75), ("municipality", "uf"))
    values = address("burial", profile, "", ("municipality", "uf"))
    return {**values, "burial.place": f"Cemitério Municipal de {values['burial.municipality']}"}


def _parent_order(rng: random.Random) -> tuple[str, str]:
    order = ("mother", "father")
    return order if rng.random() < 0.5 else order[::-1]


def _parent_view(record, rng: random.Random, prefix: str, role: str) -> dict[str, Any]:
    name = record.value(f"{role}.name")
    profile = pick_profile(record, rng, f"{prefix}.naturalidade", parts=("municipality", "uf"))
    naturalidade = address(f"{prefix}.naturalidade", profile, "", ("municipality", "uf"))
    grandmother, grandfather = parent_names(record, rng, name)
    return {f"{prefix}.name": name, f"{prefix}.grandparents_names": f"{grandmother} e {grandfather}", **naturalidade}


def parent1(record, rng: random.Random) -> dict[str, Any]:
    first, _ = record.link("parent_order", _parent_order)
    return _parent_view(record, rng, "parent1", first)


def parent2(record, rng: random.Random) -> dict[str, Any]:
    _, second = record.link("parent_order", _parent_order)
    return _parent_view(record, rng, "parent2", second)


SPOUSE_NATIONALITIES = ("BRASILEIRA",)
SPOUSE_NATIONALITY_WEIGHTS = (1,)
MARITAL_STATUS_BEFORE = ("SOLTEIRO(A)", "DIVORCIADO(A)", "VIÚVO(A)")


def _spouse(record, rng: random.Random, prefix: str) -> dict[str, Any]:
    sex = rng.choice(("F", "M"))
    name = record.draw(f"{prefix}.name", rng, lambda: person_name(record, rng, sex))
    nationality = record.pick(rng, f"{prefix}.nationality", SPOUSE_NATIONALITIES, SPOUSE_NATIONALITY_WEIGHTS)
    profile = pick_profile(record, rng, f"{prefix}.naturalidade", parts=("municipality", "uf"))
    naturalidade = address(f"{prefix}.naturalidade", profile, "", ("municipality", "uf"))
    mother_name, father_name = parent_names(record, rng, name)
    return {
        f"{prefix}.name": name,
        f"{prefix}.cpf": record.draw(f"{prefix}.cpf", rng, lambda: cpf_digits(rng)),
        f"{prefix}.nationality": nationality,
        f"{prefix}.parents_names": f"{mother_name} e {father_name}",
        **naturalidade,
    }


def spouse1(record, rng: random.Random) -> dict[str, Any]:
    return _spouse(record, rng, "spouse1")


def spouse2(record, rng: random.Random) -> dict[str, Any]:
    return _spouse(record, rng, "spouse2")


def _name_after(rng: random.Random, own: str, other: str) -> str:
    own_first, own_family, other_last = own.split()[0], family_names(own), family_names(other)[-1]
    choice = rng.choices(("keep", "add", "drop"), (55, 35, 10))[0]
    if choice == "add" and other_last not in own_family:
        family = own_family + [other_last]
    elif choice == "drop" and len(own_family) > 1:
        family = own_family[:-1]
    else:
        family = own_family
    return " ".join([own_first, *family])


PROPERTY_REGIMES = ("COMUNHÃO PARCIAL DE BENS", "COMUNHÃO UNIVERSAL DE BENS", "SEPARAÇÃO TOTAL DE BENS",
                    "PARTICIPAÇÃO FINAL NOS AQUESTOS")
PROPERTY_REGIME_WEIGHTS = (70, 14, 13, 3)
MANDATORY_SEPARATION_AGE = 70


def _spouse_marital_status(rng: random.Random, record, prefix: str, age: int) -> str:
    weights = (96, 3, 1) if age < 24 else (70, 27, 3) if age < 45 else (45, 40, 15)
    return record.pick(rng, f"{prefix}.marital_status_before", MARITAL_STATUS_BEFORE, weights)


def marriage(record, rng: random.Random) -> dict[str, Any]:
    day = record.pick(rng, "marriage.date", DateSpan(EVENT_START, EVENT_END))
    age1, born1 = age_and_birth_date(record, rng, "spouse1", day, SPOUSE_AGES, SPOUSE_AGE_WEIGHTS)
    age2, born2 = age_and_birth_date(record, rng, "spouse2", day, SPOUSE_AGES, SPOUSE_AGE_WEIGHTS)
    if max(age1, age2) >= MANDATORY_SEPARATION_AGE:
        regime = "SEPARAÇÃO OBRIGATÓRIA DE BENS"
    else:
        regime = record.pick(rng, "marriage.property_regime", PROPERTY_REGIMES, PROPERTY_REGIME_WEIGHTS)
    sex = rng.choice(("F", "M"))
    name1, name2 = record.value("spouse1.name"), record.value("spouse2.name")
    return {
        "marriage.date": day, "marriage.property_regime": regime,
        "marriage.celebrant.name": record.draw("marriage.celebrant.name", rng, lambda: person_name(record, rng, sex)),
        "spouse1.name_after": _name_after(rng, name1, name2), "spouse2.name_after": _name_after(rng, name2, name1),
        "spouse1.age": age1, "spouse2.age": age2, "spouse1.birth_date": born1, "spouse2.birth_date": born2,
        "spouse1.marital_status_before": _spouse_marital_status(rng, record, "spouse1", age1),
        "spouse2.marital_status_before": _spouse_marital_status(rng, record, "spouse2", age2),
    }


def marriage_registration(record, rng: random.Random) -> dict[str, Any]:
    start = record.value("marriage.date")
    span = DateSpan(start, min(start + timedelta(days=60), EVENT_END))
    day = record.pick(rng, "marriage_registration.date", span,
                      [10 if index == 0 else 4 if index <= 5 else 0.3 for index in range(len(span))])
    return {"marriage_registration.date": day, "marriage_registration.matricula": build_matricula(rng, day.year, "2")}


CLERK_ROLES = ("OFICIAL DE REGISTRO CIVIL", "OFICIAL SUBSTITUTO", "ESCREVENTE AUTORIZADO", "ESCREVENTE SUBSTITUTO")
CLERK_ROLE_WEIGHTS = (55, 15, 20, 10)
OFFICE_REGISTRATION_DATE_KEYS = {"certidao_nascimento": "registration.date", "declaracao_obito": "death_registration.date",
                                 "certidao_casamento": "marriage_registration.date"}


def office(record, rng: random.Random) -> dict[str, Any]:
    context = record.context
    if context == "certidao_nascimento":
        municipality, uf = record.value("registration.office.municipality"), record.value("registration.office.uf")
    elif context == "declaracao_obito":
        municipality, uf = record.value("death_registration.office.municipality"), record.value("death_registration.office.uf")
    else:
        chosen = rng.choice(LOCATION_PROFILES)
        municipality, uf = chosen["city"], chosen["uf"]
    profile = next(item for item in LOCATION_PROFILES if item["city"] == municipality)
    number = house_number(record, rng, "office.address_number")
    return {
        "office.cns": record.draw("office.cns", rng, lambda: digits(rng, 6)),
        "office.official_name": record.draw("office.official_name", rng, lambda: person_name(record, rng, rng.choice(("F", "M")))),
        "office.city_uf": f"{municipality} - {uf}",
        "office.address": (f"{profile['street']}, nº {number}, {profile['neighborhood']}, "
                           f"CEP {profile['cep'][:5]}-{profile['cep'][5:]}, {municipality} - {uf}"),
    }


def certificate(record, rng: random.Random) -> dict[str, Any]:
    context = record.context
    if context in OFFICE_REGISTRATION_DATE_KEYS:
        start = record.value(OFFICE_REGISTRATION_DATE_KEYS[context])
    else:
        start = max((record.value(key) for key in OFFICE_REGISTRATION_DATE_KEYS.values()), default=EVENT_START)
    span = DateSpan(start, min(start + timedelta(days=5 * 365), EVENT_END))
    day = record.pick(rng, "certificate.issue_date", span,
                      [8 if index <= 30 else 2 if index <= 365 else 0.2 for index in range(len(span))])
    sex = rng.choice(("F", "M"))
    city_uf = record.value("office.city_uf")
    return {
        "certificate.issue_date": day,
        "certificate.clerk_name": record.draw("certificate.clerk_name", rng, lambda: person_name(record, rng, sex)),
        "certificate.clerk_role": record.pick(rng, "certificate.clerk_role", CLERK_ROLES, CLERK_ROLE_WEIGHTS),
        "certificate.issue_summary": f"{city_uf}, {format_date(day, 'EXTENSO_PALAVRAS')}",
    }


GENERATORS: dict[str, Callable[[Any, random.Random], dict[str, Any]]] = {
    "newborn": newborn, "mother": mother, "father": father, "guardian": guardian, "parents": parents,
    "grandparents": grandparents, "birth": birth, "pregnancy": pregnancy, "filling": filling, "registration": registration,
    "deceased": deceased, "death": death, "certifier": certifier, "twin": twin, "death_registration": death_registration,
    "burial": burial, "parent1": parent1, "parent2": parent2, "spouse1": spouse1, "spouse2": spouse2,
    "marriage": marriage, "marriage_registration": marriage_registration, "office": office, "certificate": certificate,
}
