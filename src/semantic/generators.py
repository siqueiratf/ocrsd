from __future__ import annotations

import hashlib
import math
import random
import re
from datetime import date, time, timedelta
from typing import Any, Callable

from src.field_rules import DNV_TEST_PREFIX, cpf
from src.semantic.constraints import REFERENCE_END, completed_years
from src.semantic.data import (COMPLEMENTS, ESTABLISHMENT_NAMES, FEMALE_ASSOCIATED_FIRST_NAMES, ISSUERS, LOCATION_PROFILES,
                               MALE_ASSOCIATED_FIRST_NAMES, OCCUPATIONS, OCCUPATION_CODES, PROFILES_BY_IBGE, REGISTRY_OFFICE_NAMES, SURNAMES)
from src.semantic.present import format_matricula

EVENT_START, EVENT_END = date(1980, 1, 1), REFERENCE_END
MINUTES = tuple(time(minute // 60, minute % 60) for minute in range(24 * 60))


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
    return {
        "newborn.anomaly_status": "Ignorado", "newborn.anomaly_description": None,
        "newborn.race": record.pick(rng, "newborn.race", ("Branca", "Preta", "Amarela", "Parda", "Indígena"), (38, 10, 1, 49, 2)),
        "newborn.weight_g": weight, "newborn.length_tenths_cm": length, "newborn.head_circumference_cm": head,
        "newborn.apgar_1min": apgar_1, "newborn.apgar_5min": record.pick(rng, "newborn.apgar_5min", range(apgar_1, 11)),
        "newborn.name": record.draw("newborn.name", rng, lambda: person_name(record, rng, sex, list(dict.fromkeys(family)))),
        "newborn.sex": sex, "newborn.cpf": record.draw("newborn.cpf", rng, lambda: cpf_digits(rng)),
        "newborn.naturalidade": record.pick(rng, "newborn.naturalidade", cities, (60, 40)),
    }


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


def death(record, rng: random.Random) -> dict[str, Any]:
    profile = pick_profile(record, rng, "death.place", parts=("municipality", "uf", "ibge"))
    return {
        "death.date": record.pick(rng, "death.date", DateSpan(EVENT_START, EVENT_END)),
        "death.time": record.pick(rng, "death.time", MINUTES),
        "death.type": "2", "death.place.category": "1", "death.medical_assistance": "1", "death.autopsy": "2",
        "death.place.establishment.name": record.pick(rng, "death.place.establishment.name", ESTABLISHMENT_NAMES),
        **address("death.place", profile, "", ("municipality", "uf", "ibge")),
    }


def deceased(record, rng: random.Random) -> dict[str, Any]:
    age, born = age_and_birth_date(record, rng, "deceased", record.value("death.date"), DECEASED_AGES, DECEASED_AGE_WEIGHTS)
    sex = record.pick(rng, "deceased.sex", ("M", "F"), (52, 48))
    name = record.draw("deceased.name", rng, lambda: person_name(record, rng, sex))
    mother_name, father_name = parent_names(record, rng, name)
    residence = pick_profile(record, rng, "deceased.residence", favoring(profile_of(record, "death.place"), 0.7))
    birthplace = rng.choice(LOCATION_PROFILES)
    occupation = "Não se aplica" if age < 14 else record.pick(rng, "deceased.occupation", OCCUPATIONS)
    return {
        "deceased.name": name, "deceased.sex": sex, "deceased.birth_date": born, "deceased.age": age,
        "deceased.cpf": record.draw("deceased.cpf", rng, lambda: cpf_digits(rng)),
        "deceased.naturalidade": f"{birthplace['city']} - {birthplace['uf']}",
        "deceased.mother_name": record.draw("deceased.mother_name", rng, lambda: mother_name),
        "deceased.father_name": record.draw("deceased.father_name", rng, lambda: father_name),
        "deceased.occupation": occupation, "deceased.occupation_cbo": OCCUPATION_CODES.get(occupation),
        **address("deceased.residence", residence, house_number(record, rng, "deceased.residence.number"),
                  ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge")),
    }


def death_registration(record, rng: random.Random) -> dict[str, Any]:
    on = record.value("death.date")
    span = DateSpan(on, min(on + timedelta(days=365), EVENT_END))
    day = record.pick(rng, "death_registration.date", span,
                      [1 if index <= 15 else 0.3 if index <= 90 else 0.01 for index in range(len(span))])
    office = pick_profile(record, rng, "death_registration.office", favoring(profile_of(record, "death.place"), 0.8),
                          ("municipality", "uf"))
    return {
        "death_registration.date": day,
        "death_registration.number": record.draw("death_registration.number", rng, lambda: str(rng.randint(1, 250000))),
        "death_registration.office.name": record.pick(rng, "death_registration.office.name", REGISTRY_OFFICE_NAMES),
        "death_registration.office.municipality": office["city"], "death_registration.office.uf": office["uf"],
        "death_registration.place.summary": f"{office['city']} - {office['uf']}",
    }


def certifier(record, rng: random.Random) -> dict[str, Any]:
    sex = rng.choice(("F", "M"))
    return {"certifier.name": record.draw("certifier.name", rng, lambda: person_name(record, rng, sex)),
            "certifier.registration": record.draw("certifier.registration", rng, lambda: str(rng.randint(1000, 999999))),
            "certifier.role": "1"}


GENERATORS: dict[str, Callable[[Any, random.Random], dict[str, Any]]] = {
    "newborn": newborn, "mother": mother, "father": father, "guardian": guardian, "parents": parents,
    "grandparents": grandparents, "birth": birth, "pregnancy": pregnancy, "filling": filling, "registration": registration,
    "deceased": deceased, "death": death, "certifier": certifier, "twin": twin, "death_registration": death_registration,
}
