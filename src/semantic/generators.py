from __future__ import annotations

import hashlib
import math
import random
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from src.field_rules import DNV_TEST_PREFIX, cpf, format_date
from src.semantic.constraints import REFERENCE_END, completed_years
from src.semantic.data import (ADULT_ESTABLISHMENT_NAMES, ANOMALIES, COMPLEMENTS, ESTABLISHMENT_NAMES, FEMALE_ASSOCIATED_FIRST_NAMES,
                               ISSUERS, LOCATION_PROFILES, MALE_ASSOCIATED_FIRST_NAMES, OCCUPATIONS, OCCUPATION_CODES,
                               OCCUPATION_GENDER, OCCUPATION_OPPOSITE_FORM, OTHER_HEALTH_ESTABLISHMENT_NAMES,
                               PROFILES_BY_IBGE, REGION_BY_UF, REGISTRY_OFFICE_NAMES, SURNAMES)
from src.semantic.present import format_matricula, matricula_check_digits
from src.semantic.cause_chains import CAUSE_CHAINS, CHILD_PART2_POOL, PART2_POOL, format_interval

EVENT_START, EVENT_END = date(1980, 1, 1), REFERENCE_END
MATRICULA_START = date(2010, 1, 1)
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
# The minimum plausible age for an occupation whose schooling is derived above (R-d): a
# "Superior completo" title needs time to finish a degree (~22), a technical/"Médio" title
# needs to at least be past compulsory schooling age (~16). Below either age the occupation
# is excluded from the draw pool, so occupation/schooling/age stay mutually consistent by
# construction instead of being drawn independently.
OCCUPATION_MIN_AGE = {occupation: (22 if minimum == "Superior completo" else 16)
                      for occupation, minimum in OCCUPATION_MIN_SCHOOLING.items()}
# Below this age a person is not plausibly holding any job but still being schooled (R-d,
# m-3): the mother's occupation collapses to "Estudante" instead of being drawn from the
# general pool.
MIN_WORKING_AGE = 16


def _max_schooling_level_for_age(age: int, medio_age: int = 15) -> int:
    """M2/R2: the highest SCHOOLING_LEVELS index plausible at `age` — Fundamental I only from 7
    (R2: a child starts 1st grade at 6, so nothing is "completed" before 7), Fundamental II from
    11, Médio from `medio_age` (15 for a child/adolescent deceased; the mother's own call site
    passes 17 — re-verification m-a: a 15-year-old mother should not show a COMPLETED Médio
    série yet), Superior incompleto from 18, Superior completo from 22 (a degree takes time to
    finish). Used for the deceased's own schooling (any age group, R2) and the mother's."""
    if age < 7:
        return 0  # nothing completed yet
    if age < 11:
        return 1  # Fundamental I at most
    if age < medio_age:
        return 2  # Fundamental II
    if age < 18:
        return 3  # Médio
    if age < 22:
        return 4  # Superior incompleto — 18-21 (inclusive) never "completo"
    return 5  # Superior completo


def _schooling_for_age(rng: random.Random, age: int, occupation: str | None = None,
                       cap_by_age: bool = False, medio_age: int = 15) -> tuple[str, int | None]:
    """`cap_by_age` (M2/R2) is opt-in: it is passed True ONLY from the DO's own deceased and
    block-IV mother call sites (see deceased()/deceased_birth_block() below) — the generic
    mother()/deceased() paths shared with the OTHER documents (certidão de nascimento, DNV,
    certidão de óbito outside the DO...) keep calling this with the old, uncapped behaviour, so
    those documents stay byte-identical to before phase 2 (M1's own requirement)."""
    minimum = OCCUPATION_MIN_SCHOOLING.get(occupation or "")
    cap = _max_schooling_level_for_age(age, medio_age) if cap_by_age else 5
    if minimum == "Superior completo":
        return "Superior completo", None
    if minimum == "Médio (antigo 2º grau)":
        levels = [level for level in SCHOOLING_LEVELS[3:] if SCHOOLING_LEVELS.index(level) <= cap] or [SCHOOLING_LEVELS[min(cap, 3)]]
        level = rng.choices(levels, (60, 20, 20)[:len(levels)])[0]
        series_range = SCHOOLING_SERIES_RANGE.get(level)
        return level, rng.randint(*series_range) if series_range else None
    if age < 5:
        return SCHOOLING_LEVELS[0], None
    if age < 7:
        levels, weights = SCHOOLING_LEVELS[:2], (60, 40)
    elif age < 15:
        levels, weights = SCHOOLING_LEVELS[:min(cap + 1, 3)], (10, 40, 50)
    else:
        levels, weights = SCHOOLING_LEVELS[:cap + 1], (8, 15, 22, 30, 10, 15)
    levels = levels[:cap + 1]
    weights = weights[:len(levels)]
    level = rng.choices(levels, weights)[0]
    series_range = SCHOOLING_SERIES_RANGE.get(level)
    if cap_by_age and series_range and level in (SCHOOLING_LEVELS[1], SCHOOLING_LEVELS[2]):
        # R2: completed série must never exceed age - 6 (a child enters 1st grade at 6) — this
        # applies to BOTH Fundamental I (série 1-4) and Fundamental II (série 5-8), not only
        # Fundamental II: a 7-year-old with Fundamental I série 4 is just as wrong.
        max_series = min(series_range[1], max(series_range[0], age - 6))
        series_range = (series_range[0], max_series)
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


def _ordered_intervals(rng: random.Random, part1: list, age_minutes: int | None = None) -> dict[int, int]:
    """Sample each line's interval value so they are non-decreasing from ``a`` to ``d`` in
    minutes, by construction (R1): walk from ``d`` (longest) down to ``a`` (shortest), capping
    each line's range at the minutes of the line already sampled below it. Lines sharing a
    `group` id (near-simultaneous events) reuse one sampled value.

    B2: `age_minutes`, when given, caps every interval so Part I never claims the condition
    started before the deceased was born/conceived of the event; a `since_birth` line (B2:
    prematurity, congenital heart disease, birth asphyxia) is forced to exactly the age, in its
    own unit (never sampled, never 0 — these conditions exist for the deceased's whole life)."""
    shared: dict[int, int] = {}
    values: dict[int, int] = {}
    cap_minutes: int | None = age_minutes
    for index in reversed(range(len(part1))):
        line = part1[index]
        if line is None:
            continue
        if line.since_birth and age_minutes is not None:
            value = max(1, age_minutes // _UNIT_MINUTES[line.unit])
            value = max(line.lo, min(line.hi, value)) if value <= line.hi else value
        elif line.group and line.group in shared:
            value = shared[line.group]
        else:
            hi = line.hi
            if cap_minutes is not None:
                capped_hi = cap_minutes // _UNIT_MINUTES[line.unit]
                hi = capped_hi if capped_hi >= line.lo else line.lo
                hi = min(hi, line.hi) if capped_hi >= line.lo else hi
            value = rng.randint(line.lo, max(line.lo, hi))
            if value == 0:
                value = 1
            if line.group:
                shared[line.group] = value
        values[index] = value
        line_minutes = value * _UNIT_MINUTES[line.unit]
        cap_minutes = line_minutes if cap_minutes is None else min(cap_minutes, line_minutes)
    return values


def _render_chain(rng: random.Random, chain, category: str, scenario: str | None,
                  age_minutes: int | None = None, age_group: str = "adult") -> dict[str, Any]:
    """Render a chosen CauseChain's Part I/II into the death.cause.* fields.

    Text variants are sampled per record (variety); interval numbers are sampled so they stay
    non-decreasing from ``a`` to ``d``, and never exceed `age_minutes` when given (R1/B2, see
    `_ordered_intervals`) — `age_minutes` is now passed for every non-fetal age group, adult and
    child included, not just infant. W00-Y34 external-cause codes get the 4th "place of
    occurrence" character appended from the DO's own field 20 (`category`); V01-V99 codes are
    stored complete in the table (R3). Part II never repeats a Part I code/category; the pool is
    picked by `age_group` (R1: the adult comorbidity pool — diabetes, hypertension, COPD... — is
    never medically plausible for a child, so children draw from their own small pool instead,
    and fetal/infant draw from neither); any pool line whose OWN minimum duration already
    exceeds the age is dropped rather than clamped up to it (R1 — clamping up to `lo` is how a
    1-year-old used to get "Etilismo crônico 13 anos").
    """
    part1 = list(chain.part1) + [None] * (4 - len(chain.part1))
    interval_values = _ordered_intervals(rng, part1, age_minutes)
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
    part2_eligible = chain.age_unit == "anos" and (chain.manner == "Natural" or scenario in _PART2_ELIGIBLE_EXTERNAL)
    source_pool = (PART2_POOL if age_group == "adult" else CHILD_PART2_POOL if age_group == "child" else ()) \
        if part2_eligible else ()
    pool = [line for line in source_pool if line.cid.split(".")[0] not in used_cids]
    if age_minutes is not None:
        pool = [line for line in pool if line.lo * _UNIT_MINUTES[line.unit] <= age_minutes]
    rng.shuffle(pool)
    count_options, count_weights = ((0, 1), (90, 10)) if chain.manner == "Não natural" else ((0, 1, 2), (35, 45, 20))
    count = rng.choices(count_options, count_weights)[0] if part2_eligible else 0
    chosen = (pool[:count] + [None, None])[:2]
    for slot, line in enumerate(chosen, start=1):
        if line is None:
            values[f"death.cause.ii{slot}_text"] = None
            values[f"death.cause.ii{slot}_cid"] = None
            values[f"death.cause.ii{slot}_interval"] = None
            continue
        hi = line.hi
        if age_minutes is not None:
            hi = min(line.hi, age_minutes // _UNIT_MINUTES[line.unit])
        value = max(1, rng.randint(line.lo, max(line.lo, hi)))
        values[f"death.cause.ii{slot}_text"] = line.text_variants[0]
        values[f"death.cause.ii{slot}_cid"] = line.cid
        values[f"death.cause.ii{slot}_interval"] = format_interval(line.unit, value)
    return values


def _weights(low: int, high: int, bands: tuple[tuple[int, float], ...]) -> tuple[float, ...]:
    return tuple(next(weight for limit, weight in bands if value <= limit) for value in range(low, high + 1))


MOTHER_AGES = range(13, 50)
MOTHER_AGE_WEIGHTS = _weights(13, 49, ((14, 0.3), (17, 3), (19, 5), (24, 9), (29, 10), (34, 9), (39, 5),
                                       (44, 1.5), (49, 0.2)))
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
# m1: SIM stillbirths skew preterm much more than live births (roughly 22-27w 15-20%, 28-31w
# 15%, 32-36w 25%, 37+w ~40-45%) — unlike WEEK_WEIGHTS above (78% term), used only for a live
# birth's own gestation.
FETAL_WEEK_WEIGHTS = _weights(22, 42, ((27, 3.2), (31, 3), (36, 5), (42, 8)))
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


def office_cns(record, name: str, ibge: str) -> str:
    """The CNS of a registry office: the same for each record that names the same office, or the CNS
    that the caller requires, so that the matrícula and the printed CNS agree."""
    required = record.required_values("office.cns") if hasattr(record, "required_values") else None
    return min(required) if required else stable_digits(6, "cns", name, ibge)


def matricula(cns: str, year: int, livro: str, book: int, page: int, term: int) -> str:
    body = f"{cns}0155{year:04d}{livro}{book:05d}{page:03d}{term:07d}"
    return body + matricula_check_digits(body)


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


def _distinct_name(rng: random.Random, record, sex: str, family: list[str] | None, others: tuple[str, ...]) -> str:
    """Draw a person_name that is not equal to any name already in `others` (item 6): avoids
    e.g. a newborn whose name happens to repeat a parent's, by chance of the gender-crossing
    first-name draw plus a shared surname."""
    name = ""
    for _ in range(128):
        name = person_name(record, rng, sex, family)
        if name not in others:
            return name
    raise ValueError("Não foi possível gerar um nome distinto após 128 tentativas")


def family_names(name: str) -> list[str]:
    return name.split()[1:] or [name]


def parent_names(record, rng: random.Random, child: str) -> tuple[str, str]:
    family = family_names(child)
    mother = person_name(record, rng, "F", surnames(rng, paternal=family[0]))
    father = person_name(record, rng, "M", surnames(rng, paternal=family[-1]))
    return mother, father


def inflect_occupation(name: str | None, sex: str) -> str | None:
    """Inflect an occupation title to agree with the person's sex (item 4): e.g. "Costureira"
    for a man becomes "Costureiro". Entries absent from OCCUPATION_GENDER (Comerciante,
    Motorista, Estudante...) are already gender-neutral in Portuguese and pass through."""
    if not name or OCCUPATION_GENDER.get(name) == sex:
        return name
    return OCCUPATION_OPPOSITE_FORM.get(name, name)


def profile_of(record, prefix: str) -> dict:
    return PROFILES_BY_IBGE[record.value(f"{prefix}.ibge")]


def favoring(favored: dict, weight: float) -> list[float]:
    """Weight LOCATION_PROFILES so `favored` gets `weight` and the rest split the remainder —
    but not uniformly (item 1): a profile in the SAME region as `favored` gets a much larger
    share than one in a distant region, because a real cross-state event (residence vs death,
    or a cartório away from both) is overwhelmingly a move to a neighbouring state or another
    state's capital for care, not an arbitrary one. Lei 6.015/73 art. 77 keeps `weight` itself
    high so crossing state lines at all stays rare."""
    remainder = 1 - weight
    same_region = [p for p in LOCATION_PROFILES if p is not favored and REGION_BY_UF[p["uf"]] == REGION_BY_UF[favored["uf"]]]
    other_region = [p for p in LOCATION_PROFILES if p is not favored and p not in same_region]
    region_share = remainder * 0.7 if same_region else 0.0
    other_share = remainder - region_share
    per_region = region_share / len(same_region) if same_region else 0.0
    per_other = other_share / len(other_region) if other_region else 0.0
    weights = []
    for profile in LOCATION_PROFILES:
        if profile is favored:
            weights.append(weight)
        elif profile in same_region:
            weights.append(per_region)
        else:
            weights.append(per_other)
    return weights


def office_weights(primary: dict, secondary: dict | None, primary_weight: float, secondary_weight: float) -> list[float]:
    """Weight LOCATION_PROFILES for a cartório/registration office so it lands ONLY in
    `primary` (e.g. the place of death/birth — Lei 6.015/73 art. 77/50) or in `secondary`
    (the residence) — never anywhere else (R-a): those are the only two places the law
    allows, so there is no legitimate "unrelated" remainder to spread over the rest of the
    country."""
    # When `secondary` coincides with `primary` (e.g. residence == death municipality), or
    # there is no secondary at all, every weight collapses onto `primary` — there is nothing
    # else to split between. Otherwise the two weights are normalized to sum to 1, so the
    # split between primary/secondary keeps its intended ratio with zero left over.
    same_profile = secondary is not None and secondary is primary
    if same_profile or secondary is None:
        primary_share, secondary_share = 1.0, 0.0
    else:
        total = primary_weight + secondary_weight
        primary_share, secondary_share = primary_weight / total, secondary_weight / total
    weights = []
    for profile in LOCATION_PROFILES:
        if profile is primary:
            weights.append(primary_share + secondary_share if same_profile else primary_share)
        elif not same_profile and secondary is not None and profile is secondary:
            weights.append(secondary_share)
        else:
            weights.append(0.0)
    return weights


def pick_profile(record, rng: random.Random, prefix: str, weights: list[float] | None = None,
                 parts: tuple[str, ...] = ("street", "neighborhood", "cep", "municipality", "uf", "ibge")) -> dict:
    fields = {"street": "street", "neighborhood": "neighborhood", "cep": "cep", "municipality": "city", "uf": "uf", "ibge": "ibge"}
    return record.pick(rng, lambda profile: {f"{prefix}.{part}": profile[fields[part]] for part in parts},
                       LOCATION_PROFILES, weights)


def pick_street(record, rng: random.Random, key: str, profile: dict) -> tuple[str, str, str]:
    """Pick one of the municipality's several streets (item 5), so two addresses in the same
    city are not always the same street; the CEP and the real/plausible bairro (m-1) both
    travel with the chosen street, so they stay consistent with it."""
    return record.pick(rng, key, profile["streets"])


def address(prefix: str, profile: dict, number: str, parts: tuple[str, ...],
           street: str | None = None, cep: str | None = None, neighborhood: str | None = None) -> dict[str, Any]:
    fields = {"street": street or profile["street"], "number": number, "neighborhood": neighborhood or profile["neighborhood"],
              "cep": cep or profile["cep"],
              "municipality": profile["city"], "uf": profile["uf"], "ibge": profile["ibge"]}
    return {f"{prefix}.{part}": fields[part] for part in parts}


def house_number(record, rng: random.Random, key: str) -> str:
    return record.draw(key, rng, lambda: str(rng.choice((rng.randint(1, 300), rng.randint(1, 3000)))))


def age_and_birth_date(record, rng: random.Random, person: str, on: date, ages: range,
                       weights: tuple[float, ...]) -> tuple[int, date]:
    age = record.pick(rng, f"{person}.age", ages, weights)
    born = record.pick(rng, f"{person}.birth_date", birth_dates_for_age(age, on))
    return completed_years(born, on), born


def age_and_birth_date_in_range(record, rng: random.Random, person: str, on: date, lo: int, hi: int,
                                age_weights: tuple[float, ...] | None = None) -> tuple[int, date]:
    """Like age_and_birth_date, but drawing from [lo, hi] (a chosen chain's own age gate)
    instead of the general population pyramid — used so the DO's deceased age is always
    inside the cause-of-death chain's plausible range, by construction. B5: when the chain
    declares its own `age_weights` (falls, traffic, Alzheimer...), they are used here instead
    of the flat draw that silently ignored them before."""
    weights = age_weights if age_weights is not None else tuple(1.0 for _ in range(lo, hi + 1))
    return age_and_birth_date(record, rng, person, on, range(lo, hi + 1), weights)


def _completed_months(born: date, on: date) -> int:
    months = (on.year - born.year) * 12 + (on.month - born.month)
    if on.day < born.day:
        months -= 1
    return months


def _subtract_calendar_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    import calendar
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _birth_date_for_months(rng: random.Random, value: int, on: date) -> date:
    """B1: the completed calendar months from `born` to `on` must equal `value` exactly — never
    30*n days (which usually completes one month short). Start from exactly `value` months
    before `on` (same day-of-month, so the month count is exact by construction: on.day is never
    < born.day there), then subtract a few extra days (0-27) for variety, retrying if that extra
    step ever changes the completed-month count (rare, near a short month)."""
    base = _subtract_calendar_months(on, value)
    for _ in range(40):
        born = base - timedelta(days=rng.randint(0, 27))
        if _completed_months(born, on) == value:
            return born
    return base


def infant_or_fetal_age(record, rng: random.Random, chain, on: date, value: int | None,
                        death_time: time | None) -> dict[str, Any]:
    """Age/birth-date fields for a fetal or infant (<1 year) chain (phase 2).

    Fetal (`chain.age_unit == "fetal"`): field 1 is "Fetal" and the DO has no age field for the
    deceased at all — the fetus "is born" the day it dies, so `deceased.birth_date` = death.date
    and `deceased.age` = 0 (keeps the general idade_do_falecido_no_obito constraint satisfied;
    field 9_anos itself is hidden by the mapping's render_when on death.age_group, not by this
    value).

    Infant (`chain.age_unit` in "minutos"/"horas"/"dias"/"meses"): `value` (drawn by death(),
    BEFORE the Part I interval is rendered — B2) is the age in the chain's own unit; the birth
    date/time is derived so recomputing the age from (birth, death date, death time) gives back
    exactly `value` (B1) — calendar months for "meses" (never 30*n days), full datetimes
    (death.time included) for "horas"/"minutos" so the hour unit is never off by a rounding
    error, and a plain calendar-day difference for "dias". `value` is never 0 (the DO rule:
    under 1 month is always given in days, never "0 meses"; under 1 day in hours, etc. — the
    chain's own [min_age, max_age] already enforces min_age >= 1 for every infant chain).
    """
    if chain.age_unit == "fetal":
        return {"deceased.age": 0, "deceased.birth_date": on,
                "deceased.age_minutes": None, "deceased.age_hours": None,
                "deceased.age_days": None, "deceased.age_months": None}
    unit_key = {"minutos": "minutes", "horas": "hours", "dias": "days", "meses": "months"}[chain.age_unit]
    if chain.age_unit == "meses":
        born = _birth_date_for_months(rng, value, on)
    elif chain.age_unit == "dias":
        born = on - timedelta(days=value)
    else:  # "horas" / "minutos": exact datetime arithmetic, including the death TIME (B1)
        on_dt = datetime.combine(on, death_time or time(0, 0))
        delta = timedelta(hours=value) if chain.age_unit == "horas" else timedelta(minutes=value)
        born = (on_dt - delta).date()
    fields = {"deceased.age_minutes": None, "deceased.age_hours": None, "deceased.age_days": None, "deceased.age_months": None}
    fields[f"deceased.age_{unit_key}"] = value
    return {"deceased.age": 0, "deceased.birth_date": born, **fields}


def gestational_history(rng: random.Random, pick: Callable, age: int, prefix: str = "mother.obstetric.") -> dict[str, int]:
    limit = max(0, (age - 13) // 2 + 1)
    weights = [weight if count <= limit else weight * 0.001 for count, weight in zip(PREGNANCY_COUNTS, PREGNANCY_COUNT_WEIGHTS)]
    gestacoes = pick(rng, prefix + "previous_pregnancies", PREGNANCY_COUNTS, weights)
    perdas = pick(rng, prefix + "fetal_losses", range(0, min(gestacoes, 3) + 1))
    births = max(0, gestacoes - perdas)
    cesareas = pick(rng, prefix + "cesareans", range(0, births + 1))
    nascidos = pick(rng, prefix + "live_births", (births, births - 1) if births else (0,), (3, 1) if births else None)
    return {prefix + "previous_pregnancies": gestacoes, prefix + "vaginal_births": births - cesareas,
            prefix + "cesareans": cesareas, prefix + "live_births": nascidos, prefix + "fetal_losses": perdas}


_WEIGHT_FOR_WEEKS = {22: 500, 24: 650, 26: 900, 28: 1150, 30: 1500, 32: 1900, 34: 2300,
                     36: 2700, 38: 3100, 39: 3250, 40: 3400, 41: 3450, 42: 3500}


def _birth_weight_for_weeks(rng: random.Random, weeks: int) -> int:
    """Birth weight (g) plausible for `weeks` of gestation (R: weight/gestation coherence) —
    linear interpolation over typical fetal-growth percentiles, +/- noise, clamped >= 100g."""
    points = sorted(_WEIGHT_FOR_WEEKS)
    w = max(min(weeks, points[-1]), points[0])
    lo = max(p for p in points if p <= w)
    hi = min(p for p in points if p >= w)
    base = (_WEIGHT_FOR_WEEKS[lo] if lo == hi else
            _WEIGHT_FOR_WEEKS[lo] + (_WEIGHT_FOR_WEEKS[hi] - _WEIGHT_FOR_WEEKS[lo]) * (w - lo) / (hi - lo))
    return max(100, min(6000, int(rng.gauss(base, max(60.0, base * 0.12)))))


INFANT_GESTATION_WEEKS = range(24, 43)
INFANT_GESTATION_WEIGHTS = _weights(24, 42, ((31, 2), (33, 3), (36, 6), (37, 10), (38, 18), (39, 22), (40, 18), (41, 8), (42, 2)))
PRETERM_GESTATION_WEEKS = range(24, 33)


def deceased_birth_block(record, rng: random.Random, age_group: str, chain) -> dict[str, Any]:
    """Fields 27-37 (block IV/V — mother's age/schooling/occupation/obstetric history, gestation,
    pregnancy/delivery type, death-vs-delivery timing, birth weight, DN number) for a fetal or
    infant (<1 year) scenario; blank for every other scenario (phase 2)."""
    if age_group not in ("fetal", "infant"):
        return {
            "deceased.mother_age": None, "deceased.mother_schooling": None, "deceased.mother_schooling_series": None,
            "deceased.mother_occupation": None, "deceased.mother_occupation_cbo": None,
            "deceased.mother_live_births": None, "deceased.mother_fetal_losses": None,
            "deceased.gestation_weeks": None, "deceased.gestation_weeks_unknown": False,
            "deceased.pregnancy_type": None, "deceased.delivery_type": None,
            "deceased.death_relative_to_delivery": None, "deceased.birth_weight_g": None, "deceased.birth.dnv_number": None,
        }
    mother_age = record.pick(rng, "deceased.mother_age", MOTHER_AGES, MOTHER_AGE_WEIGHTS)
    if mother_age < MIN_WORKING_AGE:
        mother_occupation_canonical = "Estudante"
    else:
        pool = tuple(o for o in (*OCCUPATIONS, *OCCUPATION_CODES) if mother_age >= OCCUPATION_MIN_AGE.get(o, 0))
        mother_occupation_canonical = record.pick(
            rng, lambda value: {"deceased.mother_occupation": value, "deceased.mother_occupation_cbo": OCCUPATION_CODES.get(value)},
            pool)
    mother_occupation = inflect_occupation(mother_occupation_canonical, "F")
    # m-a: the mother's own cap uses medio_age=17 (not 15) — re-verification found 15-year-old
    # mothers with a COMPLETED "Médio" level, which needs at least two more years from the
    # typical starting age to show any real complete série.
    mother_schooling, mother_series = _schooling_for_age(rng, mother_age, mother_occupation_canonical,
                                                         cap_by_age=True, medio_age=17)
    obstetric = gestational_history(rng, record.pick, mother_age, prefix="deceased.mother_obstetric_")

    # B4: a prematurity/SDR chain declares its own plausible gestation band
    # (`gestation_weeks_range`) — never "unknown" (which would otherwise draw a term weight for
    # a chain whose text says "Prematuridade"), and the weeks drawn are ALWAYS inside that band
    # so the ICD-10 code (chosen per-chain, P07.2 vs P07.3) stays consistent with the weeks.
    gestation_range = getattr(chain, "gestation_weeks_range", None)
    weeks_unknown = (False if gestation_range else record.pick(rng, "deceased.gestation_weeks_unknown", (False, True), (95, 5)))
    if gestation_range is not None:
        weeks = record.pick(rng, "deceased.gestation_weeks_chain_range", range(gestation_range[0], gestation_range[1] + 1))
    elif age_group == "fetal":
        # m1: SIM stillbirths are mostly preterm, unlike live births — a fetal-specific weight.
        weeks = None if weeks_unknown else record.pick(rng, "deceased.gestation_weeks", WEEKS, FETAL_WEEK_WEIGHTS)
    else:
        weeks = None if weeks_unknown else record.pick(rng, "deceased.gestation_weeks_infant", INFANT_GESTATION_WEEKS, INFANT_GESTATION_WEIGHTS)
    weight = _birth_weight_for_weeks(rng, weeks if weeks is not None else 38)

    twins = record.pick(rng, "deceased.pregnancy_twins", (False, True), (97, 3))
    pregnancy_type = "Dupla" if twins else "Única"
    delivery_type = record.pick(rng, "deceased.delivery_type", ("Vaginal", "Cesáreo", "Ignorado"), (52, 44, 4))
    if age_group == "fetal":
        # B3: field 34 is tied to the chosen chain (fetal_timing) — never drawn independently,
        # so it never contradicts the chain's own P20.0 ("Antes")/P20.1 ("Durante") coding.
        relative = chain.fetal_timing or record.pick(rng, "deceased.death_relative_to_delivery", ("Antes", "Durante"), (70, 30))
        dn_number = None
    else:
        relative = "Depois"
        dn_number = record.draw("deceased.birth.dnv_number", rng, lambda: DNV_TEST_PREFIX + digits(rng, 8) + digits(rng, 1))
    return {
        "deceased.mother_age": mother_age, "deceased.mother_schooling": mother_schooling,
        "deceased.mother_schooling_series": mother_series,
        "deceased.mother_occupation": mother_occupation, "deceased.mother_occupation_cbo": OCCUPATION_CODES.get(mother_occupation_canonical),
        "deceased.mother_live_births": obstetric["deceased.mother_obstetric_live_births"],
        "deceased.mother_fetal_losses": obstetric["deceased.mother_obstetric_fetal_losses"],
        "deceased.gestation_weeks": weeks, "deceased.gestation_weeks_unknown": weeks_unknown,
        "deceased.pregnancy_type": pregnancy_type, "deceased.delivery_type": delivery_type,
        "deceased.death_relative_to_delivery": relative, "deceased.birth_weight_g": weight, "deceased.birth.dnv_number": dn_number,
    }


def birth(record, rng: random.Random) -> dict[str, Any]:
    day = record.pick(rng, "birth.date", DateSpan(EVENT_START, EVENT_END))
    moment = record.pick(rng, "birth.time", MINUTES)
    profile = pick_profile(record, rng, "birth.place")
    establishment = record.pick(rng, "birth.establishment.name", ESTABLISHMENT_NAMES)
    street, cep, neighborhood = pick_street(record, rng, "birth.place.street", profile)
    values = {
        "birth.date": day, "birth.time": moment, "birth.establishment.name": establishment,
        "birth.occurrence_type": "Hospital",
        "birth.establishment.cnes": stable_digits(7, "cnes", establishment, profile["ibge"]),
        **address("birth.place", profile, house_number(record, rng, "birth.place.number"),
                  ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge"), street, cep, neighborhood),
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
    # Lei 6.015/73 art. 50: registration at the place of birth or of the mother's residence
    # (item 1) — an interstate birth (mother's residence in a different UF from the birth
    # place) is uncommon in practice, so the favored weight is high and any cross-state
    # alternative is biased to the same region (favoring()).
    residence = pick_profile(record, rng, "mother.residence", favoring(profile_of(record, "birth.place"), 0.93))
    cities = [profile["city"] for profile in LOCATION_PROFILES]
    naturalidade = record.pick(rng, "mother.naturalidade", cities,
                               [0.55 if city == residence["city"] else 0.45 / (len(cities) - 1) for city in cities])
    # R-d: occupation, schooling and age must agree — reuse the DO's occupation -> minimum
    # schooling rule (OCCUPATION_MIN_SCHOOLING/OCCUPATION_MIN_AGE) instead of drawing them
    # independently. Below MIN_WORKING_AGE the mother is still of school age, so her
    # "occupation" collapses to "Estudante" (no graduate/technical title under that age).
    if age < MIN_WORKING_AGE:
        occupation_canonical = "Estudante"
    else:
        occupation_pool = tuple(o for o in (*OCCUPATIONS, *OCCUPATION_CODES) if age >= OCCUPATION_MIN_AGE.get(o, 0))
        occupation_canonical = record.pick(
            rng, lambda value: {"mother.occupation": value, "mother.occupation_cbo": OCCUPATION_CODES.get(value)},
            occupation_pool)
    occupation = inflect_occupation(occupation_canonical, "F")
    schooling, _series = _schooling_for_age(rng, age, occupation_canonical)
    marital_status = "Solteira(o)" if age < 18 else record.pick(rng, "mother.marital_status", ("Solteira(o)", "Casada(o)", "União estável"))
    return {
        "mother.name": name, "mother.birth_date": born, "mother.age": age,
        "mother.schooling": schooling, "mother.marital_status": marital_status,
        "mother.race": record.pick(rng, "mother.race", ("Branca", "Preta", "Amarela", "Parda", "Indígena"), (38, 10, 1, 49, 2)),
        "mother.cpf": record.draw("mother.cpf", rng, lambda: cpf_digits(rng)),
        "mother.cns": record.draw("mother.cns", rng, lambda: cns_digits(rng)),
        "mother.occupation": occupation, "mother.occupation_cbo": OCCUPATION_CODES.get(occupation_canonical),
        "mother.naturalidade": naturalidade,
        **address("mother.residence", residence, house_number(record, rng, "mother.residence.number"),
                  ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge"),
                  *pick_street(record, rng, "mother.residence.street", residence)),
        "mother.residence.complement": record.pick(rng, "mother.residence.complement", COMPLEMENTS),
        **gestational_history(rng, record.pick, age),
    }


def father(record, rng: random.Random) -> dict[str, Any]:
    age, born = age_and_birth_date(record, rng, "father", record.value("birth.date"), FATHER_AGES, FATHER_AGE_WEIGHTS)
    mother_name = record.value("mother.name")

    def sample() -> str:
        name = mother_name
        for _ in range(128):
            name = person_name(record, rng, "M")
            if name != mother_name:
                return name
        raise ValueError("Não foi possível gerar um pai com nome distinto da mãe após 128 tentativas")
    return {"father.name": record.draw("father.name", rng, sample),
            "father.birth_date": born, "father.age": age}


def _anomaly_block(record, rng: random.Random, sex: str) -> dict[str, Any]:
    """Field 6 ("Detectada alguma anomalia congênita?") and, when "Sim", field 41's
    description + ICD-10 code (R8/item 1): real DNVs record "Não" in the large majority of
    cases, "Sim" in ~1-2% (SINASC prevalence of recorded anomalies is ~0.8-1%), and the
    remainder "Ignorado". Only "Sim" carries a description; it is drawn from ANOMALIES,
    restricted to the newborn's sex and weighted so Down syndrome is more likely with an
    older mother (maternal-age-related trisomy 21 risk)."""
    status = record.pick(rng, "newborn.anomaly_status", ("Não", "Sim", "Ignorado"), (975, 15, 10))
    if status != "Sim":
        return {"newborn.anomaly_status": status, "newborn.anomaly_description": None}
    mother_age = record.value("mother.age")
    pool = tuple(entry for entry in ANOMALIES if entry[2] in (None, sex))
    weights = [weight * (3 if code == "Q90.9" and mother_age >= 35 else 1) for _, code, _, weight in pool]
    description, code, _, _ = record.pick(
        rng, lambda entry: {"newborn.anomaly_description": f"{entry[0]} ({entry[1]})"}, pool, weights)
    return {"newborn.anomaly_status": status, "newborn.anomaly_description": f"{description} ({code})"}


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
        **_anomaly_block(record, rng, sex),
        "newborn.race": record.pick(rng, "newborn.race", ("Branca", "Preta", "Amarela", "Parda", "Indígena"), (38, 10, 1, 49, 2)),
        "newborn.weight_g": weight, "newborn.length_tenths_cm": length, "newborn.head_circumference_cm": head,
        "newborn.apgar_1min": apgar_1, "newborn.apgar_5min": record.pick(rng, "newborn.apgar_5min", range(apgar_1, 11)),
        "newborn.name": record.draw("newborn.name", rng, lambda: _distinct_name(
            rng, record, sex, list(dict.fromkeys(family)),
            (record.value("mother.name"), record.value("father.name")))),
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
    newborn_name, registration_matricula = record.value("newborn.name"), record.value("registration.matricula")
    sex = rng.choice(("F", "M"))
    name = newborn_name
    for _ in range(128):
        name = person_name(record, rng, sex, family_names(newborn_name))
        if name != newborn_name:
            break
    else:
        raise ValueError("Não foi possível gerar um gêmeo distinto após 128 tentativas")
    body = registration_matricula[:23] + f"{int(registration_matricula[23:30]) % 9999999 + 1:07d}"
    return {"twin.name_matricula": f"{name}, {format_matricula(body + matricula_check_digits(body))}"}


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
    # Lei 6.015/73 art. 50: registration at the place of birth or of the parents' residence
    # (item 1) — biased accordingly instead of leaving a flat 5% to "anywhere in the country".
    born_in, lives_in = profile_of(record, "birth.place"), profile_of(record, "mother.residence")
    weights = office_weights(born_in, lives_in, 0.6, 0.35)
    office = pick_profile(record, rng, "registration.office", weights, ("municipality", "uf"))
    name = record.pick(rng, "registration.office.name", REGISTRY_OFFICE_NAMES)
    term = rng.randint(1, 250000)
    code = matricula(office_cns(record, name, office["ibge"]), day.year, "1", rng.randint(1, 400), rng.randint(1, 300), term)
    return {
        "registration.date": day, "registration.number": str(term), "registration.matricula": code,
        "registration.dnv_number": record.draw("registration.dnv_number", rng,
                                               lambda: DNV_TEST_PREFIX + digits(rng, 8) + digits(rng, 1)),
        "registration.office.name": name, "registration.office.municipality": office["city"],
        "registration.office.uf": office["uf"], "registration.place.summary": f"{office['city']} - {office['uf']}",
    }


EXTERNAL_SOURCE_BY_SCENARIO = {
    "external_traffic": "Boletim de Ocorrência", "external_fall": "Boletim de Ocorrência",
    "external_homicide": "Boletim de Ocorrência", "external_suicide": "Família",
    "child_external": "Boletim de Ocorrência",
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
    ("child_external", "hospital"): "Vítima de causa externa, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("child_external", "home_own"): "Vítima de causa externa no próprio domicílio.",
    ("child_external", "home_other"): "Vítima de causa externa em domicílio.",
    ("child_external", "public"): "Vítima de causa externa, com óbito no local.",
}
# M4: field 51 gets a chain-specific description (not the generic "causa externa" fallback
# above) for the child-external chains whose event is specific enough to describe precisely.
CHILD_EXTERNAL_DESCRIPTIONS = {
    ("crianca_afogamento", "hospital"): "Vítima de afogamento, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("crianca_afogamento", "home_own"): "Vítima de afogamento em piscina do próprio domicílio.",
    ("crianca_afogamento", "home_other"): "Vítima de afogamento em piscina residencial.",
    ("crianca_afogamento", "public"): "Vítima de afogamento em rio, represa ou praia.",
    ("crianca_atropelamento", "hospital"): "Vítima de atropelamento por veículo, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("crianca_atropelamento", "public"): "Vítima de atropelamento por veículo em via pública.",
    ("adolescente_paf", "hospital"): "Vítima de agressão por disparo de arma de fogo, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("adolescente_paf", "home_own"): "Vítima de agressão por disparo de arma de fogo no próprio domicílio.",
    ("adolescente_paf", "home_other"): "Vítima de agressão por disparo de arma de fogo em domicílio.",
    ("adolescente_paf", "public"): "Vítima de agressão por disparo de arma de fogo, com óbito no local, em via pública.",
    ("adolescente_moto", "hospital"): "Vítima de acidente de trânsito com motocicleta, socorrida e encaminhada ao hospital, onde veio a óbito.",
    ("adolescente_moto", "public"): "Vítima de acidente de trânsito com motocicleta, com óbito no local, em via pública.",
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
    # Drawn here (not inline in the `values` dict below) because B1 needs the death TIME to
    # compute an infant's hour/minute age precisely (datetime arithmetic, not a day-rounding).
    death_date = record.pick(rng, "death.date", DateSpan(event_start, EVENT_END))
    death_time = record.pick(rng, "death.time", MINUTES)

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
            # Phase 2: an AI-required deceased.age is still always in completed YEARS (the AI
            # slot only offers adult ages — see assisted.py's open item in the final report), so
            # only "anos"-gated chains (adult/child) are eligible here; fetal/infant chains
            # (age_unit "fetal"/"dias"/"horas"/"meses") are never forced by this path.
            eligible = [(sid, c) for sid in SCENARIOS_BY_ID for c in chains_for_scenario(sid)
                       if c.age_unit == "anos" and c.min_age <= target_age <= c.max_age]
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
        age_group = picked.age_group
        external = manner == "Não natural"
        if chain.setting == "hospital":
            category = rng.choices(["1", "2"], (90, 10))[0]
        elif chain.place_weights:
            # M4: a chain-specific place distribution (e.g. a pedestrian death is never "at
            # home") overrides the scenario's own generic distribution.
            cats = list(chain.place_weights)
            category = rng.choices(cats, [chain.place_weights[c] for c in cats])[0]
        else:
            category = pick_place(rng, picked)
        hospital = category in ("1", "2")
    else:
        age_group = "adult"

    # B1/B2/R1: the age (in the chain's own unit — "anos" for adult/child, minutos/horas/
    # dias/meses for infant) is drawn HERE, before the Part I/Part II intervals are rendered
    # below, so EVERY interval (not just an infant's) can be capped at it by construction;
    # deceased() (which renders the birth date) reuses this exact value instead of drawing a
    # second, inconsistent one. Fetal has no real duration to cap against (age is always 0).
    age_unit_value = age_unit_minutes = None
    if is_do and chain is not None and chain.age_unit != "fetal":
        if chain.age_unit == "anos":
            # Drawn under the "deceased.age" key (via a callable project) so an AI-required
            # deceased.age (R5) still filters the candidates here, exactly as it used to when
            # this draw happened inside deceased() — required_ages/eligible above already
            # restricted the scenario/chain pair to one whose gate contains the required age.
            age_unit_value = record.pick(rng, lambda v: {"deceased.age": v},
                                         range(chain.min_age, chain.max_age + 1), chain.age_weights)
        else:
            age_unit_value = record.pick(rng, "death.cause.age_unit_value",
                                         range(chain.min_age, chain.max_age + 1), chain.age_weights)
        age_unit_minutes = age_unit_value * _UNIT_MINUTES[chain.age_unit]

    # Whether a category-"3" (domicílio) death happened in the deceased's OWN residence (~85%,
    # field 22 blank, field 51 says "próprio domicílio") or a different address (~15%, field 22
    # filled with a profile street, field 51 says "em domicílio") — R4. Shared via record.link so
    # deceased() (which renders field 22) and death() (which renders field 51's text) agree.
    own_home = category == "3" and record.link("death.own_residence", _own_residence_draw)

    values: dict[str, Any] = {
        "death.date": death_date,
        "death.time": death_time,
        "death.type": "1" if age_group == "fetal" else "2",
        "death.age_group": age_group,
        "death.place.category": category,
        "death.cause.manner": manner,
        "death.scenario": scenario,
        "death.cause.age_unit_value": age_unit_value,
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

    if not hospital:
        establishment_pool = None
    elif category == "2":
        # Field-20 category 2 ("outro estabelecimento de saúde") is UPA/UBS/policlínica/pronto
        # atendimento, never "Hospital ..." (item 2).
        establishment_pool = OTHER_HEALTH_ESTABLISHMENT_NAMES
    else:
        establishment_pool = ADULT_ESTABLISHMENT_NAMES if is_do else ESTABLISHMENT_NAMES
    values["death.place.establishment.name"] = (
        record.pick(rng, "death.place.establishment.name", establishment_pool) if establishment_pool else None)
    values["death.place.establishment.cnes"] = (
        record.draw("death.place.establishment.cnes", rng, lambda: digits(rng, 7)) if hospital else None)
    values["death.place.summary"] = (
        values["death.place.establishment.name"] if hospital
        else "Domicílio" if category == "3" else "Via pública")

    if external:
        work_hour = values["death.time"].hour
        is_work_accident = (scenario in ("external_traffic", "external_fall") and 7 <= work_hour <= 18
                            and rng.random() < 0.12)
        place_group = _place_group(category, own_home)
        values.update({
            "death.external.type": chain.external_type,
            "death.external.work_accident": "1" if is_work_accident else "2",
            "death.external.source": "Hospital" if hospital else EXTERNAL_SOURCE_BY_SCENARIO[scenario],
            "death.external.description": CHILD_EXTERNAL_DESCRIPTIONS.get((chain.chain_id, place_group))
                                          or EXTERNAL_DESCRIPTIONS[(scenario, place_group)],
            "death.external.place.municipality": profile["city"], "death.external.place.uf": profile["uf"],
        })
    else:
        values.update({
            "death.external.type": None, "death.external.work_accident": None, "death.external.source": None,
            "death.external.description": None, "death.external.place.municipality": None, "death.external.place.uf": None,
        })

    if is_do:
        values.update(_render_chain(rng, chain, category, scenario, age_unit_minutes, age_group))
    else:
        values.update({f"death.cause.{letter}_{suffix}": None
                       for letter in _CAUSE_LINE_LETTERS for suffix in ("text", "cid", "interval")})
        values.update({"death.cause.chain_id": None, "death.cause.ii1_text": None, "death.cause.ii1_cid": None,
                       "death.cause.ii1_interval": None, "death.cause.ii2_text": None, "death.cause.ii2_cid": None,
                       "death.cause.ii2_interval": None})
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
    street, cep, neighborhood = pick_street(record, rng, "deceased.occurrence.street", profile)
    return {
        "deceased.occurrence.address": street,
        "deceased.occurrence.number": house_number(record, rng, "deceased.occurrence.number"),
        "deceased.occurrence.complement": (record.pick(rng, "deceased.occurrence.complement", COMPLEMENTS)
                                           if rng.random() < 0.3 else None),
        "deceased.occurrence.cep": cep, "deceased.occurrence.neighborhood": neighborhood,
    }


def deceased(record, rng: random.Random) -> dict[str, Any]:
    is_do = record.context == "declaracao_obito"
    on = record.value("death.date")
    infant_fields: dict[str, Any] = {}
    if is_do:
        from src.semantic.cause_chains import CAUSE_CHAINS
        chain_id = record.value("death.cause.chain_id")
        chain = next(c for c in CAUSE_CHAINS if c.chain_id == chain_id)
        if chain.age_unit == "anos":
            # R1: the age was already drawn in death() (under the "deceased.age" key, so an
            # AI-required age still filtered it there) — reuse it instead of drawing again, so
            # death()'s Part I/Part II interval caps are guaranteed consistent with this age.
            age = record.value("death.cause.age_unit_value")
            born = record.pick(rng, "deceased.birth_date", birth_dates_for_age(age, on))
        else:
            infant_value = record.value("death.cause.age_unit_value") if chain.age_unit != "fetal" else None
            death_time = record.value("death.time")
            infant_fields = infant_or_fetal_age(record, rng, chain, on, infant_value, death_time)
            age, born = infant_fields["deceased.age"], infant_fields["deceased.birth_date"]
        sex = chain.sex or record.pick(rng, "deceased.sex", ("M", "F"), chain.sex_weights or (52, 48))
    else:
        chain = None
        age, born = age_and_birth_date(record, rng, "deceased", on, DECEASED_AGES, DECEASED_AGE_WEIGHTS)
        sex = record.pick(rng, "deceased.sex", ("M", "F"), (52, 48))
    age_group = record.value("death.age_group") if is_do else "adult"
    name = (record.draw("deceased.name", rng, lambda: "Natimorto") if age_group == "fetal"
            else record.draw("deceased.name", rng, lambda: person_name(record, rng, sex)))
    # A fetal death ("Natimorto") has no surname of its own to derive the parents' surnames from
    # (parent_names() would otherwise literally surname them "Natimorto") — draw the parents
    # independently instead, same as any other unrelated pair of names.
    mother_name, father_name = ((person_name(record, rng, "F"), person_name(record, rng, "M")) if age_group == "fetal"
                                else parent_names(record, rng, name))
    category = record.value("death.place.category") if is_do else None
    # Home deaths (category "3") must be in the deceased's own residence city (M1) — a death at
    # home is, by definition, in the municipality the person lived in. Otherwise (item 1), Lei
    # 6.015/73 keeps registration at the place of death OR of residence, and in practice an
    # interstate death (e.g. referred to a capital for hospital care) is the exception: ~5%,
    # not the ~23% the old 0.7 weight gave, and favoring() now biases any cross-state pick
    # toward a neighbouring region instead of spreading it uniformly over the whole country.
    residence_bias = 1.0 if (is_do and category == "3") else 0.95
    residence = pick_profile(record, rng, "deceased.residence", favoring(profile_of(record, "death.place"), residence_bias))
    # M1: for a fetal/neonatal in-hospital death, naturalidade is the municipality of
    # delivery/death, not an arbitrary one — mostly the death municipality itself (fetal/early
    # neonatal: delivery and death are the same hospital), and for older infants mostly the
    # death municipality or residence.
    if is_do and age_group in ("fetal", "infant"):
        death_profile_for_birthplace = profile_of(record, "death.place")
        birthplace = record.pick(rng, "deceased.naturalidade_profile", LOCATION_PROFILES,
                                 [92 if p is death_profile_for_birthplace else 8 / (len(LOCATION_PROFILES) - 1)
                                  for p in LOCATION_PROFILES])
    else:
        birthplace = rng.choice(LOCATION_PROFILES)
    # "Estudante" is implausible for an elderly DO death (M4) — exclude it from the pool once
    # the person is well past typical student age in the DO context. M2: under 18, occupation
    # is "Estudante" (>= 6) or "Não se aplica" — never the full adult occupation pool (no
    # "Engenheiro"/"Superior completo" at 14).
    # R2/m-a: OCCUPATION_MIN_AGE (a graduate title needs ~22, a technical/"Médio" title ~16) is
    # applied to the DO deceased's own pool too — this is what actually fixed the 78 DO adults
    # aged 18-20 with "Superior completo": the schooling cap alone cannot catch it, because the
    # occupation-driven fast path in _schooling_for_age is unconditional once the occupation
    # itself is drawn. Outside the DO this filter is skipped (byte-identical to before phase 2).
    occupation_pool = tuple(o for o in OCCUPATIONS if o != "Estudante") if (is_do and age >= 65) else OCCUPATIONS
    if is_do:
        occupation_pool = tuple(o for o in occupation_pool if age >= OCCUPATION_MIN_AGE.get(o, 0))
    if is_do and age < 6:
        occupation_canonical = "Não se aplica"
    elif is_do and age < 18:
        occupation_canonical = rng.choices(("Estudante", "Não se aplica"), (90, 10))[0]
    elif not is_do and age < 14:
        # Outside the DO (other documents sharing this generator) keep the exact pre-phase-2
        # rule — M2's "Estudante <18" fix is DO-specific, so those documents stay byte-identical.
        occupation_canonical = "Não se aplica"
    else:
        occupation_canonical = record.pick(rng, "deceased.occupation", occupation_pool)
    # R-c: always inflect to the deceased's sex, even when the AI required a specific
    # canonical value — assisted.py widens that requirement to accept either grammatical
    # form, so the inflected value still satisfies it.
    occupation = inflect_occupation(occupation_canonical, sex)
    residence_fields = address("deceased.residence", residence, house_number(record, rng, "deceased.residence.number"),
                               ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge"),
                               *pick_street(record, rng, "deceased.residence.street", residence))
    # Field 9_anos ("Idade - Anos completos") genuinely has no value for fetal/infant deaths —
    # not just a blank render: the field's own validation (min=1) would otherwise reject the
    # internal age=0 used above for occupation/schooling logic, so the EXPORTED value is None
    # (field 9's own "menores de 1 ano" / "fetal" blocks are what carry the age instead).
    exported_age = None if age_group in ("fetal", "infant") else age
    values = {
        "deceased.name": name, "deceased.sex": sex, "deceased.birth_date": born, "deceased.age": exported_age,
        "deceased.cpf": record.draw("deceased.cpf", rng, lambda: cpf_digits(rng)),
        "deceased.naturalidade": f"{birthplace['city']} - {birthplace['uf']}",
        "deceased.naturalidade.municipality": birthplace["city"], "deceased.naturalidade.uf": birthplace["uf"],
        "deceased.mother_name": record.draw("deceased.mother_name", rng, lambda: mother_name),
        "deceased.father_name": record.draw("deceased.father_name", rng, lambda: father_name),
        "deceased.parents_names": f"{mother_name}; {father_name}",
        "deceased.occupation": occupation, "deceased.occupation_cbo": OCCUPATION_CODES.get(occupation_canonical),
        "deceased.age_minutes": infant_fields.get("deceased.age_minutes"), "deceased.age_hours": infant_fields.get("deceased.age_hours"),
        "deceased.age_days": infant_fields.get("deceased.age_days"), "deceased.age_months": infant_fields.get("deceased.age_months"),
        **residence_fields,
    }
    if is_do:
        values.update(deceased_birth_block(record, rng, age_group, chain))
        # Field 37 ("a morte ocorreu...") targets any woman of fertile age (10-49) at death,
        # independent of scenario — the vast majority of such deaths are unrelated to pregnancy.
        # M5: "No parto"/"No aborto" imply an obstetric (O-code) cause, which no chain in this
        # library models yet — so those two values are dropped until such chains exist; the
        # realistic "Na gravidez"/puerperal rates are also lowered (~1-2% each, not ~8% total).
        if sex == "F" and age_group in ("adult", "child") and 10 <= age <= 49:
            values["deceased.fertile_context"] = record.pick(
                rng, "deceased.fertile_context",
                ("Não ocorreu nestes períodos", "Na gravidez", "Até 42 dias após o parto",
                 "De 43 dias a 1 ano após o parto", "Ignorado"),
                (93.5, 1.5, 1.5, 1.5, 2))
        else:
            values["deceased.fertile_context"] = None
    else:
        values.update(deceased_birth_block(record, rng, "adult", None))
        values["deceased.fertile_context"] = None
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
    # R2/m-a: the age cap now applies to the DO deceased's own schooling at EVERY age group, not
    # only children — this is what fixed 78 DO adults aged 18-20 showing "Superior completo"
    # (the occupation fast path inside _schooling_for_age is unconditional, so a real graduate
    # occupation like "Advogada" is unaffected: OCCUPATION_MIN_AGE already keeps it out of the
    # pool before the real minimum age).
    schooling, series = _schooling_for_age(rng, age, occupation_canonical if is_do else None,
                                           cap_by_age=is_do)
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
    if is_do and age_group == "fetal":
        # M1: a stillbirth has no CNS (field 3), marital status (12), schooling (13) or
        # occupation (14) of its own — the DO instructions describe these as fields of the
        # DECEASED, and a fetus never acquired any of them; left entirely BLANK (None), not
        # "Não se aplica" (that value is for an infant/child who is simply too young, which
        # still legally has e.g. a marital status concept — a fetal death does not).
        values["deceased.cns"] = None
        values["deceased.marital_status"] = None
        values["deceased.marital_status_do"] = None
        values["deceased.spouse_name"] = None
        values["deceased.schooling"] = None
        values["deceased.schooling_series"] = None
        values["deceased.occupation"] = None
        values["deceased.occupation_cbo"] = None
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
    # Lei 6.015/73 art. 77: registration at the place of death or of residence (item 1) — so the
    # cartório UF should almost always be one of the two, not an unrelated third state.
    death_profile = profile_of(record, "death.place")
    residence_profile = profile_of(record, "deceased.residence")
    office = pick_profile(record, rng, "death_registration.office",
                          office_weights(death_profile, residence_profile, 0.75, 0.18), ("municipality", "uf"))
    office_name = record.pick(rng, "death_registration.office.name", REGISTRY_OFFICE_NAMES)
    values = {
        "death_registration.date": day,
        "death_registration.number": record.draw("death_registration.number", rng, lambda: str(rng.randint(1, 250000))),
        "death_registration.office.name": office_name,
        "death_registration.office.municipality": office["city"], "death_registration.office.uf": office["uf"],
        "death_registration.place.summary": f"{office['city']} - {office['uf']}",
    }
    sex = rng.choice(("F", "M"))
    values["death_registration.declarant_name"] = record.draw("death_registration.declarant_name", rng,
                                                               lambda: person_name(record, rng, sex))
    values["death_registration.matricula"] = build_matricula(rng, office_cns(record, office_name, office["ibge"]), day.year, "4")
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
        # Outside the DO, the certifier's role is not rendered, but it still feeds
        # certifier.iml_svo_* fields; keeping it random let an IML/SVO certifier appear
        # together with a "Não" necropsy context that doesn't exist on these documents (item 3).
        # Fix it to "1" (médico assistente) so no document models an IML/SVO without a DO.
        role = "1"
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


def build_matricula(rng: random.Random, cns: str, year: int, livro: str) -> str:
    return matricula(cns, year, livro, rng.randint(1, 400), rng.randint(1, 300), rng.randint(1, 250000))


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


SAME_SEX_MARRIAGE_START = date(2013, 5, 16)  # CNJ Resolução 175/2013
SAME_SEX_MARRIAGE_SHARE = 0.01


def couple_sexes(record) -> tuple[str, str]:
    """The sexes of the two spouses, drawn together. A same-sex couple marries only from
    SAME_SEX_MARRIAGE_START, at about SAME_SEX_MARRIAGE_SHARE of the marriages of that period."""
    def draw(rng: random.Random) -> tuple[str, str]:
        required = record.required_values("marriage.date") if hasattr(record, "required_values") else None
        later = (EVENT_END - SAME_SEX_MARRIAGE_START).days / (EVENT_END - EVENT_START).days
        share = SAME_SEX_MARRIAGE_SHARE * later / (1 - SAME_SEX_MARRIAGE_SHARE + SAME_SEX_MARRIAGE_SHARE * later)
        if required is not None and not any(day >= SAME_SEX_MARRIAGE_START for day in required):
            share = 0.0
        if rng.random() < share:
            return rng.choice((("F", "F"), ("M", "M")))
        return rng.choice((("F", "M"), ("M", "F")))
    return record.link("marriage.couple", draw)


def _spouse(record, rng: random.Random, prefix: str) -> dict[str, Any]:
    sex = couple_sexes(record)[int(prefix[-1]) - 1]
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
    same_sex = len(set(couple_sexes(record))) == 1
    day = record.pick(rng, "marriage.date", DateSpan(SAME_SEX_MARRIAGE_START if same_sex else EVENT_START, EVENT_END))
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
    return {"marriage_registration.date": day,
            "marriage_registration.matricula": build_matricula(rng, record.value("office.cns"), day.year, "2")}


CLERK_ROLES = ("OFICIAL DE REGISTRO CIVIL", "OFICIAL SUBSTITUTO", "ESCREVENTE AUTORIZADO", "ESCREVENTE SUBSTITUTO")
CLERK_ROLE_WEIGHTS = (55, 15, 20, 10)
OFFICE_REGISTRATION_DATE_KEYS = {"certidao_nascimento": "registration.date", "declaracao_obito": "death_registration.date",
                                 "certidao_casamento": "marriage_registration.date"}


def office(record, rng: random.Random) -> dict[str, Any]:
    context = record.context
    if context == "certidao_nascimento":
        municipality, uf = record.value("registration.office.municipality"), record.value("registration.office.uf")
        name = record.value("registration.office.name")
    elif context == "declaracao_obito":
        municipality, uf = record.value("death_registration.office.municipality"), record.value("death_registration.office.uf")
        name = record.value("death_registration.office.name")
    else:
        chosen = rng.choice(LOCATION_PROFILES)
        municipality, uf = chosen["city"], chosen["uf"]
        name = rng.choice(REGISTRY_OFFICE_NAMES)
    profile = next(item for item in LOCATION_PROFILES if item["city"] == municipality)
    number = house_number(record, rng, "office.address_number")
    return {
        "office.cns": office_cns(record, name, profile["ibge"]),
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
    if start >= MATRICULA_START:
        span = DateSpan(start, min(start + timedelta(days=5 * 365), EVENT_END))
        weights = [8 if index <= 30 else 2 if index <= 365 else 0.2 for index in range(len(span))]
    else:
        # The certidão with a matrícula exists only from 2010 (Provimento CNJ 3/2009): a copy of an
        # older registration is issued at a date between 2010 and the reference date.
        span, weights = DateSpan(MATRICULA_START, EVENT_END), None
    day = record.pick(rng, "certificate.issue_date", span, weights)
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
