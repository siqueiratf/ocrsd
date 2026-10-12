import json
import re
import time as clock
import unittest
import unicodedata
from datetime import date, time, timedelta

from src.semantic import REGISTRY, VERSION, SemanticConstraintError, SemanticField, SemanticRecord, semantic_field, semantic_options
from src.semantic.constraints import CONSTRAINTS, violations
from src.semantic.data import FEMALE_ASSOCIATED_FIRST_NAMES, FIRST_NAMES, LOCATION_PROFILES, MALE_ASSOCIATED_FIRST_NAMES, UF_VALUES
from src.semantic.generators import GENERATORS
from src.semantic.registry import ENTITY_KEYS, KINDS

REQUIRED_KEYS = """
newborn.name newborn.sex newborn.cpf newborn.naturalidade mother.name mother.birth_date mother.age mother.cpf mother.cns
mother.occupation mother.naturalidade mother.residence.street mother.residence.number mother.residence.complement
mother.residence.neighborhood mother.residence.cep mother.residence.municipality mother.residence.uf mother.residence.ibge
mother.obstetric.previous_pregnancies mother.obstetric.vaginal_births mother.obstetric.cesareans mother.obstetric.live_births
mother.obstetric.fetal_losses father.name father.birth_date father.age guardian.name guardian.age parents.names
grandparents.names birth.date birth.time birth.establishment.name birth.establishment.cnes birth.place.street birth.place.number
birth.place.neighborhood birth.place.cep birth.place.municipality birth.place.uf birth.place.ibge birth.place.summary
pregnancy.weeks pregnancy.last_menstrual_period pregnancy.prenatal_visits pregnancy.prenatal_start_month filling.date
filling.responsible.name filling.responsible.document_number filling.responsible.issuer registration.date registration.number
registration.matricula registration.dnv_number registration.office.name registration.office.municipality registration.office.uf
registration.place.summary deceased.name deceased.sex deceased.birth_date deceased.age deceased.cpf deceased.mother_name
deceased.father_name deceased.occupation deceased.residence.street deceased.residence.number deceased.residence.neighborhood
deceased.residence.cep deceased.residence.municipality deceased.residence.uf deceased.residence.ibge death.date death.time
death.place.establishment.name death.place.municipality death.place.uf death.place.ibge certifier.name certifier.registration
birth.pregnancy_type birth.delivery_type birth.cesarean_before_labor newborn.anomaly_status newborn.anomaly_description mother.occupation_cbo
birth.twins twin.name_matricula death_registration.date death_registration.number death_registration.office.name
death_registration.office.municipality death_registration.office.uf death_registration.place.summary
""".split()
ADDRESS_PREFIXES = ("birth.place", "mother.residence", "deceased.residence", "death.place", "registration.office")
PROFILE_FIELDS = {"municipality": "city", "uf": "uf", "ibge": "ibge", "cep": "cep", "street": "street", "neighborhood": "neighborhood"}


def completed_years(born: date, on: date) -> int:
    years = on.year - born.year
    return years - 1 if (on.month, on.day) < (born.month, born.day) else years


def valid_cpf(value: str) -> bool:
    if not re.fullmatch(r"\d{11}", value):
        return False
    numbers = [int(digit) for digit in value]
    for length in (9, 10):
        total = sum(number * weight for number, weight in zip(numbers, range(length + 1, 1, -1)))
        if numbers[length] != (0 if total % 11 < 2 else 11 - total % 11):
            return False
    return True


def matches_kind(kind: str, value) -> bool:
    digit_widths = {"cep": 8, "ibge": 7, "cnes": 7, "cns": 15, "dnv_number": 11, "matricula": 32}
    if kind in digit_widths:
        return isinstance(value, str) and re.fullmatch(rf"\d{{{digit_widths[kind]}}}", value) is not None
    checks = {
        "name": lambda: isinstance(value, str) and len(value.split()) >= 2,
        "sex": lambda: value in {"M", "F"},
        "date": lambda: type(value) is date,
        "time": lambda: isinstance(value, time) and value.second == 0 and value.microsecond == 0,
        "integer": lambda: type(value) is int,
        "boolean": lambda: type(value) is bool,
        "uf": lambda: value in UF_VALUES,
        "cpf": lambda: valid_cpf(value),
        "document_number": lambda: isinstance(value, str) and value.isdigit(),
        "house_number": lambda: isinstance(value, str) and value.isdigit(),
        "issuer": lambda: isinstance(value, str) and value.isalpha() and value.isupper(),
    }
    return checks.get(kind, lambda: isinstance(value, str) and bool(value.strip()))()


def resolved(seed: int, **kwargs) -> tuple[SemanticRecord, dict]:
    record = SemanticRecord(seed, **kwargs)
    return record, record.resolve(REQUIRED_KEYS)


class RegistryTests(unittest.TestCase):
    def test_registry_exposes_contract_keys(self):
        self.assertEqual(set(REQUIRED_KEYS), set(REGISTRY))
        self.assertIsInstance(VERSION, str)
        for key, field in REGISTRY.items():
            self.assertIsInstance(field, SemanticField)
            self.assertEqual(field.key, key)
            self.assertIn(field.kind, KINDS)
            self.assertIn(field.entity, GENERATORS)
            self.assertTrue(field.label and field.description and field.group and field.contexts and field.synonyms)
            self.assertTrue(set(field.depends_on) <= set(REGISTRY), key)
            self.assertTrue(set(field.contexts) <= {"dnv", "certidao_nascimento", "declaracao_obito"})
            for synonym in field.synonyms:
                stripped = "".join(c for c in unicodedata.normalize("NFKD", synonym) if not unicodedata.combining(c)).lower()
                self.assertEqual(synonym, stripped, key)
        self.assertEqual(set(ENTITY_KEYS), set(GENERATORS))

    def test_lookup_and_options(self):
        self.assertEqual(semantic_field("mother.age").kind, "integer")
        with self.assertRaises(KeyError):
            semantic_field("mother.shoe_size")
        options = semantic_options()
        self.assertEqual([option["key"] for option in options], list(REGISTRY))
        self.assertEqual(set(options[0]), {"key", "label", "group", "kind", "description"})
        json.dumps(options)

    def test_name_lists_cover_existing_names_without_overlap(self):
        self.assertTrue(set(FIRST_NAMES) <= set(FEMALE_ASSOCIATED_FIRST_NAMES) | set(MALE_ASSOCIATED_FIRST_NAMES))
        self.assertFalse(set(FEMALE_ASSOCIATED_FIRST_NAMES) & set(MALE_ASSOCIATED_FIRST_NAMES))


class SemanticRecordTests(unittest.TestCase):
    def test_every_key_generates_declared_kind(self):
        for seed in range(25):
            _, values = resolved(seed)
            for key, value in values.items():
                if value is None:
                    self.assertTrue(REGISTRY[key].optional, key)
                    continue
                self.assertTrue(matches_kind(REGISTRY[key].kind, value), f"{seed} {key}={value!r}")
            self.assertTrue(values["registration.dnv_number"].startswith("00"))
            self.assertTrue(4 <= len(values["certifier.registration"]) <= 6)

    def test_dates_and_ages_over_many_seeds(self):
        for seed in range(300):
            _, values = resolved(seed)
            born, birth_day = values["mother.birth_date"], values["birth.date"]
            self.assertTrue(date(1980, 1, 1) <= birth_day <= date(2025, 12, 31))
            self.assertLess(born, birth_day)
            self.assertEqual(values["mother.age"], completed_years(born, birth_day))
            self.assertTrue(10 <= values["mother.age"] <= 60)
            self.assertEqual(values["father.age"], completed_years(values["father.birth_date"], birth_day))
            self.assertTrue(12 <= values["father.age"] <= 90)
            self.assertTrue(values["guardian.age"] is None or values["guardian.age"] >= 18)
            self.assertEqual(values["guardian.name"] is None, values["guardian.age"] is None)
            self.assertNotEqual(values["guardian.name"], values["mother.name"])
            for key in ("birth.date", "filling.date", "registration.date", "death.date", "death_registration.date"):
                self.assertLessEqual(values[key], date(2025, 12, 31), key)
            self.assertGreaterEqual(values["death_registration.date"], values["death.date"])
            self.assertGreaterEqual(values["registration.date"], birth_day)
            self.assertTrue(birth_day <= values["filling.date"] <= birth_day + timedelta(days=60))
            self.assertLessEqual(values["filling.date"], values["registration.date"])
            weeks = values["pregnancy.weeks"]
            self.assertTrue(22 <= weeks <= 42)
            self.assertLessEqual(abs((birth_day - timedelta(days=7 * weeks) - values["pregnancy.last_menstrual_period"]).days), 6)
            self.assertTrue(1 <= values["pregnancy.prenatal_start_month"] <= 9)
            self.assertTrue(0 <= values["pregnancy.prenatal_visits"] <= 20)
            history = {key.rsplit(".", 1)[1]: value for key, value in values.items() if key.startswith("mother.obstetric.")}
            self.assertTrue(all(value >= 0 for value in history.values()))
            self.assertEqual(history["vaginal_births"] + history["cesareans"], history["previous_pregnancies"] - history["fetal_losses"])
            self.assertLessEqual(history["live_births"], history["vaginal_births"] + history["cesareans"])
            self.assertLessEqual(values["deceased.birth_date"], values["death.date"])
            self.assertEqual(values["deceased.age"], completed_years(values["deceased.birth_date"], values["death.date"]))
            self.assertTrue(0 <= values["deceased.age"] <= 110)
            self.assertEqual(values["registration.matricula"][10:14], str(values["registration.date"].year))
            self.assertEqual(values["registration.number"], str(int(values["registration.matricula"][23:30])))

    def test_addresses_come_from_one_location_profile(self):
        for seed in range(150):
            _, values = resolved(seed)
            for prefix in ADDRESS_PREFIXES:
                present = {field: values[f"{prefix}.{part}"] for part, field in PROFILE_FIELDS.items() if f"{prefix}.{part}" in values}
                matches = [profile for profile in LOCATION_PROFILES if all(profile[field] == value for field, value in present.items())]
                self.assertEqual(len(matches), 1, f"{seed} {prefix} {present}")
            self.assertEqual(values["birth.place.summary"],
                             f"{values['birth.establishment.name']}, {values['birth.place.municipality']} - {values['birth.place.uf']}")
            self.assertEqual(values["registration.place.summary"],
                             f"{values['registration.office.municipality']} - {values['registration.office.uf']}")

    def test_shared_identities(self):
        same_residence = 0
        for seed in range(200):
            _, values = resolved(seed)
            self.assertEqual(values["parents.names"], f"{values['mother.name']} e {values['father.name']}")
            pairs = values["grandparents.names"].split("; ")
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(len(pair.split(" e ")) == 2 for pair in pairs))
            self.assertIn(values["newborn.naturalidade"], {values["birth.place.municipality"], values["mother.residence.municipality"]})
            same_residence += values["birth.place.ibge"] == values["mother.residence.ibge"]
        self.assertTrue(100 < same_residence < 200)

    def test_surnames_are_statistically_inherited(self):
        inherited = 0
        for seed in range(200):
            record = SemanticRecord(seed)
            names = record.resolve(["newborn.name", "mother.name", "father.name"])
            family = set(names["mother.name"].split()[1:]) | set(names["father.name"].split()[1:])
            inherited += bool(set(names["newborn.name"].split()[1:]) & family)
        self.assertGreater(inherited, 160)

    def test_maternal_surname_is_a_generator_preference_not_a_constraint(self):
        for seed in range(300):
            values = SemanticRecord(seed).resolve(["newborn.name", "mother.name"])
            self.assertTrue(set(values["newborn.name"].split()[1:]) & set(values["mother.name"].split()[1:]), values)
        broken = {**resolved(3)[1], "newborn.name": "Ana Zzz"}
        self.assertNotIn("sobrenome_materno_do_recem_nascido", {constraint.name for constraint, _ in violations(broken)})

    def test_optional_guardian_and_twins(self):
        minors = adults = minor_guardians = adult_guardians = twins = 0
        for seed in range(600):
            values = SemanticRecord(seed).resolve(["guardian.name", "guardian.age", "mother.age", "birth.twins", "twin.name_matricula",
                                                   "newborn.name", "registration.matricula"])
            present = values["guardian.name"] is not None
            if values["mother.age"] < 18:
                minors, minor_guardians = minors + 1, minor_guardians + present
            else:
                adults, adult_guardians = adults + 1, adult_guardians + present
            if values["birth.twins"]:
                twins += 1
                name, matricula = values["twin.name_matricula"].split(", ")
                self.assertNotEqual(name, values["newborn.name"])
                self.assertEqual(name.split()[1:], values["newborn.name"].split()[1:])
                self.assertEqual(re.sub(r"\D", "", matricula)[:23], values["registration.matricula"][:23])
            else:
                self.assertIsNone(values["twin.name_matricula"])
        self.assertGreater(minor_guardians, 0.75 * minors)
        self.assertLess(adult_guardians, 0.12 * adults)
        self.assertTrue(2 <= twins <= 30, twins)
        forced = SemanticRecord(5)
        forced.require("mother.age", range(14, 17), "mãe adolescente")
        self.assertTrue(forced.resolve(["mother.age", "guardian.name"]))

    def test_death_registration_follows_the_death(self):
        for seed in range(150):
            record = SemanticRecord(seed, "declaracao_obito")
            values = record.resolve(["death_registration.date", "death_registration.place.summary", "deceased.name"])
            self.assertTrue(record.value("death.date") <= values["death_registration.date"] <= date(2025, 12, 31))
            self.assertEqual(values["death_registration.place.summary"],
                             f'{record.value("death_registration.office.municipality")} - {record.value("death_registration.office.uf")}')
            self.assertFalse({"newborn.name", "mother.name", "filling.date"} & set(record.snapshot()))

    def test_determinism_and_order_independence(self):
        for seed in (0, 5, 123456789):
            first, values = resolved(seed)
            second, _ = resolved(seed)
            self.assertEqual(first.snapshot(), second.snapshot())
            reversed_record = SemanticRecord(seed)
            self.assertEqual(reversed_record.resolve(list(reversed(REQUIRED_KEYS))), values)
            for subset in (["mother.age"], ["registration.matricula", "newborn.name"], ["death.date", "pregnancy.weeks"]):
                self.assertEqual(SemanticRecord(seed).resolve(subset), {key: values[key] for key in subset})
            lazy = SemanticRecord(seed)
            self.assertEqual(lazy.value("grandparents.names"), values["grandparents.names"])
        self.assertNotEqual(resolved(1)[0].snapshot(), resolved(2)[0].snapshot())

    def test_requirements_are_honoured(self):
        for seed in range(40):
            record = SemanticRecord(seed)
            record.require("mother.age", {17}, "idade fixa")
            record.require("birth.date", lambda value: value.year == 2020, "nascidos em 2020")
            values = record.resolve(["mother.age", "mother.birth_date", "newborn.name"])
            self.assertEqual(values["mother.age"], 17)
            self.assertEqual(record.value("birth.date").year, 2020)
            self.assertEqual(completed_years(values["mother.birth_date"], record.value("birth.date")), 17)

    def test_multiple_requirements_intersect(self):
        record = SemanticRecord(9)
        record.require("pregnancy.weeks", range(30, 40), "faixa do campo")
        record.require("pregnancy.weeks", lambda value: value % 2 == 1, "ímpar")
        weeks = record.resolve(["pregnancy.weeks", "pregnancy.last_menstrual_period"])["pregnancy.weeks"]
        self.assertIn(weeks, {31, 33, 35, 37, 39})
        located = SemanticRecord(9)
        located.require("birth.place.uf", {"PR"}, "UF do campo")
        located.require("registration.office.municipality", {"Curitiba"}, "cartório")
        values = located.resolve(["birth.place.municipality", "birth.place.ibge", "registration.office.uf"])
        self.assertEqual(values, {"birth.place.municipality": "Curitiba", "birth.place.ibge": "4106902", "registration.office.uf": "PR"})

    def test_requirement_on_free_value_uses_allowed_values(self):
        record = SemanticRecord(3)
        record.require("mother.name", ["Maria Souza Lima"], "valores permitidos")
        values = record.resolve(["parents.names", "mother.name"])
        self.assertTrue(values["parents.names"].startswith("Maria Souza Lima e "))

    def test_impossible_requirement_fails_quickly_with_readable_message(self):
        started = clock.perf_counter()
        record = SemanticRecord(4)
        record.require("mother.age", lambda value: value == 5, "idade impossível")
        with self.assertRaises(SemanticConstraintError) as raised:
            record.resolve(["mother.name"])
        self.assertLess(clock.perf_counter() - started, 5)
        self.assertIn("mother.age", str(raised.exception))
        self.assertIn("idade impossível", str(raised.exception))
        self.assertTrue(raised.exception.violations)
        self.assertIsInstance(raised.exception, ValueError)

    def test_constraints_reject_inconsistent_values(self):
        _, values = resolved(11)
        self.assertEqual(violations(values), [])
        broken = {**values, "mother.age": values["mother.age"] + 1, "birth.place.uf": "AC",
                  "parents.names": "Alguém e Outra Pessoa", "filling.date": values["birth.date"] + timedelta(days=90)}
        names = {constraint.name for constraint, _ in violations(broken)}
        self.assertTrue({"idade_da_mae_no_nascimento", "endereco_coerente:birth.place", "filiacao_coerente",
                         "preenchimento_ate_60_dias"} <= names)
        self.assertTrue(all(constraint.regenerate and constraint.keys for constraint in CONSTRAINTS))

    def test_snapshot_is_json_serializable(self):
        record, values = resolved(21)
        snapshot = json.loads(json.dumps(record.snapshot(), ensure_ascii=False))
        self.assertEqual(set(snapshot), set(REQUIRED_KEYS))
        self.assertEqual(snapshot["birth.date"], values["birth.date"].isoformat())
        self.assertRegex(snapshot["birth.time"], r"^\d{2}:\d{2}$")

    def test_name_sex_association_is_statistical(self):
        outcomes = set()
        for seed in range(120):
            values = SemanticRecord(seed, name_sex_association=0.5).resolve(["newborn.name", "newborn.sex"])
            first = values["newborn.name"].split()[0]
            outcomes.add((values["newborn.sex"], first in FEMALE_ASSOCIATED_FIRST_NAMES))
        self.assertEqual(outcomes, {("F", True), ("F", False), ("M", True), ("M", False)})
        for seed in range(30):
            values = SemanticRecord(seed, name_sex_association=0.0).resolve(["newborn.name", "newborn.sex", "deceased.name", "deceased.sex"])
            for person in ("newborn", "deceased"):
                female_name = values[f"{person}.name"].split()[0] in FEMALE_ASSOCIATED_FIRST_NAMES
                self.assertEqual(female_name, values[f"{person}.sex"] == "M")

    def test_links_share_values_within_a_record(self):
        record = SemanticRecord(1)
        first = record.link("hospital", lambda rng: rng.randrange(10 ** 6))
        self.assertEqual(record.link("hospital", lambda rng: -1), first)
        self.assertEqual(SemanticRecord(1).link("hospital", lambda rng: rng.randrange(10 ** 6)), first)
        self.assertEqual(record.link("fixo", lambda: "valor"), "valor")

    def test_unknown_keys_raise_key_error(self):
        record = SemanticRecord(0)
        for call in (lambda: record.value("x.y"), lambda: record.require("x.y", {1}, "r"), lambda: record.resolve(["x.y"])):
            with self.assertRaises(KeyError):
                call()


if __name__ == "__main__":
    unittest.main()
