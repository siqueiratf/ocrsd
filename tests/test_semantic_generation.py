import json
import os
import subprocess
import sys
import tempfile
import time as clock
import unittest
from datetime import date, datetime
from pathlib import Path

import fitz

from src.field_rules import format_date
from src.semantic import VERSION
from src.semantic.constraints import completed_years
from src.semantic.data import LOCATION_PROFILES
from src.synthetic_dataset import field_texts, generate_dataset
from src.web.models import GenerationOptions
from src.web.services import Catalog, ServiceError, mapping_issues, semantic_inferred, validate_mapping
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
DNV_TEMPLATE = ROOT / "documents" / "dnv" / "template.pdf"
DNV_MAPPING = ROOT / "documents" / "dnv" / "mappings" / "mapeamento_seguindo_norma.json"
DNV_LEGACY = ROOT / "documents" / "dnv" / "mappings" / "test_w_datetime.json"
CERT_TEMPLATE = ROOT / "documents" / "certidao_nascimento" / "certidao_nascimento.pdf"
CERT_MAPPING = ROOT / "documents" / "certidao_nascimento" / "certidao_nascimento_fields.json"
COUNT = 8


def all_ids(mapping: Path) -> set[str]:
    return {str(field["field_id"]) for field in json.loads(mapping.read_text(encoding="utf-8"))["fields"]}


def generate(template: Path, mapping: Path, output: Path, count: int = COUNT, seed: int = 21, ids: set[str] | None = None) -> list[dict]:
    documents = generate_dataset(template, mapping, ids or all_ids(mapping), output, count, seed=seed,
                                 handwriting=False, text_handwriting=False)
    return [json.loads(document.json_path.read_text(encoding="utf-8")) for document in documents]


def fields(truth: dict) -> dict[str, dict]:
    return {field["field_id"]: field for field in truth["fields"]}


def ddmmyyyy(text: str) -> date:
    return datetime.strptime(text, "%d%m%Y").date()


def profile(**parts) -> dict | None:
    return next((item for item in LOCATION_PROFILES if all(item[key] == value for key, value in parts.items())), None)


def blank_page(path: Path) -> None:
    with fitz.open() as document:
        document.new_page(width=600, height=400)
        document.save(path)


def write_mapping(path: Path, template_id: str, entries: list[dict]) -> None:
    path.write_text(json.dumps({"template_id": template_id, "dpi": 72, "page_size_px": [600, 400], "fields": entries}), encoding="utf-8")


class SemanticGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.dnv = generate(DNV_TEMPLATE, DNV_MAPPING, root / "dnv")
        cls.certidao = generate(CERT_TEMPLATE, CERT_MAPPING, root / "certidao")
        cls.manifest = json.loads((root / "dnv" / "dataset.json").read_text(encoding="utf-8"))
        cls.cert_lists = {str(field["field_id"]): field.get("validation", {}).get("allowed_values")
                          for field in json.loads(CERT_MAPPING.read_text(encoding="utf-8"))["fields"]}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_dnv_conditional_fields_follow_the_semantic_record(self):
        pregnancy = {"Única": "tipo_gravidez_0", "Dupla": "tipo_gravidez_1"}
        delivery = {"Vaginal": "tipo_parto_0", "Cesáreo": "tipo_parto_1"}
        cesarean = {"Não se aplica": "cesarea_ocorreu_antes_do_trabalho_de_parto_iniciar_2",
                    "Ignorado": "cesarea_ocorreu_antes_do_trabalho_de_parto_iniciar_3"}
        for truth in self.dnv:
            record, by_id = truth["semantic_record"], fields(truth)
            self.assertEqual(by_id["35"]["value"], pregnancy[record["birth.pregnancy_type"]])
            self.assertEqual(by_id["38"]["value"], delivery[record["birth.delivery_type"]])
            self.assertEqual(by_id["39"]["value"], cesarean[record["birth.cesarean_before_labor"]])
            self.assertEqual(record["birth.pregnancy_type"], "Dupla" if record["birth.twins"] else "Única")
            self.assertEqual(by_id["6"]["semantic"]["value"], "Ignorado")
            self.assertEqual(by_id["41"]["value"], "")
            self.assertTrue(by_id["41"]["blank"])
            self.assertEqual(by_id["17_codigo_cbo"]["value"], record["mother.occupation_cbo"] or "")

    def test_dnv_values_come_from_one_coherent_record(self):
        self.assertEqual(self.manifest["semantic_version"], VERSION)
        for truth in self.dnv:
            with self.subTest(document=truth["document_id"]):
                record, by_id = truth["semantic_record"], fields(truth)
                self.assertEqual(truth["semantic_version"], VERSION)
                born = ddmmyyyy(by_id["2"]["value"])
                mother_born = ddmmyyyy(by_id["18"]["value"])
                self.assertEqual(born.isoformat(), record["birth.date"])
                self.assertEqual(int(by_id["19"]["value"]), completed_years(mother_born, born))
                self.assertEqual(int(by_id["19"]["value"]), record["mother.age"])
                self.assertGreaterEqual(ddmmyyyy(by_id["50"]["value"]), born)
                filled = date.fromisoformat(record["filling.date"])
                self.assertEqual(by_id["42"]["value"], filled.strftime("%d%m%y"))
                self.assertTrue(0 <= (filled - born).days <= 60)
                occurrence = profile(city=by_id["12"]["value"], uf=by_id["13"]["value"], ibge=by_id["12_codigo"]["value"],
                                     cep=by_id["10"]["value"])
                self.assertIsNotNone(occurrence, "local da ocorrência incoerente")
                self.assertEqual(by_id["11"]["value"], occurrence["neighborhood"])
                residence = profile(city=by_id["26"]["value"], uf=by_id["27"]["value"], cep=by_id["24"]["value"])
                self.assertIsNotNone(residence, "residência incoerente")
                self.assertIsNotNone(profile(city=by_id["51"]["value"], uf=by_id["52"]["value"]))
                counts = {name: int(by_id[f"30_{name}"]["value"]) for name in (
                    "gestacoes_anteriores", "partos_vaginais", "cesareas", "nascidos_vivos", "perdas_fetais_abortos")}
                self.assertEqual(counts["partos_vaginais"] + counts["cesareas"], counts["gestacoes_anteriores"] - counts["perdas_fetais_abortos"])
                self.assertLessEqual(counts["nascidos_vivos"], counts["partos_vaginais"] + counts["cesareas"])
                self.assertRegex(by_id["9_numero"]["value"], r"^\d{3}$")
                self.assertEqual(int(by_id["9_numero"]["value"]), int(record["birth.place.number"]))
                self.assertEqual(by_id["15_cpf"]["value"], record["mother.cpf"])
                checked = [element for element in by_id["3"]["elements"] if element["checked"]]
                self.assertEqual(len(checked), 1)
                self.assertEqual(by_id["3"]["semantic"]["value"], record["newborn.sex"])
                self.assertEqual(checked[0]["label"], {"M": "Masculino", "F": "Feminino"}[record["newborn.sex"]])
                self.assertEqual(by_id["14"]["semantic"], {"key": "mother.name", "value": record["mother.name"], "source": "inferred"})
                self.assertEqual(by_id["14"]["value"], record["mother.name"])
                self.assertNotIn("semantic", by_id["a"])
                surnames = set(record["newborn.name"].split()[1:]) & set(record["mother.name"].split()[1:])
                self.assertTrue(surnames)
                if record["guardian.name"] is None:
                    self.assertTrue(by_id["28"]["blank"] and by_id["29"]["blank"])
                    self.assertEqual((by_id["28"]["value"], by_id["28"]["semantic"]["value"]), ("", None))
                else:
                    self.assertEqual(by_id["28"]["value"], record["guardian.name"])
                    self.assertNotEqual(record["guardian.name"], record["mother.name"])

    def test_certidao_values_come_from_one_coherent_record(self):
        for truth in self.certidao:
            with self.subTest(document=truth["document_id"]):
                record, by_id = truth["semantic_record"], fields(truth)
                born = date(int(by_id["ano_nascimento"]["value"]), int(by_id["mes_nascimento"]["value"]), int(by_id["dia_nascimento"]["value"]))
                self.assertEqual(born.isoformat(), record["birth.date"])
                self.assertEqual(by_id["data_nascimento_por_extenso"]["value"], format_date(born, "EXTENSO_PALAVRAS"))
                self.assertEqual(by_id["dia_nascimento"]["semantic"]["part"], "day")
                self.assertIn(record["mother.name"], by_id["filiacao"]["value"])
                self.assertEqual(by_id["filiacao"]["value"], f'{record["mother.name"]} e {record["father.name"]}')
                for field_id in ("cpf", "matricula", "numero_dnv"):
                    self.assertIn(by_id[field_id]["value"], self.cert_lists[field_id])
                    self.assertNotIn("semantic", by_id[field_id])
                registered = date.fromisoformat(record["registration.date"])
                self.assertEqual(by_id["gemeos"]["value"], "SIM" if record["birth.twins"] else "NÃO")
                self.assertEqual(by_id["nome_matricula_gemeos"].get("blank", False), not record["birth.twins"])
                self.assertGreaterEqual(registered, born)
                self.assertEqual(by_id["data_registro_por_extenso"]["value"], format_date(registered, "EXTENSO_PALAVRAS"))
                place = by_id["local_municipio_nascimento_uf"]["value"]
                self.assertEqual(place, f'{record["birth.establishment.name"]}, {record["birth.place.municipality"]} - {record["birth.place.uf"]}')
                self.assertIsNotNone(profile(city=record["birth.place.municipality"], uf=record["birth.place.uf"]))
                self.assertIn(by_id["naturalidade"]["value"], {record["birth.place.municipality"], record["mother.residence.municipality"]})
                self.assertEqual(by_id["sexo"]["value"], {"M": "MASCULINO", "F": "FEMININO"}[record["newborn.sex"]])

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = generate(CERT_TEMPLATE, CERT_MAPPING, root / "a", 3, seed=5)
            second = generate(CERT_TEMPLATE, CERT_MAPPING, root / "b", 3, seed=5)
            other = generate(CERT_TEMPLATE, CERT_MAPPING, root / "c", 3, seed=6)
            ids = {"1", "2", "3", "9_numero", "12", "13", "14", "18", "19", "30_cesareas", "42", "50"}
            dnv = [generate(DNV_TEMPLATE, DNV_MAPPING, root / name, 2, seed=8, ids=ids) for name in ("d", "e")]
        self.assertEqual(json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False))
        self.assertEqual(json.dumps(dnv[0], ensure_ascii=False), json.dumps(dnv[1], ensure_ascii=False))
        self.assertNotEqual([truth["semantic_record"] for truth in first], [truth["semantic_record"] for truth in other])

    def test_generation_does_not_depend_on_hash_seed(self):
        script = ("import json, sys, tempfile; from pathlib import Path; from src.synthetic_dataset import generate_dataset\n"
                  "mapping = Path(sys.argv[2]); ids = {str(f['field_id']) for f in json.loads(mapping.read_text())['fields']}\n"
                  "with tempfile.TemporaryDirectory() as d:\n"
                  "    docs = generate_dataset(Path(sys.argv[1]), mapping, ids, Path(d) / 'o', 2, seed=13, handwriting=False, text_handwriting=False)\n"
                  "    print(''.join(doc.json_path.read_text() for doc in docs))\n")
        outputs = []
        for hash_seed in ("1", "2"):
            result = subprocess.run([sys.executable, "-c", script, str(CERT_TEMPLATE), str(CERT_MAPPING)], cwd=ROOT, capture_output=True,
                                    text=True, env={**os.environ, "PYTHONHASHSEED": hash_seed}, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_impossible_constraints_fail_quickly_with_message(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blank_page(root / "template.pdf")
            write_mapping(root / "mapping.json", "dnv_v1", [
                {"field_id": "idade", "description": "Idade (da parturiente)", "data_type": "integer", "bbox": [10, 10, 200, 40],
                 "validation": {"allowed_values": ["70", "75"]}}])
            started = clock.perf_counter()
            with self.assertRaises(ValueError) as caught:
                generate_dataset(root / "template.pdf", root / "mapping.json", {"idade"}, root / "out", 1, seed=1)
        self.assertLess(clock.perf_counter() - started, 10)
        self.assertIn("mother.age", str(caught.exception))
        self.assertIn("Idade (da parturiente)", str(caught.exception))
        self.assertIn("nenhum valor coerente", str(caught.exception))

    def test_explicit_bindings_and_unbound_mappings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            blank_page(root / "template.pdf")
            boxes = [{"element_id": f"n{index}", "index": index, "bbox": [10 + 30 * index, 150, 35 + 30 * index, 180]} for index in range(2)]
            write_mapping(root / "explicit.json", "modelo_web", [
                {"field_id": "quem", "label": "Campo livre", "data_type": "text", "bbox": [10, 10, 590, 40], "semantic": "mother.name"},
                {"field_id": "ano", "label": "Outro", "data_type": "text", "bbox": [10, 50, 200, 80], "semantic": {"key": "birth.date", "part": "year"}},
                {"field_id": "pais", "label": "Pais", "data_type": "text", "bbox": [10, 90, 590, 120], "semantic": "parents.names"},
                {"field_id": "numero", "label": "Número", "data_type": "digit_sequence", "bbox": [10, 150, 80, 180], "elements": boxes,
                 "semantic": "mother.residence.number"}])
            for truth in generate(root / "template.pdf", root / "explicit.json", root / "explicit", 4, seed=3):
                record, by_id = truth["semantic_record"], fields(truth)
                self.assertEqual(by_id["quem"]["value"], record["mother.name"])
                self.assertEqual(by_id["ano"]["value"], record["birth.date"][:4])
                self.assertEqual(by_id["ano"]["semantic"], {"key": "birth.date", "value": record["birth.date"], "source": "explicit", "part": "year"})
                self.assertTrue(by_id["pais"]["value"].startswith(record["mother.name"] + " e "))
                self.assertRegex(by_id["numero"]["value"], r"^\d{2}$")
                self.assertEqual(int(by_id["numero"]["value"]), int(record["mother.residence.number"]))
            write_mapping(root / "plain.json", "tipos", [
                {"field_id": "peso", "label": "Peso", "data_type": "digit_sequence", "bbox": [10, 150, 80, 180], "elements": boxes},
                {"field_id": "obs", "label": "Observação", "data_type": "text", "bbox": [10, 10, 590, 40]}])
            for truth in generate(root / "template.pdf", root / "plain.json", root / "plain", 2, seed=3):
                self.assertFalse(any("semantic" in field for field in truth["fields"]))
                self.assertRegex(fields(truth)["peso"]["value"], r"^\d{2}$")
            legacy = generate(DNV_TEMPLATE, DNV_LEGACY, root / "legacy", 2, seed=4)
        for truth in legacy:
            record, by_id = truth["semantic_record"], fields(truth)
            self.assertEqual(by_id["1"]["value"], record["newborn.name"])
            self.assertEqual(by_id["25"]["value"], record["mother.residence.street"])
            self.assertEqual(int(by_id["n_cesareas"]["value"]), record["mother.obstetric.cesareans"])
            self.assertEqual(len(truth["fields"]), len(all_ids(DNV_LEGACY)))

    def test_invalid_explicit_binding_is_rejected(self):
        definition = {"template_id": "modelo_web", "dpi": 300, "page_size_px": [1000, 1000], "fields": [
            {"field_id": "a", "label": "A", "bbox": [10, 10, 100, 50], "semantic": "mother.unknown"}]}
        with self.assertRaisesRegex(ServiceError, "Chave semântica desconhecida"):
            validate_mapping(definition)
        self.assertTrue(any("Chave semântica desconhecida" in issue for issue in mapping_issues(definition)))
        valid = {**definition, "fields": [{**definition["fields"][0], "semantic": {"key": "birth.date", "part": "day"}}]}
        self.assertEqual(validate_mapping(valid)["fields"][0]["semantic"], {"key": "birth.date", "part": "day"})
        self.assertEqual(mapping_issues(valid), [])


def definition(template_id: str, entries: list[dict]) -> dict:
    return {"template_id": template_id, "dpi": 72, "page_size_px": [600, 400], "fields": entries}


def box(index: int) -> list[int]:
    return [10, 10 + 25 * index, 590, 30 + 25 * index]


class SemanticPlanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        blank_page(self.root / "template.pdf")

    def run_mapping(self, template_id: str, entries: list[dict], count: int = 5, seed: int = 2) -> list[dict]:
        write_mapping(self.root / "mapping.json", template_id, entries)
        ids = {entry["field_id"] for entry in entries}
        return generate(self.root / "template.pdf", self.root / "mapping.json", self.root / f"out{seed}", count, seed, ids)

    def test_narrow_ranges_steer_the_record(self):
        entries = [{"field_id": "idade", "description": "Idade (da parturiente)", "data_type": "integer", "bbox": box(0),
                    "validation": {"min_value": "20", "max_value": 21}},
                   {"field_id": "dia", "label": "Dia do nascimento", "data_type": "integer", "bbox": box(1),
                    "validation": {"min_value": 1, "max_value": 3}}]
        for truth in self.run_mapping("certidao_nascimento", entries[1:], 6):
            self.assertIn(int(fields(truth)["dia"]["value"]), {1, 2, 3})
        for truth in self.run_mapping("dnv_v1", entries[:1], 6):
            self.assertIn(truth["semantic_record"]["mother.age"], {20, 21})
            self.assertIn(fields(truth)["idade"]["value"], {"20", "21"})

    def test_infeasible_ranges_are_rejected_before_submission(self):
        entries = [{"field_id": "idade", "description": "Idade (da parturiente)", "data_type": "integer", "bbox": box(0),
                    "validation": {"min_value": 70, "max_value": 80}}]
        mapping = definition("dnv_v1", entries)
        issues = mapping_issues(mapping)
        self.assertTrue(any("mother.age" in issue and "intervalo 70–80" in issue for issue in issues), issues)
        catalog = Catalog(Settings(storage=self.root / "web", min_free_bytes=0))
        options = GenerationOptions(document_type="dnv", mapping_id="custom", fields=["idade"])
        with self.assertRaises(ServiceError) as caught:
            catalog.validate(options, None, mapping)
        self.assertEqual(caught.exception.code, "invalid_mapping")
        self.assertIn("nenhum valor coerente", caught.exception.message)
        with self.assertRaises(ValueError):
            field_texts({**entries[0]}, __import__("random").Random(1), {"document": "dnv_v1"})

    def test_format_rules_keep_the_legacy_generator(self):
        entries = [{"field_id": "dnv", "label": "Número da DNV", "data_type": "dnv_number", "bbox": box(0),
                    "validation": {"format": "digitos", "prefix": "35"}},
                   {"field_id": "mae", "label": "Nome da mãe", "data_type": "text", "bbox": box(1), "validation": {"mask": "AA-####"}},
                   {"field_id": "pai", "label": "Nome do pai", "data_type": "text", "bbox": box(2), "validation": {"max_length": 5}},
                   {"field_id": "rn", "label": "Nome do recém-nascido", "data_type": "text", "bbox": box(3), "validation": {"max_length": 60}}]
        for truth in self.run_mapping("certidao_nascimento", entries):
            by_id = fields(truth)
            self.assertRegex(by_id["dnv"]["value"], r"^35\d{9}$")
            self.assertRegex(by_id["mae"]["value"], r"^[A-Z]{2}-\d{4}$")
            self.assertLessEqual(len(by_id["pai"]["value"]), 5)
            self.assertFalse({"semantic"} & (set(by_id["dnv"]) | set(by_id["mae"]) | set(by_id["pai"])))
            self.assertEqual(by_id["rn"]["value"], truth["semantic_record"]["newborn.name"])
        self.assertEqual(mapping_issues(definition("certidao_nascimento", entries)), [])

    def test_explicit_incompatible_binding_is_reported(self):
        boxes = [{"element_id": f"b{index}", "index": index, "bbox": [10 + 20 * index, 10, 28 + 20 * index, 30]} for index in range(2)]
        mapping = definition("modelo_web", [{"field_id": "x", "label": "Caixas", "data_type": "digit_sequence", "bbox": box(0),
                                             "elements": boxes, "semantic": "mother.name"}])
        issues = mapping_issues(mapping)
        self.assertTrue(any("Caixas" in issue and "não cabe no formato" in issue for issue in issues), issues)
        self.assertEqual(validate_mapping({**mapping, "page_size_px": [600, 400]})["fields"][0]["semantic"], "mother.name")
        with self.assertRaisesRegex(ValueError, "não cabe no formato"):
            self.run_mapping("modelo_web", mapping["fields"], 1)

    def test_semantic_keys_share_bound_values(self):
        entries = [{"field_id": "nasc", "label": "Nascimento", "data_type": "date", "bbox": box(0), "semantic": "birth.date",
                    "validation": {"format": "DD/MM/AAAA", "semantic_key": "nasc"}},
                   {"field_id": "outra", "label": "Outra data", "data_type": "date", "bbox": box(1),
                    "validation": {"format": "DD de mês de AAAA", "semantic_key": "nasc"}},
                   {"field_id": "sexo", "label": "Sexo", "data_type": "sex", "bbox": box(2), "semantic": "newborn.sex",
                    "validation": {"format": "abreviado"}},
                   {"field_id": "dnv", "label": "DNV", "data_type": "dnv_number", "bbox": box(3), "validation": {"format": "digitos"}}]
        for truth in self.run_mapping("certidao_nascimento", entries):
            by_id, record = fields(truth), truth["semantic_record"]
            self.assertNotIn("semantic", by_id["outra"])
            self.assertEqual(by_id["outra"]["date"], record["birth.date"])
            self.assertEqual(by_id["sexo"]["value"], record["newborn.sex"])
            self.assertEqual(by_id["dnv"]["value"], record["registration.dnv_number"])

    def test_death_certificate_context_is_isolated(self):
        entries = [{"field_id": "nome", "label": "Nome do falecido", "bbox": box(0)},
                   {"field_id": "obito", "label": "Data do óbito", "data_type": "date", "bbox": box(1)},
                   {"field_id": "registro", "label": "Data do registro", "data_type": "date", "bloco": "VIII", "bbox": box(2)},
                   {"field_id": "cartorio", "label": "Cartório", "bloco": "VIII", "bbox": box(3)},
                   {"field_id": "municipio", "label": "Município", "bloco": "VIII", "bbox": box(4)},
                   {"field_id": "uf", "label": "UF", "data_type": "uf", "bloco": "VIII", "bbox": box(5)},
                   {"field_id": "sexo", "label": "Sexo", "data_type": "sex", "bbox": box(6)},
                   {"field_id": "endereco", "label": "Endereço do cartório", "bbox": box(7)}]
        for truth in self.run_mapping("declaracao_obito", entries, 8):
            by_id, record = fields(truth), truth["semantic_record"]
            self.assertFalse([key for key in record if key.split(".")[0] in {"newborn", "mother", "filling", "registration", "birth"}], record)
            self.assertEqual(by_id["registro"]["semantic"]["key"], "death_registration.date")
            self.assertTrue(record["death.date"] <= record["death_registration.date"] <= "2025-12-31")
            self.assertIsNotNone(profile(city=by_id["municipio"]["value"], uf=by_id["uf"]["value"]))
            self.assertEqual(by_id["sexo"]["sex"], record["deceased.sex"])
            self.assertEqual(by_id["endereco"]["value"], profile(city=record["death_registration.office.municipality"])["street"])

    def test_editor_detection_follows_the_plan(self):
        mapping = definition("certidao_nascimento", [
            {"field_id": "cpf", "label": "CPF", "data_type": "text", "bbox": box(0), "validation": {"allowed_values": ["123.456.789-09"]}},
            {"field_id": "dia", "label": "Dia do nascimento", "data_type": "date", "bbox": box(1)}])
        self.assertEqual(semantic_inferred(mapping, "certidao_nascimento"), {"cpf": None, "dia": "birth.date"})


if __name__ == "__main__":
    unittest.main()
