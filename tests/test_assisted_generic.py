import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fitz

from src.mapping_assist.providers import OpenAICompatibleAssistant
from src.semantic.assisted import (AssistedGenerationError, AssistedRecords, ai_coverage, build_generic_slots,
                                   validate_generic_value)
from src.synthetic_dataset import generate_dataset
from src.web.services import ReplayRecords, accepted_records
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parents[1]

try:
    import httpx2  # noqa: F401
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

PROCURACAO_FIELDS = [
    {"field_id": "nome", "label": "Nome do outorgante", "description": "Nome completo do outorgante",
     "data_type": "person_name", "bbox": [100, 100, 2000, 200]},
    {"field_id": "estado_civil", "label": "Estado civil", "data_type": "enum",
     "validation": {"allowed_values": ["SOLTEIRO(A)", "CASADO(A)"]}, "bbox": [100, 250, 900, 330]},
    {"field_id": "idade", "label": "Idade", "data_type": "integer", "validation": {"min_value": 18, "max_value": 90},
     "bbox": [100, 400, 400, 480]},
    {"field_id": "uf", "label": "UF", "data_type": "text_sequence", "bbox": [100, 550, 300, 630],
     "elements": [{"element_id": "uf_0", "index": 0, "bbox": [100, 550, 200, 630]},
                  {"element_id": "uf_1", "index": 1, "bbox": [200, 550, 300, 630]}]},
    {"field_id": "sexo", "label": "Sexo", "data_type": "checkbox", "selection_mode": "single",
     "bbox": [100, 700, 900, 800],
     "elements": [{"element_id": "sexo_m", "label": "Masculino", "bbox": [100, 700, 300, 800]},
                  {"element_id": "sexo_f", "label": "Feminino", "bbox": [500, 700, 700, 800]}]},
    {"field_id": "cpf", "label": "CPF", "data_type": "document", "validation": {"document_kind": "cpf"},
     "bbox": [100, 850, 900, 930]},
    {"field_id": "matricula", "label": "Matrícula", "data_type": "text", "validation": {"mask": "######-#"},
     "bbox": [100, 950, 900, 1030]},
]
CNJ_LIKE_FIELDS = [
    {"field_id": "nome", "label": "Nome", "data_type": "person_name", "bbox": [100, 100, 2000, 200]},
    {"field_id": "cpf", "label": "Número do CPF", "data_type": "document",
     "validation": {"pattern": r"^[0-9]{3}\.[0-9]{3}\.[0-9]{3}-[0-9]{2}$", "document_kind": "cpf"},
     "bbox": [100, 250, 900, 330]},
    {"field_id": "matricula", "label": "Matrícula", "data_type": "text",
     "validation": {"mask": "######.##.##.####.#.#####.###.#######-##"}, "bbox": [100, 400, 900, 480]},
    {"field_id": "cns_oficio", "label": "CNS", "data_type": "digit_sequence",
     "elements": [{"element_id": f"cns_{index}", "index": index, "bbox": [100 + index * 10, 550, 110 + index * 10, 630]}
                  for index in range(6)], "bbox": [100, 550, 160, 630]},
    {"field_id": "oficial_nome", "label": "Nome do Oficial", "data_type": "person_name", "bbox": [100, 700, 900, 780]},
]


class FakeAssistant:
    provider = "openai"
    model = "fake"

    def __init__(self, responder=None):
        self.responder = responder or self._default
        self.calls: list[dict] = []

    def _default(self, payload):
        values = {}
        for campo in payload["campos"]:
            key = campo["chave"]
            if "opcoes" in campo:
                values[key] = campo["opcoes"][0]
            elif "intervalo" in campo:
                values[key] = sum(campo["intervalo"]) // 2
            elif campo.get("tipo") == "data":
                values[key] = "15/06/1990"
            elif campo.get("tipo") == "sequência de caracteres":
                values[key] = "SC"[:campo.get("max_caracteres") or 2].ljust(campo.get("max_caracteres") or 2, "X")
            else:
                values[key] = "Maria Clara Souza"
        return values

    def complete_json(self, system, payload, *, temperature, seed):
        self.calls.append(copy.deepcopy(payload))
        values = self.responder(payload)
        return {"valores": values}, {"model": "fake", "seconds": 0.01, "prompt_tokens": 1, "completion_tokens": 1}


def _template(root: Path) -> Path:
    template = root / "template.pdf"
    document = fitz.open()
    document.new_page(width=1000, height=1200)
    document.save(template)
    document.close()
    return template


def _mapping(root: Path, fields: list[dict], template_id: str = "procuracao_teste", dpi: int = 300) -> Path:
    mapping = root / "mapping.json"
    mapping.write_text(json.dumps({"template_id": template_id, "dpi": dpi, "page_size_px": [1000, 1200], "fields": fields}),
                       encoding="utf-8")
    return mapping


class GenericSlotTests(unittest.TestCase):
    def test_excludes_identifiers_masks_and_numeric_patterns(self):
        slots = {slot.field_id: slot for slot in build_generic_slots(PROCURACAO_FIELDS, dpi=300)}
        self.assertEqual(set(slots), {"nome", "estado_civil", "idade", "uf", "sexo"})
        self.assertNotIn("cpf", slots)
        self.assertNotIn("matricula", slots)

    def test_excludes_cnj_identifiers_and_digit_sequences(self):
        slots = {slot.field_id: slot for slot in build_generic_slots(CNJ_LIKE_FIELDS, dpi=300)}
        self.assertEqual(set(slots), {"nome", "oficial_nome"})
        self.assertNotIn("cpf", slots)
        self.assertNotIn("matricula", slots)
        self.assertNotIn("cns_oficio", slots)

    def test_opcoes_from_allowed_values_and_checkbox_elements(self):
        slots = {slot.field_id: slot for slot in build_generic_slots(PROCURACAO_FIELDS, dpi=300)}
        self.assertEqual(slots["estado_civil"].choices, ("SOLTEIRO(A)", "CASADO(A)"))
        self.assertEqual(slots["sexo"].choices, ("Masculino", "Feminino"))

    def test_intervalo_from_validation(self):
        slots = {slot.field_id: slot for slot in build_generic_slots(PROCURACAO_FIELDS, dpi=300)}
        self.assertEqual((slots["idade"].low, slots["idade"].high), (18, 90))

    def test_max_caracteres_combines_validation_and_bbox_capacity(self):
        narrow = {"field_id": "narrow", "label": "Campo estreito", "data_type": "text", "bbox": [0, 0, 90, 22],
                  "validation": {"max_length": 500}}
        slot = build_generic_slots([narrow], dpi=300)[0]
        self.assertLess(slot.max_length, 500)
        self.assertGreaterEqual(slot.max_length, 1)

    def test_bbox_capacity_follows_the_line_height(self):
        from src.semantic.assisted import bbox_capacity
        self.assertEqual(bbox_capacity([0, 0, 1000, 35], 300), 51)
        self.assertEqual(bbox_capacity([0, 0, 500, 17.5], 150), 51)
        self.assertEqual(bbox_capacity([0, 0, 2000, 250], 300), 86 * 5)

    def test_text_sequence_max_caracteres_equals_element_count(self):
        slots = {slot.field_id: slot for slot in build_generic_slots(PROCURACAO_FIELDS, dpi=300)}
        self.assertEqual(slots["uf"].element_count, 2)
        self.assertEqual(slots["uf"].max_length, 2)

    def test_ai_coverage_finds_registry_and_generic_ids(self):
        registry_ids, generic_ids = ai_coverage(PROCURACAO_FIELDS, "procuracao_teste", 300)
        self.assertEqual(registry_ids, set())
        self.assertEqual(generic_ids, {"nome", "estado_civil", "idade", "uf", "sexo"})


class ValidateGenericValueTests(unittest.TestCase):
    def test_date_parses_ddmmyyyy_and_rejects_other_formats(self):
        from src.semantic.assisted import GenericSlot
        slot = GenericSlot("data", "data", "date")
        value, error = validate_generic_value(slot, "15/06/1990")
        self.assertIsNone(error)
        self.assertEqual(value.isoformat(), "1990-06-15")
        self.assertIsNotNone(validate_generic_value(slot, "1990-06-15")[1])

    def test_integer_checks_bounds(self):
        from src.semantic.assisted import GenericSlot
        slot = GenericSlot("idade", "idade", "integer", low=18, high=90)
        self.assertEqual(validate_generic_value(slot, 40), (40, None))
        self.assertIsNotNone(validate_generic_value(slot, 5)[1])
        self.assertIsNotNone(validate_generic_value(slot, 999)[1])

    def test_opcoes_match_case_insensitively(self):
        from src.semantic.assisted import GenericSlot
        slot = GenericSlot("sexo", "sexo", "checkbox", choices=("Masculino", "Feminino"))
        self.assertEqual(validate_generic_value(slot, "masculino"), ("Masculino", None))


class AssistedRecordsGenericTests(unittest.TestCase):
    def test_generic_fields_rendered_including_checkbox_and_composite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _template(root)
            mapping = _mapping(root, PROCURACAO_FIELDS)
            assistant = FakeAssistant()
            factory = AssistedRecords(assistant, fields=PROCURACAO_FIELDS, dpi=300,
                                      document_meta={"nome": "Procuração Teste", "orgao": "MRE", "descricao": "Modelo de teste"},
                                      rules_text="Regra um.\n\nRegra dois.", max_fields_per_request=2)
            generated = generate_dataset(template, mapping, {field["field_id"] for field in PROCURACAO_FIELDS},
                                         root / "out", 1, seed=42, record_factory=factory)
            truth = json.loads(generated[0].json_path.read_text())
            values = {entry["field_id"]: entry["value"] for entry in truth["fields"]}
            self.assertEqual(values["nome"], "Maria Clara Souza")
            self.assertRegex(values["uf"], r"^[A-Z]{2}$")
            self.assertIn(values["sexo"], {"sexo_m", "sexo_f"})
            self.assertRegex(values["cpf"], r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
            self.assertRegex(values["matricula"], r"^\d{6}-\d$")
            provenance = truth["semantic_provenance"]
            self.assertEqual(provenance["mode"], "ai")
            self.assertEqual(set(provenance["ai_fields"]), {"nome", "estado_civil", "idade", "uf", "sexo"})
            self.assertEqual(provenance["fallback_fields"], {})
            self.assertGreater(len(assistant.calls), 1)
            second_batch = assistant.calls[1]
            self.assertIn("ja_definidos", second_batch)
            self.assertTrue(second_batch["ja_definidos"])
            self.assertIn("regras", second_batch["documento"])
            self.assertEqual(second_batch["documento"]["nome"], "Procuração Teste")

    def test_semantic_values_are_sent_as_ja_definidos(self):
        fields = [{"field_id": "nome", "label": "Nome", "data_type": "person_name", "semantic": "newborn.name",
                   "bbox": [100, 100, 2000, 200]},
                  {"field_id": "dia", "label": "Dia", "data_type": "integer", "semantic": {"key": "birth.date", "part": "day"},
                   "bbox": [100, 250, 300, 330]},
                  {"field_id": "mes", "label": "Mês", "data_type": "integer", "semantic": {"key": "birth.date", "part": "month"},
                   "bbox": [400, 250, 600, 330]},
                  {"field_id": "observacao", "label": "Observação", "data_type": "text", "semantic": False,
                   "bbox": [100, 400, 2000, 480]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant = FakeAssistant()
            factory = AssistedRecords(assistant, fields=fields, dpi=300)
            generate_dataset(_template(root), _mapping(root, fields, "certidao_nascimento"), {f["field_id"] for f in fields},
                             root / "out", 1, seed=3, record_factory=factory)
            generic_call = next(call for call in assistant.calls
                                if [campo["chave"] for campo in call["campos"]] == ["observacao"])
            known = generic_call["ja_definidos"]
            self.assertTrue({"nome", "dia"} <= set(known))
            self.assertRegex(known["dia"], r"^\d{2}/\d{2}/\d{4}$")

    def test_invalid_value_triggers_correction_then_deterministic_fallback(self):
        def always_invalid_age(payload):
            values = {}
            for campo in payload["campos"]:
                key = campo["chave"]
                if key == "idade":
                    values[key] = 999
                elif "opcoes" in campo:
                    values[key] = campo["opcoes"][0]
                else:
                    values[key] = "Maria Clara Souza"
            return values

        fields = [field for field in PROCURACAO_FIELDS if field["field_id"] in {"nome", "idade"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _template(root)
            mapping = _mapping(root, fields)
            assistant = FakeAssistant(always_invalid_age)
            factory = AssistedRecords(assistant, fields=fields, dpi=300, attempts=2)
            generated = generate_dataset(template, mapping, {"nome", "idade"}, root / "out", 1, seed=1, record_factory=factory)
            truth = json.loads(generated[0].json_path.read_text())
            values = {entry["field_id"]: entry["value"] for entry in truth["fields"]}
            self.assertTrue(values["idade"].isdigit())
            self.assertTrue(18 <= int(values["idade"]) <= 90)
            provenance = truth["semantic_provenance"]
            self.assertEqual(provenance["ai_fields"], ["nome"])
            self.assertIn("idade", provenance["fallback_fields"])
            self.assertNotIn("idade", truth.get("ai_values", {}))

    def test_provider_unreachable_for_every_attempt_fails_pure_generic_document(self):
        class DownAssistant:
            provider = "openai"
            model = "fake"

            def complete_json(self, system, payload, *, temperature, seed):
                raise RuntimeError("conexão recusada")

        fields = [field for field in PROCURACAO_FIELDS if field["field_id"] == "nome"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _template(root)
            mapping = _mapping(root, fields)
            factory = AssistedRecords(DownAssistant(), fields=fields, dpi=300, attempts=2)
            with self.assertRaises(AssistedGenerationError):
                generate_dataset(template, mapping, {"nome"}, root / "out", 1, seed=1, record_factory=factory)

    def test_no_eligible_fields_raises(self):
        fields = [field for field in PROCURACAO_FIELDS if field["field_id"] in {"cpf", "matricula"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _template(root)
            mapping = _mapping(root, fields)
            factory = AssistedRecords(FakeAssistant(), fields=fields, dpi=300)
            with self.assertRaises(AssistedGenerationError):
                generate_dataset(template, mapping, {"cpf", "matricula"}, root / "out", 1, seed=1, record_factory=factory)


class ReplayWithoutProviderTests(unittest.TestCase):
    def test_replay_reproduces_generic_values_without_calling_the_assistant(self):
        fields = PROCURACAO_FIELDS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = _template(root)
            mapping = _mapping(root, fields)
            assistant = FakeAssistant()
            factory = AssistedRecords(assistant, fields=fields, dpi=300)
            selected = {field["field_id"] for field in fields}
            first = generate_dataset(template, mapping, selected, root / "out1", 1, seed=7, record_factory=factory)
            replay_payload = accepted_records(root / "out1")
            replay_payload["source_job"] = "job-1"
            calls_before_replay = len(assistant.calls)
            second = generate_dataset(template, mapping, selected, root / "out2", 1, seed=7,
                                      record_factory=ReplayRecords(replay_payload))
            self.assertEqual(len(assistant.calls), calls_before_replay)
            first_truth = json.loads(first[0].json_path.read_text())
            second_truth = json.loads(second[0].json_path.read_text())
            self.assertEqual([entry["value"] for entry in first_truth["fields"]], [entry["value"] for entry in second_truth["fields"]])
            self.assertEqual(second_truth["semantic_provenance"]["mode"], "ai_replay")


class StructuredOutputTests(unittest.TestCase):
    def test_schema_request_falls_back_to_json_object_after_bad_request(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body["response_format"]["type"])
                if body["response_format"]["type"] == "json_schema":
                    self.send_response(400)
                    self.end_headers()
                    return
                payload = json.dumps({"model": "fake", "choices": [{"message": {"content": json.dumps({"valores": {"nome": "Ana Souza"}})}}],
                                      "usage": {"prompt_tokens": 1, "completion_tokens": 1}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        assistant = OpenAICompatibleAssistant(f"http://127.0.0.1:{server.server_address[1]}/v1", "fake")
        payload = {"campos": [{"chave": "nome", "descricao": "nome", "max_caracteres": 40}]}
        answer, _ = assistant.complete_json("sistema", payload, temperature=0.5, seed=1)
        self.assertEqual(answer, {"valores": {"nome": "Ana Souza"}})
        self.assertTrue(assistant.schema_unsupported)
        self.assertEqual(calls, ["json_schema", "json_object"])
        assistant.complete_json("sistema", payload, temperature=0.5, seed=2)
        self.assertEqual(calls, ["json_schema", "json_object", "json_object"])

    def test_schema_built_from_slot_shapes(self):
        from src.semantic.assisted import GenericSlot
        assistant = OpenAICompatibleAssistant("http://127.0.0.1:1/v1", "fake")
        campos = [GenericSlot("sexo", "sexo", "checkbox", choices=("Masculino", "Feminino")).describe(),
                  GenericSlot("idade", "idade", "integer", low=18, high=90).describe(),
                  GenericSlot("obs", "observações", "text", max_length=120).describe()]
        schema = assistant._schema_format(campos)["json_schema"]["schema"]["properties"]["valores"]
        self.assertEqual(schema["properties"]["sexo"], {"type": "string", "enum": ["Masculino", "Feminino"]})
        self.assertEqual(schema["properties"]["idade"], {"type": "integer", "minimum": 18, "maximum": 90})
        self.assertEqual(schema["properties"]["obs"], {"type": "string", "maxLength": 120})
        self.assertEqual(set(schema["required"]), {"sexo", "idade", "obs"})
        self.assertFalse(schema["additionalProperties"])


class SettingsBudgetTests(unittest.TestCase):
    def test_ai_seconds_per_document_used_for_ai_jobs(self):
        settings = Settings(ai_seconds_per_document=50, job_seconds_per_document=5, job_timeout=100)
        self.assertEqual(settings.job_time_budget(10, "ai"), 100 + 10 * 50)
        self.assertEqual(settings.job_time_budget(10, "traditional"), 100 + 10 * 5)
        self.assertEqual(settings.job_time_budget(10), 100 + 10 * 5)


class FakeWebRecordModel(BaseHTTPRequestHandler):
    requests = 0

    def log_message(self, *args):
        pass

    def send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).requests += 1
        if self.path.endswith("/api/tags"):
            return self.send({"models": [{"name": "fake", "digest": "sha256:procuracaoteste0001"}]})
        self.send({"data": [{"id": "fake"}]})

    def do_POST(self):
        type(self).requests += 1
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = json.loads(request["messages"][1]["content"])
        values = FakeAssistant()._default(content)
        self.send({"model": "fake", "choices": [{"message": {"content": json.dumps({"valores": values})}}],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 20}})


@unittest.skipUnless(HTTPX_AVAILABLE, "Instale os extras web e test")
class WebGenericAiEndToEndTests(unittest.TestCase):
    def setUp(self):
        from tests.test_web import ASGITestClient
        from src.web.api import create_app

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        documents = self.root / "documents"
        folder = documents / "procuracao_teste"
        (folder / "mappings").mkdir(parents=True)
        shutil.copy(ROOT / "documents/dnv/template.pdf", folder / "template.pdf")
        (folder / "modelo.json").write_text(json.dumps({"active_mapping": "default", "label": "Procuração de Teste",
                                                        "description": "Modelo de teste sem campos de registro civil."}),
                                            encoding="utf-8")
        definition = {"template_id": "procuracao_teste", "dpi": 300, "page_size_px": [4330, 5717],
                      "fields": [field for field in PROCURACAO_FIELDS if field["field_id"] != "matricula"]}
        (folder / "mappings" / "default.json").write_text(json.dumps(definition), encoding="utf-8")
        (folder / "regras.md").write_text("# Regras\n\nPreencha com dados fictícios plausíveis para o teste.\n", encoding="utf-8")
        FakeWebRecordModel.requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeWebRecordModel)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.settings = Settings(storage=self.root / "web", documents=documents, min_free_bytes=0, neural_enabled=False,
                                 ai_provider="openai", ai_model="fake", ai_base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1")
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        response = self.client.get("/modelos")
        self.cookie = re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)
        self.owner = self.app.state.store.workspace(self.cookie)

    def hx(self, method, path, **kwargs):
        return self.client.request(method, path, headers={"Cookie": f"ocrsd_workspace={self.cookie}", "HX-Request": "true"}, **kwargs)

    def run_worker(self):
        import os

        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()
                                        if value is not None},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_document_without_registry_fields_generates_with_ai_and_replays(self):
        fields = [field["field_id"] for field in PROCURACAO_FIELDS if field["field_id"] != "matricula"]
        page = self.hx("GET", "/geracao?modelo=procuracao_teste")
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        for step, data in ((1, {"document_type": "procuracao_teste"}), (2, {"fields": fields}),
                           (3, {"count": "1", "generation_mode": "ai"}), (4, {"preset": "padrao_fonte_natural"}),
                           (5, {"filter_preset": "none"})):
            response = self.hx("POST", f"/geracao/etapa/{step}", data={"draft_id": draft, **data})
            self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.hx("POST", "/geracao/gerar", data={"draft_id": draft}).status_code, 200)
        job = self.app.state.store.list_jobs(self.owner)[0]
        self.assertEqual(job["options"]["generation_mode"], "ai")
        self.run_worker()
        row = self.app.state.store.get(job["id"])
        self.assertEqual(row["status"], "succeeded", row["error"])
        directory = self.app.state.store.directory(job["id"])
        manifest = json.loads((directory / "dataset.json").read_text())
        self.assertEqual(manifest["generation"]["mode"], "ai")
        truth = json.loads((directory / manifest["documents"][0]["ground_truth"]).read_text())
        self.assertEqual(truth["semantic_provenance"]["mode"], "ai")
        self.assertTrue(truth["semantic_provenance"]["ai_fields"])

        contacted = FakeWebRecordModel.requests
        replayed = self.hx("POST", f'/resultados/{job["id"]}/reproduzir')
        self.assertEqual(replayed.status_code, 200, replayed.text)
        replay_id = replayed.headers["hx-redirect"].rsplit("/", 1)[1]
        self.run_worker()
        self.assertEqual(FakeWebRecordModel.requests, contacted)
        replay_row = self.app.state.store.get(replay_id)
        self.assertEqual(replay_row["status"], "succeeded", replay_row["error"])
        replay_directory = self.app.state.store.directory(replay_id)
        replay_manifest = json.loads((replay_directory / "dataset.json").read_text())
        replay_truth = json.loads((replay_directory / replay_manifest["documents"][0]["ground_truth"]).read_text())
        self.assertEqual([field["value"] for field in replay_truth["fields"]], [field["value"] for field in truth["fields"]])
        self.assertEqual(replay_truth["semantic_provenance"]["mode"], "ai_replay")


if __name__ == "__main__":
    unittest.main()
