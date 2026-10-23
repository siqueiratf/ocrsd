import json
import os
from datetime import date
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import httpx2  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from pydantic import ValidationError

from src.web.api import create_app
from src.web.models import GenerationOptions
from src.web.settings import Settings
from tests.test_web import ASGITestClient

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["1", "3", "14", "19", "38"]
DO_FIELDS = ["2", "5", "8", "9_anos", "10", "14", "18"]
NAMES = {"mother.name": "Maria Clara Souza", "father.name": "Paulo Henrique Souza", "newborn.name": "Ana Beatriz Souza"}
NUMBERS = {"mother.age": 28, "father.age": 31, "pregnancy.weeks": 39}


class FakeRecordModel(BaseHTTPRequestHandler):
    mode = "valid"
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
            return self.send({"models": [{"name": "fake", "digest": "sha256:0123456789abcdef0123"}]})
        self.send({"data": [{"id": "fake"}]})

    def do_POST(self):
        type(self).requests += 1
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        slots = json.loads(request["messages"][1]["content"])["campos"]
        values = {}
        for slot in slots:
            key = slot["chave"]
            if key == "newborn.sex":
                values[key] = "F"
            elif "opcoes" in slot:
                values[key] = slot["opcoes"][1] if key == "deceased.occupation" else slot["opcoes"][0]
            elif "intervalo" in slot:
                values[key] = NUMBERS.get(key, sum(slot["intervalo"]) // 2)
            else:
                values[key] = NAMES.get(key, "Clara Souza")
        if type(self).mode == "partial":
            values = {key: value for index, (key, value) in enumerate(values.items()) if index % 2 == 0}
        if type(self).mode == "contradictory":
            values["deceased.age"] = 2
        content = json.dumps({"valores": values}) if type(self).mode != "invalid" else "não sei responder"
        self.send({"model": "fake", "choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 20}})


class GenerationModeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeRecordModel)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        FakeRecordModel.mode, FakeRecordModel.requests = "valid", 0
        self.start(ai_provider="none")

    def start(self, **changes):
        self.settings = Settings(storage=self.root / "web", min_free_bytes=0, neural_enabled=False, **changes)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        response = self.client.get("/modelos")
        self.cookie = re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)
        self.owner = self.app.state.store.workspace(self.cookie)

    def start_ai(self):
        self.start(ai_provider="openai", ai_model="fake", ai_base_url=f"http://127.0.0.1:{self.server.server_address[1]}/v1")

    def hx(self, method, path, **kwargs):
        return self.client.request(method, path, headers={"Cookie": f"ocrsd_workspace={self.cookie}", "HX-Request": "true"}, **kwargs)

    def wizard(self, generation_mode="traditional", count="2", document_type="dnv", fields=None):
        page = self.hx("GET", f"/geracao?modelo={document_type}")
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        for step, data in ((1, {"document_type": document_type}), (2, {"fields": fields or FIELDS}), (3, {"count": count, "generation_mode": generation_mode}),
                           (4, {"preset": "padrao_fonte_natural"}), (5, {"filter_preset": "none"})):
            response = self.hx("POST", f"/geracao/etapa/{step}", data={"draft_id": draft, **data})
            self.assertEqual(response.status_code, 200, response.text)
        return draft

    def run_worker(self):
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()
                                        if value is not None},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stderr)

    def outputs(self, job_id):
        directory = self.app.state.store.directory(job_id)
        manifest = json.loads((directory / "dataset.json").read_text())
        return manifest, [json.loads((directory / entry["ground_truth"]).read_text()) for entry in manifest["documents"]]

    def test_default_mode_and_old_options_compatibility(self):
        old = {"document_type": "dnv", "mapping_id": "mapeamento_seguindo_norma", "fields": ["1"], "count": 1, "seed": 9}
        self.assertEqual(GenerationOptions.model_validate_json(json.dumps(old)).generation_mode, "traditional")
        with self.assertRaises(ValidationError):
            GenerationOptions.model_validate({**old, "generation_mode": "magic"})
        store = self.app.state.store
        row, _ = store.submit(GenerationOptions.model_validate(old), "teste", self.owner)
        with store.connection() as db:
            db.execute("UPDATE jobs SET options=? WHERE id=?", (json.dumps(old), row["id"]))
        self.assertEqual(store.detail(store.get(row["id"])).options.generation_mode, "traditional")
        details = self.hx("GET", f'/resultados/{row["id"]}/detalhes')
        self.assertIn("Modo de geração</dt><dd>Tradicional", details.text)
        config = store.save_config(self.owner, "Antiga", GenerationOptions.model_validate(old))
        page = self.hx("GET", f'/geracao?configuracao={config["id"]}')
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        self.assertEqual(store.wizard_draft(draft, self.owner)["generation_mode"], "traditional")

    def test_ai_option_disabled_and_rejected_when_provider_is_unavailable(self):
        page = self.hx("GET", "/geracao?modelo=dnv")
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        self.hx("POST", "/geracao/etapa/1", data={"draft_id": draft, "document_type": "dnv"})
        self.hx("POST", "/geracao/etapa/2", data={"draft_id": draft, "fields": FIELDS})
        quantity = self.hx("GET", f"/geracao?draft={draft}&etapa=3")
        self.assertIn('name="generation_mode" value="traditional" checked', quantity.text)
        self.assertRegex(quantity.text, r'name="generation_mode" value="ai"\s+disabled')
        self.assertIn("Assistente de IA desativado", quantity.text)
        self.assertIn("OCRSD_AI_PROVIDER=openai", quantity.text)
        self.assertIn("qwen3:1.7b", quantity.text)
        rejected = self.hx("POST", "/geracao/etapa/3", data={"draft_id": draft, "count": "1", "generation_mode": "ai"})
        self.assertEqual(rejected.status_code, 422)
        self.assertIn("indisponível", rejected.text)
        api = self.client.post("/api/v1/jobs", headers={"X-Workspace-Token": self.cookie},
                               json={"document_type": "dnv", "fields": FIELDS, "count": 1, "generation_mode": "ai"})
        self.assertEqual((api.status_code, api.json()["error"]["code"]), (422, "ai_unavailable"))
        self.assertEqual(self.app.state.store.list_jobs(self.owner), [])

    def test_ai_generation_records_provenance_and_exact_replay_without_inference(self):
        self.start_ai()
        quantity_page = self.hx("GET", "/geracao?modelo=dnv")
        self.assertEqual(quantity_page.status_code, 200)
        draft = self.wizard("ai")
        review = self.hx("GET", f"/geracao?draft={draft}&etapa=6")
        self.assertIn("Assistida por IA", review.text)
        self.assertEqual(self.hx("POST", "/geracao/gerar", data={"draft_id": draft}).status_code, 200)
        job = self.app.state.store.list_jobs(self.owner)[0]
        self.assertEqual(job["options"]["generation_mode"], "ai")
        self.run_worker()
        row = self.app.state.store.get(job["id"])
        self.assertEqual(row["status"], "succeeded", row["error"])
        manifest, truths = self.outputs(job["id"])
        generation = manifest["generation"]
        self.assertEqual((generation["mode"], generation["provider"], generation["model"]), ("ai", "openai", "fake"))
        self.assertEqual(generation["model_digest"], "sha256:0123456789abcdef0123")
        self.assertEqual([item["status"] for item in generation["documents"]], ["accepted", "accepted"])
        for truth in truths:
            self.assertEqual(truth["semantic_provenance"]["mode"], "ai")
            self.assertEqual(truth["semantic_record"]["mother.name"], "Maria Clara Souza")
        details = self.hx("GET", f'/resultados/{job["id"]}/detalhes')
        self.assertIn("Modo de geração</dt><dd>Assistida por IA", details.text)
        self.assertIn("0123456789ab", details.text)
        self.assertIn("2 documento(s) validado(s)", details.text)
        self.assertNotIn("Você cria cenários", details.text)
        contacted = FakeRecordModel.requests
        replayed = self.hx("POST", f'/resultados/{job["id"]}/reproduzir')
        self.assertEqual(replayed.status_code, 200, replayed.text)
        replay_id = replayed.headers["hx-redirect"].rsplit("/", 1)[1]
        self.run_worker()
        self.assertEqual(FakeRecordModel.requests, contacted)
        replay_row = self.app.state.store.get(replay_id)
        self.assertEqual(replay_row["status"], "succeeded", replay_row["error"])
        replay_manifest, replay_truths = self.outputs(replay_id)
        self.assertEqual((replay_manifest["generation"]["mode"], replay_manifest["generation"]["source_job"]), ("ai_replay", job["id"]))
        self.assertEqual([truth["semantic_record"] for truth in replay_truths], [truth["semantic_record"] for truth in truths])
        self.assertEqual([[field["value"] for field in truth["fields"]] for truth in replay_truths],
                         [[field["value"] for field in truth["fields"]] for truth in truths])
        self.assertTrue(all(truth["semantic_provenance"]["mode"] == "ai_replay" for truth in replay_truths))
        replay_details = self.hx("GET", f"/resultados/{replay_id}/detalhes")
        self.assertIn("Reprodução de registros da IA", replay_details.text)
        self.assertIn("fake", replay_details.text)

    def test_invalid_ai_output_fails_job_with_diagnostic(self):
        self.start_ai()
        FakeRecordModel.mode = "invalid"
        draft = self.wizard("ai", count="1")
        self.hx("POST", "/geracao/gerar", data={"draft_id": draft})
        job = self.app.state.store.list_jobs(self.owner)[0]
        self.run_worker()
        row = self.app.state.store.get(job["id"])
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["error"].startswith("A geração assistida por IA falhou: A IA não produziu um registro válido"), row["error"])
        self.assertIn("A geração assistida por IA falhou", self.hx("GET", f'/resultados/{job["id"]}').text)
        self.assertEqual(self.hx("POST", f'/resultados/{job["id"]}/reproduzir').status_code, 422)

    def test_partial_ai_response_records_which_values_were_generated_by_ai(self):
        self.start_ai()
        FakeRecordModel.mode = "partial"
        draft = self.wizard("ai", count="1")
        self.assertEqual(self.hx("POST", "/geracao/gerar", data={"draft_id": draft}).status_code, 200)
        job = self.app.state.store.list_jobs(self.owner)[0]
        self.run_worker()
        self.assertEqual(self.app.state.store.get(job["id"])["status"], "succeeded")
        manifest, truths = self.outputs(job["id"])
        provenance = truths[0]["semantic_provenance"]
        self.assertTrue(provenance["ai_keys"])
        self.assertEqual(provenance["ai_keys"], sorted(provenance["accepted"]))
        self.assertGreater(provenance["validation"]["rejected_values"], 0)
        self.assertNotIn("father.age", provenance["ai_keys"])
        self.assertIn("father.age", truths[0]["semantic_record"])
        self.assertTrue(all(truths[0]["semantic_record"][key] == value for key, value in provenance["accepted"].items()))
        self.assertEqual(manifest["generation"]["documents"][0]["status"], "accepted")

    def test_do_ai_generation_preserves_age_chronology_and_rejects_impossible_occupation(self):
        self.start_ai()
        draft = self.wizard("ai", count="1", document_type="do", fields=DO_FIELDS)
        self.assertEqual(self.hx("POST", "/geracao/gerar", data={"draft_id": draft}).status_code, 200)
        job = self.app.state.store.list_jobs(self.owner)[0]
        self.run_worker()
        row = self.app.state.store.get(job["id"])
        self.assertEqual(row["status"], "succeeded", row["error"])
        _, truths = self.outputs(job["id"])
        record = truths[0]["semantic_record"]
        born, died = date.fromisoformat(record["deceased.birth_date"]), date.fromisoformat(record["death.date"])
        self.assertLessEqual(born, died)
        self.assertEqual(record["deceased.age"], died.year - born.year - ((died.month, died.day) < (born.month, born.day)))
        self.assertEqual(truths[0]["semantic_provenance"]["mode"], "ai")

        FakeRecordModel.mode = "contradictory"
        draft = self.wizard("ai", count="1", document_type="do", fields=DO_FIELDS)
        self.assertEqual(self.hx("POST", "/geracao/gerar", data={"draft_id": draft}).status_code, 200)
        rejected_job = self.app.state.store.list_jobs(self.owner)[0]
        self.run_worker()
        rejected = self.app.state.store.get(rejected_job["id"])
        self.assertEqual(rejected["status"], "failed")
        self.assertIn("A IA não produziu um registro válido", rejected["error"])


class BuiltinMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        documents = Path(temporary.name) / "documents"
        folder = documents / "obito_teste"
        (folder / "mappings").mkdir(parents=True)
        shutil.copy(ROOT / "documents/dnv/template.pdf", folder / "template.pdf")
        (folder / "modelo.json").write_text(json.dumps({"active_mapping": "default", "label": "Declaração de Teste (DT)",
                                                        "description": "Modelo de teste com campo sem gerador."}), encoding="utf-8")
        definition = {"template_id": "obito_teste", "dpi": 300, "page_size_px": [4330, 5717], "fields": [
            {"field_id": "nome", "label": "Nome", "data_type": "person_name", "bbox": [100, 100, 900, 200], "section_label": "I – Identificação"},
            {"field_id": "causa", "label": "Causa", "data_type": "text", "semantic": "custom.death.cause", "bbox": [100, 300, 900, 400],
             "section_label": "II – Condições e causas"}],
            "semantic_definitions": {"custom.death.cause": {"key": "custom.death.cause", "label": "Causa do óbito",
                                                            "description": "Causa declarada", "data_type": "text",
                                                            "generator": {"type": "unavailable"}}}}
        (folder / "mappings" / "default.json").write_text(json.dumps(definition), encoding="utf-8")
        self.settings = Settings(storage=Path(temporary.name) / "web", documents=documents, min_free_bytes=0)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        response = self.client.get("/modelos")
        self.gallery = response.text
        self.cookie = re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)

    def hx(self, method, path, **kwargs):
        return self.client.request(method, path, headers={"Cookie": f"ocrsd_workspace={self.cookie}", "HX-Request": "true"}, **kwargs)

    def test_label_description_protection_and_blocked_fields(self):
        self.assertIn("Declaração de Teste (DT)", self.gallery)
        self.assertIn("Modelo de teste com campo sem gerador.", self.gallery)
        self.assertIn("Padrão", self.gallery)
        self.assertIn("Duplicar e editar", self.gallery)
        self.assertEqual(self.hx("POST", "/modelos/obito_teste/mapeamento", data={"definition": "{}", "revision": "", "layout": ""}).status_code, 403)
        page = self.hx("GET", "/geracao?modelo=obito_teste")
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        self.assertEqual(self.hx("POST", "/geracao/etapa/1", data={"draft_id": draft, "document_type": "obito_teste"}).status_code, 200)
        fields = self.hx("GET", f"/geracao?draft={draft}&etapa=2").text
        self.assertIn("I – Identificação", fields)
        self.assertIn("II – Condições e causas", fields)
        self.assertRegex(fields, r'name="fields" value="causa" disabled')
        self.assertIn("Sem gerador validado", fields)
        self.assertNotIn('data-overlay-field="causa"', fields)
        self.assertIn("0 de 1 campos selecionados", fields)
        self.assertEqual(self.hx("POST", "/geracao/etapa/2", data={"draft_id": draft, "fields": ["nome", "causa"]}).status_code, 422)
        self.assertEqual(self.hx("POST", "/geracao/etapa/2", data={"draft_id": draft, "fields": ["nome"]}).status_code, 200)
        copied = self.hx("POST", "/modelos/obito_teste/copia")
        self.assertEqual(copied.status_code, 200, copied.text)


if __name__ == "__main__":
    unittest.main()
