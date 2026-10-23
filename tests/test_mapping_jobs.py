import base64
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from src.mapping_assist.providers import NullAssistant
from src.web.jobs import JobStore
from src.web.models import GenerationOptions
from src.web.services import BUILTIN_PROTECTED, Catalog, ServiceError, apply_proposal, create_model, mapping_issues, template_page_size
from src.web.settings import Settings
from src.web.worker import process_analysis, run_analysis
from tests.test_mapping_analysis import synthetic_pdf

ROOT = Path(__file__).resolve().parents[1]
CERTIDAO = ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf"


class AnalysisJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(storage=Path(temporary.name), min_free_bytes=0, ai_provider="none")
        self.store = JobStore(self.settings)
        self.catalog = Catalog(self.settings)
        self.owner, _ = self.store.create_workspace()
        self.other, _ = self.store.create_workspace()
        self.model = create_model(self.store, self.settings, self.owner, "Declaração de Nascido Vivo", "", "default", "modelo.pdf",
                                  synthetic_pdf())

    def analyze(self, page: int = 1, model_id: str | None = None) -> dict:
        queued = self.store.submit_analysis(model_id or self.model["id"], self.owner, page)
        process_analysis(self.store, self.store.claim_analysis())
        return self.store.analysis(queued["id"], self.owner)

    def test_lifecycle_snapshot_and_ownership(self):
        queued = self.store.submit_analysis(self.model["id"], self.owner)
        self.assertEqual((queued["status"], queued["progress"], queued["page"], queued["proposal"]), ("queued", 0, 1, None))
        self.assertEqual(self.store.submit_analysis(self.model["id"], self.owner)["id"], queued["id"])
        self.assertEqual(self.store.latest_analysis(self.model["id"], self.owner)["id"], queued["id"])
        self.assertIsNone(self.store.latest_analysis(self.model["id"], self.other))
        for call in (lambda: self.store.analysis(queued["id"], self.other), lambda: self.store.submit_analysis(self.model["id"], self.other)):
            with self.assertRaises(ServiceError) as hidden:
                call()
            self.assertEqual(hidden.exception.status, 404)
        (self.store.model_directory(self.model["id"]) / "template.pdf").write_bytes(b"changed")
        claimed = self.store.claim_analysis()
        self.assertEqual((claimed["id"], claimed["status"], claimed["context"]), (queued["id"], "running", "dnv"))
        self.assertIsNone(self.store.claim_analysis())
        process_analysis(self.store, claimed)
        result = self.store.analysis(queued["id"], self.owner)
        self.assertEqual((result["status"], result["progress"], result["error"]), ("succeeded", 100, None))
        proposal = result["proposal"]
        self.assertEqual(proposal["template_id"], "dnv")
        self.assertEqual(proposal["provenance"]["analysis_id"], queued["id"])
        self.assertEqual((proposal["provenance"]["page"], proposal["provenance"]["reviewed"], proposal["provenance"]["model_version"]),
                         (0, False, 1))
        self.assertEqual(len(proposal["provenance"]["pdf_sha256"]), 64)
        self.assertEqual(result["provenance"]["provider"], "none")
        self.assertEqual(result["provenance"]["analysis_id"], queued["id"])
        self.assertGreater(len(proposal["fields"]), 1)
        self.assertTrue(all(field["suggestion"]["status"] == "suggested" for field in proposal["fields"]))
        self.assertFalse(self.store.analysis_directory(queued["id"]).exists())

    def test_provider_unavailable_still_produces_proposal(self):
        self.store.settings = Settings(**{**self.settings.model_dump(), "ai_provider": "openai", "ai_base_url": "http://127.0.0.1:9/v1"})
        result = self.analyze()
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["proposal"]["provenance"]["provider"], "none")
        self.assertTrue(any("indisponível" in warning for warning in result["proposal"]["provenance"]["warnings"]))
        self.assertGreater(len(result["proposal"]["fields"]), 1)
        definition = self.catalog.definition(self.model["id"], "default", self.owner)
        two_pages = self.store.create_model(self.owner, "Duas páginas", "", "default", synthetic_pdf(pages=2), definition)
        blank = self.analyze(page=2, model_id=two_pages["id"])
        self.assertEqual((blank["status"], blank["proposal"]["fields"], blank["proposal"]["provenance"]["page"]), ("succeeded", [], 1))

    def test_invalid_requests_failure_recovery_and_cleanup(self):
        with self.assertRaises(ServiceError) as invalid:
            self.store.submit_analysis(self.model["id"], self.owner, page=2)
        self.assertEqual(invalid.exception.status, 422)
        queued = self.store.submit_analysis(self.model["id"], self.owner)
        (self.store.analysis_directory(queued["id"]) / "template.pdf").write_bytes(b"%PDF-corrompido")
        run_analysis(self.store, self.store.claim_analysis())
        failed = self.store.analysis(queued["id"], self.owner)
        self.assertEqual((failed["status"], failed["proposal"]), ("failed", None))
        self.assertIn("Não foi possível analisar o PDF", failed["error"])
        running = self.store.submit_analysis(self.model["id"], self.owner)
        with self.assertRaises(ServiceError) as active:
            self.store.delete_model(self.model["id"], self.owner)
        self.assertEqual(active.exception.status, 409)
        self.store.claim_analysis()
        self.store.recover()
        self.assertEqual(self.store.analysis(running["id"], self.owner)["status"], "failed")
        with self.store.connection() as db:
            db.execute("UPDATE analyses SET expires_at=?", (time.time() - 1,))
        self.store.cleanup()
        with self.assertRaises(ServiceError) as expired:
            self.store.analysis(running["id"], self.owner)
        self.assertEqual(expired.exception.status, 410)
        self.store.update_model(self.model["id"], self.owner, self.model["label"], "", "published")
        with self.assertRaises(ServiceError) as published:
            self.store.submit_analysis(self.model["id"], self.owner)
        self.assertEqual(published.exception.status, 409)
        self.store.delete_model(self.model["id"], self.owner)
        self.assertFalse(self.store.analysis_directory(running["id"]).exists())

    def test_apply_proposal_keeps_accepted_and_confirmed_fields(self):
        definition = self.catalog.definition(self.model["id"], "default", self.owner)
        confirmed = {"field_id": "manual", "label": "Campo manual", "data_type": "text", "bbox": [100, 3000, 900, 3080]}
        pending = {**confirmed, "field_id": "pendente", "bbox": [100, 3100, 900, 3180], "suggestion": {"status": "suggested"}}
        self.store.update_mapping(self.model["id"], self.owner, {**definition, "fields": [confirmed, pending]})
        proposal = self.analyze()["proposal"]
        accepted = [field["field_id"] for field in proposal["fields"] if field["data_type"] != "checkbox"][:2]
        version = self.store.model(self.model["id"], self.owner)["version"]
        self.store.set_status(self.model["id"], self.owner, "published")
        saved = apply_proposal(self.store, self.catalog, self.model["id"], self.owner, proposal, accepted)
        self.assertEqual([field["field_id"] for field in saved["fields"]], ["manual", *accepted])
        self.assertFalse(any("suggestion" in field for field in saved["fields"]))
        self.assertEqual(saved["template_id"], definition["template_id"])
        provenance = saved["provenance"]
        self.assertEqual((provenance["method"], provenance["provider"], provenance["model"], provenance["reviewed"]),
                         ("ocrsd-hybrid", "none", None, True))
        self.assertEqual(provenance["analysis_id"], proposal["provenance"]["analysis_id"])
        self.assertEqual(provenance["registry_version"], proposal["provenance"]["registry_version"])
        self.assertEqual((provenance["accepted"], provenance["rejected"]), (2, len(proposal["fields"]) - 2))
        model = self.store.model(self.model["id"], self.owner)
        self.assertEqual((model["status"], model["version"], model["field_count"]), ("draft", version + 1, 3))
        self.assertFalse(any("sugest" in issue or "revisão" in issue for issue in mapping_issues(saved)))
        again = apply_proposal(self.store, self.catalog, self.model["id"], self.owner, proposal, accepted[:1])
        self.assertEqual([field["field_id"] for field in again["fields"]], ["manual", *accepted, f"{accepted[0]}_2"])

    def test_apply_proposal_rejections(self):
        proposal = self.analyze()["proposal"]
        identifier = proposal["fields"][0]["field_id"]
        cases = ((self.catalog, "certidao_nascimento", self.owner, proposal, [identifier], 403),
                 (self.catalog, self.model["id"], self.other, proposal, [identifier], 404),
                 (self.catalog, self.model["id"], self.owner, proposal, ["inexistente"], 422),
                 (self.catalog, self.model["id"], self.owner, {**proposal, "page_size_px": [100, 100]}, [identifier], 422),
                 (self.catalog, self.model["id"], self.owner,
                  {**proposal, "fields": [{**proposal["fields"][0], "bbox": [0, 0, 99999, 10]}]}, [identifier], 422))
        for catalog, model_id, owner, value, accepted, status in cases:
            with self.assertRaises(ServiceError) as error:
                apply_proposal(self.store, catalog, model_id, owner, value, accepted)
            self.assertEqual(error.exception.status, status)
        self.assertEqual(self.store.model(self.model["id"], self.owner)["version"], 1)


class GateJobTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(storage=Path(temporary.name), min_free_bytes=0, ai_provider="none", max_pending_analyses=2,
                                 jobs_per_hour=3)
        self.store = JobStore(self.settings)
        self.catalog = Catalog(self.settings)
        self.owner, _ = self.store.create_workspace()

    def model(self, owner: str | None = None) -> dict:
        return create_model(self.store, self.settings, owner or self.owner, "Declaração de Nascido Vivo", "", "default", "modelo.pdf",
                            synthetic_pdf())

    def test_page_size_matches_analysis_pixmap(self):
        import fitz

        for path in (CERTIDAO, ROOT / "documents/dnv/template.pdf", ROOT / "documents/do/do.pdf"):
            with fitz.open(path) as document:
                pixmap = document[0].get_pixmap(dpi=300)
                self.assertEqual(template_page_size(path.read_bytes()), (pixmap.width, pixmap.height), path.name)
        self.assertEqual(template_page_size(synthetic_pdf()), (2480, 3509))

    def test_apply_scales_and_clamps_to_stored_page_size(self):
        legacy = self.store.create_model(self.owner, "Antigo", "", "default", synthetic_pdf(),
                                         {"template_id": "modelo_web", "dpi": 300, "page_size_px": [2479, 3508], "fields": []})
        proposal = {"page_size_px": [2481, 3509], "provenance": {"method": "ocrsd-hybrid"}, "fields": [
            {"field_id": "nome", "label": "Nome", "data_type": "text", "bbox": [100, 100, 2481, 3509],
             "suggestion": {"status": "suggested", "candidate_id": "c001"}},
            {"field_id": "caixas", "label": "Caixas", "data_type": "digit_sequence", "bbox": [2400, 300, 2481, 360],
             "elements": [{"element_id": "caixas_0", "index": 0, "bbox": [2440, 300, 2481, 360]}],
             "suggestion": {"status": "suggested", "candidate_id": "c002"}}]}
        saved = apply_proposal(self.store, self.catalog, legacy["id"], self.owner, proposal, ["c001", "caixas"])
        self.assertEqual(saved["page_size_px"], [2479, 3508])
        fields = {field["field_id"]: field for field in saved["fields"]}
        self.assertEqual(fields["nome"]["bbox"], [100, 100, 2479, 3508])
        self.assertEqual(fields["caixas"]["elements"][0]["bbox"][2], 2479)
        imported = create_model(self.store, self.settings, self.owner, "Importado", "", "default", "modelo.pdf", synthetic_pdf(),
                                {"template_id": "modelo_web", "dpi": 300, "page_size_px": [2478, 3507],
                                 "fields": [{"field_id": "a", "label": "A", "data_type": "text", "bbox": [0, 0, 2478, 3507]}]})
        definition = self.catalog.definition(imported["id"], "default", self.owner)
        self.assertEqual((definition["page_size_px"], definition["fields"][0]["bbox"]), ([2480, 3509], [0, 0, 2480, 3509]))

    def test_analysis_queue_is_separate_and_limited_by_client(self):
        first, second, third = self.model(), self.model(), self.model()
        self.store.submit_analysis(first["id"], self.owner, client="10.0.0.1")
        self.store.submit_analysis(second["id"], self.owner, client="10.0.0.1")
        with self.assertRaises(ServiceError) as full:
            self.store.submit_analysis(third["id"], self.owner, client="10.0.0.2")
        self.assertEqual((full.exception.status, full.exception.code), (429, "queue_full"))
        options = GenerationOptions.model_validate({"document_type": "certidao_nascimento", "mapping_id": "certidao_nascimento_fields",
                                                    "fields": ["cpf"], "count": 1, "seed": 3,
                                                    "writing": {"model": "font", "handwriting": False, "text_handwriting": False}})
        self.store.submit(options, "10.0.0.9")
        with self.store.connection() as db:
            db.execute("UPDATE analyses SET status='failed'")
        other, _ = self.store.create_workspace()
        self.assertEqual(self.store.submit_analysis(self.model(other)["id"], other, client="10.0.0.1")["status"], "queued")
        fresh, _ = self.store.create_workspace()
        with self.assertRaises(ServiceError) as limited:
            self.store.submit_analysis(self.model(fresh)["id"], fresh, client="10.0.0.1")
        self.assertEqual((limited.exception.status, limited.exception.code), (429, "rate_limit"))
        self.assertEqual(self.store.submit_analysis(self.model(fresh)["id"], fresh, client="10.0.0.3")["status"], "queued")

    def test_timeout_keeps_provenance_and_settings_budget(self):
        with self.assertRaises(ValueError):
            Settings(ai_budget=300, analysis_timeout=300)
        self.assertLess(Settings().ai_budget, Settings().analysis_timeout)
        queued = self.store.submit_analysis(self.model()["id"], self.owner)
        self.store.settings.analysis_timeout = 0
        run_analysis(self.store, self.store.claim_analysis())
        failed = self.store.analysis(queued["id"], self.owner)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("Tempo máximo de análise", failed["error"])
        self.assertEqual((failed["provenance"]["analysis_id"], failed["provenance"]["provider"]), (queued["id"], "none"))
        self.assertEqual(len(failed["provenance"]["pdf_sha256"]), 64)

    def test_ai_cache_is_counted_and_cleaned(self):
        self.store.ai_cache.mkdir()
        old, recent = self.store.ai_cache / "old.json", self.store.ai_cache / "recent.json"
        old.write_text("{}" * 100)
        recent.write_text("{}")
        os.utime(old, (time.time() - 48 * 3600,) * 2)
        self.assertEqual(self.store.cache_bytes(), 202)
        self.store.cleanup()
        self.assertEqual((old.exists(), recent.exists()), (False, True))

    def test_successful_analysis_records_assistant_summary(self):
        queued = self.store.submit_analysis(self.model()["id"], self.owner)
        process_analysis(self.store, self.store.claim_analysis())
        result = self.store.analysis(queued["id"], self.owner)
        self.assertEqual(result["provenance"]["assistant"]["status"], "unavailable")
        self.assertEqual(result["proposal"]["provenance"]["assistant"]["configured"], "none")


class AnalysisApiTests(unittest.TestCase):
    def setUp(self):
        try:
            from tests.test_web import ASGITestClient
        except unittest.SkipTest:
            self.skipTest("Instale os extras web e test")
        from src.web.api import create_app

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(storage=Path(temporary.name), min_free_bytes=0, neural_enabled=False)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        self.owner = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        self.other = self.client.post("/api/v1/workspaces", json={}).json()["token"]

    def request(self, method: str, path: str, token: str | None = None, **kwargs):
        return self.client.request(method, path, headers={"X-Workspace-Token": token} if token else {}, **kwargs)

    def run_worker(self):
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items() if value is not None},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_analysis_endpoints_ownership_and_generation_priority(self):
        created = self.request("POST", "/api/v1/models", self.owner, json={
            "label": "Certidão de nascimento importada", "description": "", "filename": "certidao.pdf", "mapping_id": "default",
            "pdf_base64": base64.b64encode(CERTIDAO.read_bytes()).decode(), "mapping": None})
        self.assertEqual(created.status_code, 201, created.text)
        model_id = created.json()["id"]
        builtin = self.request("POST", "/api/v1/models/certidao_nascimento/analyses", self.owner, json={"page": 1})
        self.assertEqual(builtin.status_code, 403)
        self.assertEqual(builtin.json()["error"]["message"], BUILTIN_PROTECTED)
        self.assertEqual(self.request("POST", f"/api/v1/models/{model_id}/analyses", json={"page": 1}).status_code, 401)
        self.assertEqual(self.request("POST", f"/api/v1/models/{model_id}/analyses", self.other, json={"page": 1}).status_code, 404)
        self.assertEqual(self.request("POST", f"/api/v1/models/{model_id}/analyses", self.owner, json={"page": 0}).status_code, 422)
        queued = self.request("POST", f"/api/v1/models/{model_id}/analyses", self.owner, json={"page": 1})
        self.assertEqual(queued.status_code, 202, queued.text)
        analysis_id = queued.json()["id"]
        self.assertEqual(queued.json()["status"], "queued")
        self.assertEqual(self.request("GET", f"/api/v1/analyses/{analysis_id}", self.other).status_code, 404)
        generation = self.request("POST", "/api/v1/jobs", self.owner, json={
            "document_type": "certidao_nascimento", "mapping_id": "certidao_nascimento_fields", "fields": ["cpf"], "count": 1,
            "seed": 7, "writing": {"model": "font", "handwriting": False, "text_handwriting": False}})
        self.assertEqual(generation.status_code, 202, generation.text)
        self.run_worker()
        self.assertEqual(self.app.state.store.get(generation.json()["id"])["status"], "succeeded")
        self.assertEqual(self.request("GET", f"/api/v1/analyses/{analysis_id}", self.owner).json()["status"], "queued")
        self.run_worker()
        result = self.request("GET", f"/api/v1/analyses/{analysis_id}", self.owner)
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual(body["status"], "succeeded", body["error"])
        self.assertEqual(body["proposal"]["template_id"], "certidao_nascimento")
        self.assertEqual(body["provenance"]["provider"], "none")
        self.assertGreater(len(body["proposal"]["fields"]), 10)

    def test_assistant_status_is_fast_and_cached(self):
        self.assertEqual(self.request("GET", "/api/v1/assistant").status_code, 401)
        started = time.monotonic()
        response = self.request("GET", "/api/v1/assistant", self.owner)
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"available": False, "provider": "none", "model": None, "detail": "Assistente de IA desativado"})
        from src.web.api import create_app

        app = create_app(Settings(**{**self.settings.model_dump(), "ai_provider": "openai", "ai_base_url": "http://127.0.0.1:9/v1"}))
        from tests.test_web import ASGITestClient

        client = ASGITestClient(app)
        self.addCleanup(client.close)
        token = client.post("/api/v1/workspaces", json={}).json()["token"]
        started = time.monotonic()
        offline = client.get("/api/v1/assistant", headers={"X-Workspace-Token": token}).json()
        self.assertLess(time.monotonic() - started, 2.5)
        self.assertEqual((offline["available"], offline["provider"], offline["model"]), (False, "openai", "qwen3:1.7b"))
        self.assertEqual(client.get("/api/v1/assistant", headers={"X-Workspace-Token": token}).json(), offline)

    def test_apply_endpoint_and_single_page_rules(self):
        from src.web.services import create_blank_model

        created = self.request("POST", "/api/v1/models", self.owner, json={
            "label": "Certidão de nascimento importada", "description": "", "filename": "certidao.pdf", "mapping_id": "default",
            "pdf_base64": base64.b64encode(CERTIDAO.read_bytes()).decode(), "mapping": None})
        model_id = created.json()["id"]
        self.assertEqual(self.request("POST", f"/api/v1/models/{model_id}/analyses", self.owner, json={"page": 2}).status_code, 422)
        store = self.app.state.store
        composed = create_blank_model(store, self.owner.partition(".")[0], "Composto", "", "a4", "portrait")
        refused = self.request("POST", f"/api/v1/models/{composed['id']}/analyses", self.owner, json={"page": 1})
        self.assertEqual((refused.status_code, refused.json()["error"]["code"]), (422, "mapping_assist_unavailable"))
        analysis_id = self.request("POST", f"/api/v1/models/{model_id}/analyses", self.owner, json={"page": 1}).json()["id"]
        path = f"/api/v1/models/{model_id}/analyses/{analysis_id}/apply"
        self.assertEqual(self.request("POST", path, self.owner, json={"accepted": []}).status_code, 409)
        process_analysis(store, store.claim_analysis())
        proposal = store.analysis(analysis_id, self.owner.partition(".")[0])["proposal"]
        candidate_id = proposal["fields"][0]["suggestion"]["candidate_id"]
        self.assertEqual(self.request("POST", path, self.other, json={"accepted": [candidate_id]}).status_code, 404)
        self.assertEqual(self.request("POST", f"/api/v1/models/certidao_nascimento/analyses/{analysis_id}/apply", self.owner,
                                      json={"accepted": [candidate_id]}).status_code, 403)
        self.assertEqual(self.request("POST", path, self.owner, json={"accepted": ["inexistente"]}).status_code, 422)
        applied = self.request("POST", path, self.owner, json={"accepted": [candidate_id]})
        self.assertEqual(applied.status_code, 200, applied.text)
        fields = applied.json()["definition"]["fields"]
        self.assertEqual([field["field_id"] for field in fields], [proposal["fields"][0]["field_id"]])
        self.assertTrue(applied.json()["definition"]["provenance"]["reviewed"])
        store.set_status(model_id, self.owner.partition(".")[0], "published")
        self.assertEqual(self.request("POST", path, self.owner, json={"accepted": [candidate_id]}).status_code, 409)

    def test_assistant_status_does_not_queue_behind_hung_check(self):
        from http.server import ThreadingHTTPServer
        from src.web.api import create_app
        from tests.test_mapping_matching import FakeModel
        from tests.test_web import ASGITestClient

        FakeModel.get_delay = 4.0
        self.addCleanup(setattr, FakeModel, "get_delay", 0.0)
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        app = create_app(Settings(**{**self.settings.model_dump(), "ai_provider": "openai", "ai_model": "fake",
                                     "ai_base_url": f"http://127.0.0.1:{server.server_address[1]}/v1"}))
        client = ASGITestClient(app)
        self.addCleanup(client.close)
        token = client.post("/api/v1/workspaces", json={}).json()["token"]
        durations = []

        def call():
            started = time.monotonic()
            client.get("/api/v1/assistant", headers={"X-Workspace-Token": token})
            durations.append(time.monotonic() - started)

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(len(durations), 3)
        self.assertLess(max(durations), 3.5, durations)


class EvaluationTests(unittest.TestCase):
    def test_certidao_text_layer_metrics(self):
        from scripts.evaluate_mapping import PRESETS, evaluate

        template, mapping = PRESETS["certidao_nascimento"]
        result = evaluate(template, mapping, "certidao_nascimento", ocr=False, assistant=NullAssistant())
        self.assertEqual((result["source"], result["provider"]), ("text_layer", "none"))
        loose, strict = result["metrics"]["iou_0.3"], result["metrics"]["iou_0.5"]
        self.assertGreaterEqual(loose["recall"], 0.6)
        self.assertGreaterEqual(loose["recall"], strict["recall"])
        self.assertIsNotNone(loose["semantic_accuracy"])
        self.assertGreater(loose["semantic_compared"], 5)
        self.assertGreaterEqual(loose["manual_corrections"], len(loose["unmatched_reference"]) + len(loose["false_positives"]))
        self.assertEqual(sum(result["status_counts"].values()), result["proposed_fields"])
        self.assertGreater(result["timings_seconds"]["total"], 0)
        self.assertGreater(result["peak_rss_mib"], 0)


if __name__ == "__main__":
    unittest.main()
