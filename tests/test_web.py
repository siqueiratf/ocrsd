from concurrent.futures import ThreadPoolExecutor
import asyncio
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from zipfile import ZipFile
from unittest.mock import patch

try:
    import httpx2
    import jsonschema
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.jobs import JobStore
from src.web.models import GenerationOptions
from src.web.services import Catalog, ServiceError, generate
from src.web.models import Writing
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


class ASGITestClient:
    def __init__(self, app):
        self.app = app
        self.local = threading.local()
        self.loops = []
        self.lock = threading.Lock()

    def request(self, method: str, path: str, **kwargs):
        async def send():
            transport = httpx2.ASGITransport(app=self.app)
            async with httpx2.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)
        loop = getattr(self.local, "loop", None)
        if loop is None:
            loop = asyncio.new_event_loop()
            self.local.loop = loop
            with self.lock:
                self.loops.append(loop)
        return loop.run_until_complete(send())

    def get(self, path: str, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs):
        return self.request("POST", path, **kwargs)

    def put(self, path: str, **kwargs):
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs):
        return self.request("DELETE", path, **kwargs)

    def close(self):
        with self.lock:
            for loop in self.loops:
                if not loop.is_closed():
                    loop.close()
            self.loops.clear()


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0, neural_enabled=False)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        self.store = self.app.state.store
        self.options = GenerationOptions(document_type="certidao_nascimento", mapping_id="certidao_nascimento_fields",
                                         fields=["nome_registrado"], count=1, seed=42)
        definition = Catalog(self.settings).definition(self.options.document_type, self.options.mapping_id)
        self.options.fields = [field["field_id"] for field in definition["fields"][:3]]

    def submit(self):
        response = self.client.post("/api/v1/jobs", json=self.options.model_dump())
        self.assertEqual(response.status_code, 202, response.text)
        job = response.json()
        return job, {"Authorization": "Bearer " + job["token"]}

    def worker_environment(self):
        return {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items() if value is not None},
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run_worker(self):
        return subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT,
                              env=self.worker_environment(), capture_output=True, text=True, timeout=90, check=True)

    def assert_contract(self, response, path, method="get"):
        schema = self.app.openapi()
        definition = schema["paths"][path][method]["responses"][str(response.status_code)]["content"]["application/json"]["schema"]
        jsonschema.Draft202012Validator({**definition, "components": schema["components"]}).validate(response.json())

    def test_catalog_contract_and_no_tkinter_import(self):
        for path in ("health", "document-types", "capabilities", "writing-presets"):
            response = self.client.get("/api/v1/" + path)
            self.assertEqual(response.status_code, 200)
            self.assert_contract(response, "/api/v1/" + path)
        for document in self.client.get("/api/v1/document-types").json():
            template = self.client.get(f'/api/v1/document-types/{document["id"]}/template')
            self.assertTrue(template.content.startswith(b"%PDF"))
            for mapping in document["mappings"]:
                response = self.client.get(f'/api/v1/document-types/{document["id"]}/mappings/{mapping["id"]}')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["definition"]["source_template"], document["id"])
                self.assert_contract(response, "/api/v1/document-types/{document_type}/mappings/{mapping_id}")
        result = subprocess.run([sys.executable, "-c", "import sys; from src.web.api import create_app; assert 'tkinter' not in sys.modules; assert 'torch' not in sys.modules"], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def workspace_cookie(self) -> str:
        response = self.client.get("/modelos")
        return re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)

    def test_generation_wizard_groups_models_by_orgao_with_hover_actions(self):
        headers = {"Cookie": f"ocrsd_workspace={self.workspace_cookie()}"}
        page = self.client.get("/geracao", headers=headers)
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn('class="wizard-org-group"', page.text)
        self.assertIn('class="wizard-model-actions"', page.text)
        self.assertIn("/modelos/certidao_nascimento_cnj/template.pdf", page.text)
        page = self.client.get("/geracao", headers=headers, params={"modelo": "certidao_nascimento_cnj"})
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn('<details class="wizard-org-group" open>', page.text)
        catalog = Catalog(self.settings)
        documentacao = catalog.types["certidao_nascimento_cnj"].documentacao
        self.assertTrue(documentacao)
        self.assertIn(f'href="{documentacao}"', page.text)
        self.assertIn('target="_blank"', page.text)

    def test_model_template_download(self):
        headers = {"Cookie": f"ocrsd_workspace={self.workspace_cookie()}"}
        response = self.client.get("/modelos/certidao_nascimento_cnj/template.pdf", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertIn("attachment", response.headers["content-disposition"])
        self.assertTrue(response.content.startswith(b"%PDF"))
        missing = self.client.get("/modelos/documento-inexistente/template.pdf", headers=headers)
        self.assertEqual(missing.status_code, 404)

    def test_catalog_endpoint_lists_orgaos_and_documentos_including_planned(self):
        data = self.client.get("/api/v1/catalog").json()
        self.assertIn("cnj", data["orgaos"])
        self.assertEqual(data["documentos"]["certidao_nascimento_cnj"]["status"], "disponivel")
        self.assertTrue(any(item["status"] == "planejado" for item in data["documentos"].values()))

    def test_strict_validation_and_bounded_body(self):
        self.settings.max_count = 10
        changes = [{"count": 0}, {"count": True}, {"count": "1"}, {"count": 11}, {"seed": -1},
                   {"fields": []}, {"fields": ["unknown"]}, {"fields": self.options.fields * 2},
                   {"document_type": "../dnv"}, {"output": "/tmp/escape"}, {"filter_preset": "other"},
                   {"writing": {"model": "sjvasquez"}}, {"writing": {"writer": 512}},
                   {"writing": {"model": "sjvasquez", "writer": 13}}, {"writing": {"intensity": "NaN"}}]
        for update in changes:
            with self.subTest(update=update):
                response = self.client.post("/api/v1/generation/validate", json={**self.options.model_dump(), **update})
                self.assertEqual(response.status_code, 422, response.text)
                self.assert_contract(response, "/api/v1/generation/validate", "post")
        response = self.client.post("/api/v1/jobs", content="{}", headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 415)
        response = self.client.post("/api/v1/jobs", content="x" * 32769, headers={"Content-Type": "application/json", "Content-Length": "2"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.get("/api/v1/missing").json()["error"]["code"], "http_error")

    def test_real_worker_download_preview_and_determinism(self):
        import fitz

        outputs = []
        for iteration in range(2):
            job, headers = self.submit()
            self.run_worker()
            base = f'/api/v1/jobs/{job["id"]}'
            response = self.client.get(base, headers=headers)
            self.assertEqual(response.json()["status"], "succeeded", response.text)
            self.assertEqual(response.json()["completed"], 1)
            self.assert_contract(response, "/api/v1/jobs/{job_id}")
            response = self.client.get(base + "/results", headers=headers)
            self.assert_contract(response, "/api/v1/jobs/{job_id}/results")
            files = response.json()["artifacts"]
            pdf = next(item for item in files if item["media_type"] == "application/pdf")
            preview = self.client.get(base + f'/artifacts/{pdf["id"]}?download=false', headers=headers)
            self.assertIn("inline", preview.headers["content-disposition"])
            with fitz.open(stream=preview.content, filetype="pdf") as document:
                self.assertIn("SEM VALIDADE LEGAL", document[0].get_text())
                raster = document[0].get_pixmap().samples
            directory = self.store.directory(job["id"])
            ground_truth = json.loads((directory / "synthetic_0001.json").read_text())
            self.assertTrue(ground_truth["synthetic"])
            self.assertNotIn(str(ROOT), json.dumps(ground_truth))
            with ZipFile(directory / "dataset.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"synthetic_0001.pdf", "synthetic_0001.json", "dataset.json"})
                self.assertTrue(json.loads(archive.read("dataset.json"))["synthetic"])
            outputs.append((ground_truth, raster))
        self.assertEqual(outputs[0], outputs[1])

    def test_generated_seed_is_preserved_for_web_reproduction(self):
        options = self.options.model_copy(update={"seed": None})
        output = Path(self.temporary.name) / "unseeded"
        generate(options, output, self.settings, lambda completed: None)
        manifest = json.loads((output / "dataset.json").read_text())
        truth = json.loads((output / "synthetic_0001.json").read_text())
        self.assertIsInstance(manifest["seed"], int)
        self.assertEqual(manifest["seed"], truth["generation_seed"])
        self.assertEqual(manifest["seed"], manifest["options"]["seed"])
        repeated = Path(self.temporary.name) / "repeated"
        generate(GenerationOptions.model_validate(manifest["options"]), repeated, self.settings, lambda completed: None)
        self.assertEqual(truth["fields"], json.loads((repeated / "synthetic_0001.json").read_text())["fields"])

    def test_artifact_isolation_cancel_and_expiration(self):
        first, headers = self.submit()
        second, second_headers = self.submit()
        base = f'/api/v1/jobs/{first["id"]}'
        self.assertEqual(self.client.get(base).status_code, 401)
        self.assertEqual(self.client.get(base, headers=second_headers).status_code, 404)
        self.assertEqual(self.client.get(base + "/results", headers=headers).status_code, 409)
        response = self.client.post(base + "/cancel", headers=headers, json={})
        self.assertEqual(response.json()["status"], "cancelled")
        self.assert_contract(response, "/api/v1/jobs/{job_id}/cancel", "post")
        directory = self.store.directory(second["id"])
        directory.mkdir()
        outside = self.settings.storage / "private.txt"
        outside.write_text("privado")
        (directory / "link.txt").symlink_to(outside)
        self.store.finish(second["id"], "succeeded", artifacts=[{"id": "0", "name": "link.txt", "media_type": "text/plain", "size": 7}])
        url = f'/api/v1/jobs/{second["id"]}/artifacts/0'
        self.assertEqual(self.client.get(url, headers=second_headers).status_code, 404)
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET expires_at=? WHERE id=?", (time.time() - 1, second["id"]))
        self.assertEqual(self.client.get(url, headers=second_headers).status_code, 410)
        self.store.cleanup()
        self.assertFalse(directory.exists())
        self.assertTrue(outside.exists())

    def test_concurrent_requests_respect_atomic_queue_limit(self):
        self.settings.max_pending = 2
        with ThreadPoolExecutor(max_workers=8) as executor:
            responses = list(executor.map(lambda _: self.client.post("/api/v1/jobs", json=self.options.model_dump()), range(8)))
        self.assertEqual([response.status_code for response in responses].count(202), 2)
        self.assertEqual([response.status_code for response in responses].count(429), 6)
        self.assertEqual(len({response.json()["id"] for response in responses if response.status_code == 202}), 2)
        with ThreadPoolExecutor(max_workers=4) as executor:
            claimed = list(executor.map(lambda _: self.store.claim(), range(4)))
        self.assertEqual(len([row for row in claimed if row]), 2)

    def test_rate_storage_and_maintenance_limits(self):
        self.settings.jobs_per_hour = 1
        self.submit()
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).json()["error"]["code"], "rate_limit")
        self.settings.accepting_jobs = False
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).status_code, 503)
        self.settings.accepting_jobs = True
        self.settings.jobs_per_hour = 10
        self.settings.max_storage_bytes = self.settings.max_job_bytes
        self.assertEqual(self.client.post("/api/v1/jobs", json=self.options.model_dump()).json()["error"]["code"], "storage_quota")

    def test_unlimited_count_has_no_quota_ceiling(self):
        self.assertIsNone(self.settings.max_count)
        catalog = Catalog(self.settings)
        self.options.count = 1000
        self.assertIs(catalog.validate(self.options), self.options)

    def test_configured_max_count_still_rejects_larger_counts(self):
        catalog = Catalog(self.settings)
        self.settings.max_count = 5
        self.options.count = 6
        with self.assertRaises(ServiceError) as caught:
            catalog.validate(self.options)
        self.assertEqual(caught.exception.code, "quota")
        self.options.count = 5
        self.assertIs(catalog.validate(self.options), self.options)

    def test_job_budgets_scale_with_document_count(self):
        self.assertEqual(self.settings.job_byte_budget(0), self.settings.max_job_bytes)
        self.assertEqual(self.settings.job_byte_budget(10), self.settings.max_job_bytes + 10 * self.settings.job_bytes_per_document)
        self.assertEqual(self.settings.job_time_budget(0), self.settings.job_timeout)
        self.assertEqual(self.settings.job_time_budget(10), self.settings.job_timeout + 10 * self.settings.job_seconds_per_document)

    def test_large_count_job_exceeds_storage_quota_via_scaled_budget(self):
        self.settings.max_storage_bytes = self.settings.job_byte_budget(1)
        self.options.count = 1000
        response = self.client.post("/api/v1/jobs", json=self.options.model_dump())
        self.assertEqual(response.json()["error"]["code"], "storage_quota")

    def test_worker_failure_and_recovery(self):
        job, headers = self.submit()
        with self.store.connection() as db:
            db.execute("UPDATE jobs SET options='{}' WHERE id=?", (job["id"],))
        self.run_worker()
        response = self.client.get(f'/api/v1/jobs/{job["id"]}', headers=headers)
        self.assertEqual(response.json()["status"], "failed")
        self.assertNotIn("Traceback", response.text)
        self.assertFalse(self.store.directory(job["id"]).exists())
        queued, token = self.store.submit(self.options, "another-client")
        self.store.claim()
        self.store.recover()
        self.assertEqual(self.store.get(queued["id"])["status"], "failed")

    def test_running_cancellation_and_timeout(self):
        for cancel in (True, False):
            with self.subTest(cancel=cancel):
                self.settings.job_timeout = 90 if cancel else 1
                self.settings.job_seconds_per_document = 0
                self.options.count = 100
                job, headers = self.submit()
                process = subprocess.Popen([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT,
                                           env=self.worker_environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    deadline = time.monotonic() + 15
                    while self.store.get(job["id"])["status"] == "queued" and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertEqual(self.store.get(job["id"])["status"], "running")
                    if cancel:
                        self.client.post(f'/api/v1/jobs/{job["id"]}/cancel', headers=headers, json={})
                    process.wait(timeout=20)
                    self.assertEqual(self.store.get(job["id"])["status"], "cancelled" if cancel else "failed")
                    finalized = self.store.artifacts(self.store.get(job["id"]))
                    self.assertEqual(self.store.directory(job["id"]).exists(), bool(finalized))
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait()

    def test_service_reuses_dnv_engine_and_filters(self):
        options = GenerationOptions(document_type="dnv", mapping_id="mapeamento_seguindo_norma", fields=["1"], filter_preset="light")
        definition = Catalog(self.settings).definition(options.document_type, options.mapping_id)
        options.fields = [definition["fields"][0]["field_id"]]
        progress = []
        output = self.settings.storage / "service"
        artifacts = generate(options, output, self.settings, progress.append)
        self.assertEqual(progress, [1])
        self.assertEqual(len(artifacts), 4)
        data = json.loads((output / "synthetic_0001.json").read_text())
        self.assertTrue(data["visual_filters"]["enabled"])
        self.assertEqual(data["fields"][0]["field_id"], options.fields[0])
        self.assertLess((output / "synthetic_0001.pdf").stat().st_size, 10 * 1024 * 1024)
        with self.assertRaises(FileExistsError):
            generate(options, output, self.settings, progress.append)

    def test_neural_strict_charset_rejected_before_submission(self):
        catalog = Catalog(self.settings)
        available = [{"id": "sjvasquez", "available": True}]
        with patch.object(catalog, "models", return_value=available):
            options = GenerationOptions(document_type="dnv", mapping_id="default", fields=["2"],
                                        writing=Writing(model="sjvasquez", text_policy="strict"))
            with self.assertRaises(ServiceError) as caught:
                catalog.validate(options)
            self.assertEqual(caught.exception.code, "writing_incompatible")
            options.writing.text_policy = "normalize"
            self.assertIs(catalog.validate(options), options)

    def test_neural_strict_check_covers_every_generated_value(self):
        catalog = Catalog(self.settings)
        with patch.object(catalog, "models", return_value=[{"id": "aaron1259", "available": True}]):
            for seed in range(12):
                options = GenerationOptions(document_type="dnv", mapping_id="mapeamento_seguindo_norma", fields=["1"], seed=seed,
                                            writing=Writing(model="aaron1259", text_policy="strict"))
                with self.assertRaises(ServiceError) as caught:
                    catalog.validate(options)
                self.assertEqual(caught.exception.code, "writing_incompatible")
                self.assertIn("Normalizar", caught.exception.message)
            dates = GenerationOptions(document_type="dnv", mapping_id="default", fields=["2"],
                                      writing=Writing(model="aaron1259", text_policy="strict", text_handwriting=False))
            with self.assertRaisesRegex(ServiceError, "/"):
                catalog.validate(dates)
            numbers = GenerationOptions(document_type="dnv", mapping_id="mapeamento_seguindo_norma", fields=["2", "4", "12_codigo"],
                                        writing=Writing(model="aaron1259", text_policy="strict", text_handwriting=False))
            self.assertIs(catalog.validate(numbers), numbers)

    def test_validation_endpoint_explains_incompatible_writing(self):
        options = {**self.options.model_dump(), "document_type": "dnv", "mapping_id": "mapeamento_seguindo_norma", "fields": ["1", "14"],
                   "writing": Writing(model="aaron1259", text_policy="strict").model_dump()}
        with patch.object(Catalog, "models", return_value=[{"id": "aaron1259", "available": True}]):
            response = self.client.post("/api/v1/generation/validate", json=options)
            submitted = self.client.post("/api/v1/jobs", json=options)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "writing_incompatible")
        self.assertEqual(submitted.status_code, 422)

    def test_all_dnv_fields_generate_with_neural_numbers(self):
        model = unittest.mock.Mock()
        model.sample.return_value = __import__("numpy").array([[0, 0, 0], [10, 1, 0], [20, 0, 1]], dtype=float)
        catalog = Catalog(self.settings)
        definition = catalog.definition("dnv", "mapeamento_seguindo_norma")
        options = GenerationOptions(document_type="dnv", mapping_id="mapeamento_seguindo_norma", count=2, seed=5,
                                    fields=[str(field["field_id"]) for field in definition["fields"]
                                            if not field.get("unsupported_reason")],
                                    writing=Writing(model="aaron1259", text_policy="normalize"))
        with patch.object(Catalog, "models", return_value=[{"id": "aaron1259", "available": True}]), \
                patch("src.synthetic_dataset.load_model", return_value=model):
            artifacts = generate(options, self.settings.storage / "neural", self.settings, lambda completed: None)
        truth = json.loads((self.settings.storage / "neural" / "synthetic_0001.json").read_text())
        code = next(field for field in truth["fields"] if field["field_id"] == "12_codigo")
        self.assertRegex(code["requested_value"], r"^[0-9]{7}$")
        self.assertEqual(sum(item["media_type"] == "application/pdf" for item in artifacts), 2)

    def test_sequence_without_boxes_is_rejected_before_submission(self):
        catalog = Catalog(self.settings)
        options = GenerationOptions(document_type="dnv", mapping_id="test_w_datetime", fields=["60"])
        with self.assertRaises(ServiceError) as caught:
            catalog.validate(options)
        self.assertEqual(caught.exception.code, "invalid_mapping")
        self.assertIn("CEP", caught.exception.message)

    def test_builtin_models_are_protected_and_duplicable(self):
        mapping_path = ROOT / "documents/dnv/mappings/mapeamento_seguindo_norma.json"
        original = mapping_path.read_bytes()
        workspace = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        headers = {"X-Workspace-Token": workspace}
        model = next(item for item in self.client.get("/api/v1/models", headers=headers).json() if item["id"] == "dnv")
        self.assertTrue(model["builtin"])
        self.assertFalse(model["editable"])
        self.assertEqual(model["mapping_id"], "mapeamento_seguindo_norma")
        mapping = self.client.get(f'/api/v1/document-types/dnv/mappings/{model["mapping_id"]}', headers=headers).json()
        mapping["definition"]["fields"][0]["description"] = "Editado na Web"
        response = self.client.put("/api/v1/models/dnv/mapping", json={"definition": mapping["definition"]}, headers=headers)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(response.json()["error"]["code"], "builtin_protected")
        self.assertEqual(mapping_path.read_bytes(), original)
        copy = self.client.post("/api/v1/models/dnv/duplicate", json={}, headers=headers)
        self.assertEqual(copy.status_code, 201, copy.text)
        self.assertFalse(copy.json()["builtin"])
        self.assertEqual(copy.json()["field_count"], model["field_count"])
        edited = self.client.put(f'/api/v1/models/{copy.json()["id"]}/mapping', json={"definition": mapping["definition"]}, headers=headers)
        self.assertEqual(edited.status_code, 200, edited.text)
        self.assertEqual(mapping_path.read_bytes(), original)

    def test_unreviewed_assisted_mapping_cannot_be_published_by_api(self):
        token = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        headers = {"X-Workspace-Token": token}
        copy = self.client.post("/api/v1/models/certidao_nascimento/duplicate", json={}, headers=headers).json()
        model_id, mapping_id = copy["id"], copy["mapping_id"]
        path = f"/api/v1/document-types/{model_id}/mappings/{mapping_id}"
        definition = self.client.get(path, headers=headers).json()["definition"]
        definition["fields"][0]["suggestion"] = {"status": "suggested", "candidate_id": "c001"}
        definition["provenance"] = {"method": "ocrsd-hybrid", "reviewed": False}
        self.assertEqual(self.client.put(f"/api/v1/models/{model_id}/mapping",
                                         json={"definition": definition}, headers=headers).status_code, 200)
        payload = {"label": copy["label"], "description": copy["description"], "status": "published"}
        blocked = self.client.put(f"/api/v1/models/{model_id}", json=payload, headers=headers)
        self.assertEqual(blocked.status_code, 422, blocked.text)
        self.assertEqual(blocked.json()["error"]["code"], "publication_blocked")
        definition["fields"][0].pop("suggestion")
        definition["provenance"]["reviewed"] = True
        self.assertEqual(self.client.put(f"/api/v1/models/{model_id}/mapping",
                                         json={"definition": definition}, headers=headers).status_code, 200)
        published = self.client.put(f"/api/v1/models/{model_id}", json=payload, headers=headers)
        self.assertEqual(published.status_code, 200, published.text)
        self.assertEqual(published.json()["status"], "published")
        definition["fields"][0]["suggestion"] = {"status": "suggested", "candidate_id": "c002"}
        definition["provenance"]["reviewed"] = False
        self.assertEqual(self.client.put(f"/api/v1/models/{model_id}/mapping",
                                         json={"definition": definition}, headers=headers).status_code, 200)
        self.assertEqual(self.store.model(model_id, token.partition(".")[0])["status"], "draft")

    def test_custom_semantic_definition_is_approved_and_embedded_in_mapping(self):
        token = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        headers = {"X-Workspace-Token": token}
        spec = {"key": "custom.referencia_local", "label": "Referência local", "description": "Código fictício",
                "data_type": "text", "generator": {"type": "choices", "values": ["A1", "B2"]}}
        created = self.client.post("/api/v1/semantic-definitions", json=spec, headers=headers)
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["status"], "approved")
        self.assertEqual(len(self.client.get("/api/v1/semantic-definitions", headers=headers).json()), 1)
        copy = self.client.post("/api/v1/models/certidao_nascimento/duplicate", json={}, headers=headers).json()
        model_id, mapping_id = copy["id"], copy["mapping_id"]
        definition = self.client.get(f"/api/v1/document-types/{model_id}/mappings/{mapping_id}",
                                     headers=headers).json()["definition"]
        definition["fields"][0]["semantic"] = spec["key"]
        definition["fields"][0]["data_type"] = "text"
        saved = self.client.put(f"/api/v1/models/{model_id}/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["definition"]["semantic_definitions"][spec["key"]], spec)
        other = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        self.assertEqual(self.client.get("/api/v1/semantic-definitions", headers={"X-Workspace-Token": other}).json(), [])

    def test_private_workspace_models_presets_history_and_revocable_sharing(self):
        workspace = self.client.post("/api/v1/workspaces", json={})
        self.assertEqual(workspace.status_code, 201, workspace.text)
        workspace_headers = {"X-Workspace-Token": workspace.json()["token"]}
        template_bytes = (ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf").read_bytes()
        uploaded = self.client.post("/api/v1/models", json={
            "label": "Template enviado", "description": "Cadastro PDF pela API", "mapping_id": "default",
            "filename": "template.pdf", "pdf_base64": "data:application/pdf;base64," + base64.b64encode(template_bytes).decode(),
        }, headers=workspace_headers)
        self.assertEqual(uploaded.status_code, 201, uploaded.text)
        self.assertEqual(self.client.get(f'/api/v1/models/{uploaded.json()["id"]}/preview', headers=workspace_headers).status_code, 200)
        self.assertEqual(self.client.get(f'/api/v1/models/{uploaded.json()["id"]}/preview').status_code, 404)
        self.assertEqual(self.client.delete(f'/api/v1/models/{uploaded.json()["id"]}', headers=workspace_headers).status_code, 200)
        models = self.client.get("/api/v1/models", headers=workspace_headers)
        self.assertGreaterEqual(len(models.json()), 2)
        source = next(item for item in models.json() if item["id"] == "certidao_nascimento")
        copied = self.client.post(f'/api/v1/models/{source["id"]}/duplicate', json={}, headers=workspace_headers)
        self.assertEqual(copied.status_code, 201, copied.text)
        model = copied.json()
        self.assertFalse(model["builtin"])
        mapping = self.client.get(
            f'/api/v1/document-types/{model["id"]}/mappings/{model["mapping_id"]}', headers=workspace_headers,
        ).json()
        mapping["definition"]["fields"][0]["description"] = "Alterado pela interface web"
        saved = self.client.put(f'/api/v1/models/{model["id"]}/mapping', json={"definition": mapping["definition"]}, headers=workspace_headers)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["definition"]["fields"][0]["description"], "Alterado pela interface web")
        preview = self.client.get(f'/api/v1/models/{model["id"]}/preview', headers=workspace_headers)
        self.assertEqual(preview.headers["content-type"], "image/png")
        writing = self.options.writing.model_dump()
        preset = self.client.post("/api/v1/presets", json={"name": "Privado", "writing": writing,
                                                           "filter_preset": "light", "shareable": False}, headers=workspace_headers)
        self.assertEqual(preset.status_code, 201, preset.text)
        self.assertFalse(preset.json()["builtin"])
        samples = self.client.post("/api/v1/handwriting/preview", json={"writing": writing, "texts": ["Ana Silva", "15092026"]}, headers=workspace_headers)
        self.assertEqual(samples.status_code, 200, samples.text)
        self.assertTrue(all(image.startswith("data:image/png;base64,") for image in samples.json()["images"]))
        realism = self.client.post("/api/v1/realism/preview", json={"document_type": model["id"], "preset": "light", "seed": 42}, headers=workspace_headers)
        self.assertEqual(realism.status_code, 200, realism.text)
        self.assertEqual(realism.json()["metadata"]["preset"], "light")
        custom_options = GenerationOptions(name="Execução privada", document_type=model["id"], mapping_id=model["mapping_id"],
                                           fields=[item["field_id"] for item in saved.json()["fields"][:2]], count=1, seed=73)
        configuration = self.client.post("/api/v1/configurations", json={"name": "Reutilizável", "options": custom_options.model_dump()}, headers=workspace_headers)
        self.assertEqual(configuration.status_code, 201, configuration.text)
        job = self.client.post("/api/v1/jobs", json=custom_options.model_dump(), headers=workspace_headers)
        self.assertEqual(job.status_code, 202, job.text)
        self.run_worker()
        history = self.client.get("/api/v1/jobs", headers=workspace_headers)
        self.assertEqual(history.json()[0]["name"], "Execução privada")
        self.assertEqual(history.json()[0]["status"], "succeeded")
        job_id = job.json()["id"]
        shared = self.client.post(f"/api/v1/jobs/{job_id}/shares", json={"allow_download": False, "expires_in_hours": 1}, headers=workspace_headers)
        self.assertEqual(shared.status_code, 201, shared.text)
        share = shared.json()
        share_headers = {"Authorization": "Bearer " + share["token"]}
        opened = self.client.get(f'/api/v1/shared/{share["id"]}', headers=share_headers)
        self.assertEqual(opened.status_code, 200, opened.text)
        pdf = next(item for item in opened.json()["artifacts"] if item["media_type"] == "application/pdf")
        self.assertEqual(self.client.get(f'/api/v1/shared/{share["id"]}/artifacts/{pdf["id"]}?download=true', headers=share_headers).status_code, 401)
        revoked = self.client.delete(f'/api/v1/shares/{share["id"]}', headers=workspace_headers)
        self.assertIsNotNone(revoked.json()["revoked_at"])
        self.assertEqual(self.client.get(f'/api/v1/shared/{share["id"]}', headers=share_headers).status_code, 410)

    def test_postman_matches_openapi_and_examples(self):
        from scripts.export_postman import build_collection

        collection = json.loads((ROOT / "postman/ocrsd.postman_collection.json").read_text())
        self.assertEqual(collection, build_collection())
        operations = set()
        schema = self.app.openapi()
        for folder in collection["item"]:
            for item in folder["item"]:
                request = item["request"]
                if isinstance(request["url"], str):
                    continue
                path = request["url"]["raw"].removeprefix("{{baseUrl}}").split("?")[0].replace("{{", "{").replace("}}", "}").replace("{pdf_id}", "{artifact_id}")
                method = request["method"].lower()
                operations.add((path, method))
                for response in item["response"]:
                    if response["header"][0]["value"] == "application/json":
                        definition = schema["paths"][path][method]["responses"][str(response["code"])]["content"]["application/json"]["schema"]
                        jsonschema.Draft202012Validator({**definition, "components": schema["components"]}).validate(json.loads(response["body"]))
        self.assertEqual(operations, {(path, method) for path, methods in schema["paths"].items() for method in methods})


if __name__ == "__main__":
    unittest.main()
