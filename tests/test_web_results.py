import os
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

try:
    import httpx2
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.models import GenerationOptions
from src.web.settings import Settings
from tests.test_web import ASGITestClient


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0, jobs_per_hour=40, max_pending=40,
                                 max_storage_bytes=12 * 1024**3)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        page = self.client.get("/resultados")
        self.cookie = re.search(r"ocrsd_workspace=([^;]+)", page.headers["set-cookie"]).group(1)
        self.owner = self.app.state.store.workspace(self.cookie)

    def request(self, method, path, **kwargs):
        return self.client.request(method, path, headers={"Cookie": f"ocrsd_workspace={self.cookie}", "HX-Request": "true"}, **kwargs)

    def submit(self, count):
        options = GenerationOptions(name=f"DNV {count}", document_type="dnv", mapping_id="mapeamento_seguindo_norma", fields=["1"], count=count)
        row, _ = self.app.state.store.submit(options, "results-test", self.owner)
        return row["id"]

    def run_worker(self):
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], env=environment,
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90, check=True)

    def test_jobs_receive_distinct_seeds_under_concurrency(self):
        from concurrent.futures import ThreadPoolExecutor

        store = self.app.state.store
        options = GenerationOptions(document_type="dnv", fields=["1"], count=1)
        with ThreadPoolExecutor(max_workers=8) as pool:
            rows = list(pool.map(lambda index: store.submit(options, f"seed-client-{index}", self.owner)[0], range(24)))
        seeds = [GenerationOptions.model_validate_json(row["options"]).seed for row in rows]
        self.assertTrue(all(isinstance(seed, int) and 0 <= seed < 2**32 for seed in seeds))
        self.assertEqual(len(set(seeds)), len(seeds))
        explicit, _ = store.submit(options.model_copy(update={"seed": seeds[0]}), "seed-client-explicit", self.owner)
        self.assertEqual(GenerationOptions.model_validate_json(explicit["options"]).seed, seeds[0])

    def test_exact_reproduction_reuses_seed_and_original_mapping(self):
        from src.web.services import Catalog, duplicate_model

        store = self.app.state.store
        catalog = Catalog(self.settings)
        model = duplicate_model(catalog, store, "dnv", self.owner)
        store.set_status(model["id"], self.owner, "published")
        options = catalog.validate(GenerationOptions(document_type=model["id"], fields=["1"], count=1), self.owner)
        row, _ = store.submit(options, "reproduction", self.owner, None, catalog.source(options, self.owner))
        original = store.get(row["id"])
        self.assertIsNotNone(store.source(original))
        definition = catalog.definition(model["id"], model["mapping_id"], self.owner)
        definition["fields"] = [{**field, "label": "Alterado depois"} if field["field_id"] == "1" else field for field in definition["fields"]]
        store.update_mapping(model["id"], self.owner, definition)
        response = self.request("POST", f'/resultados/{row["id"]}/reproduzir')
        self.assertEqual(response.status_code, 200, response.text)
        reproduced = store.get(response.headers["hx-redirect"].rsplit("/", 1)[1])
        self.assertNotEqual(reproduced["id"], original["id"])
        self.assertEqual(GenerationOptions.model_validate_json(reproduced["options"]).seed, GenerationOptions.model_validate_json(original["options"]).seed)
        self.assertEqual(reproduced["source_mapping"], original["source_mapping"])
        self.assertNotIn("Alterado depois", reproduced["source_mapping"])
        self.assertIn("reprodução", store.detail(reproduced).name)
        reused = self.request("POST", f'/resultados/{row["id"]}/reutilizar')
        self.assertIn("/geracao?draft=", reused.headers["hx-redirect"])
        self.run_worker()
        self.run_worker()
        first, second = (store.get(identifier) for identifier in (original["id"], reproduced["id"]))
        self.assertEqual((first["status"], second["status"]), ("succeeded", "succeeded"))
        truths = []
        for job in (first, second):
            artifact = next(item for item in store.artifacts(job) if item["name"] == "synthetic_0001.json")
            truths.append(json.loads(store.artifact(job, artifact["id"])[0].read_text()))
        self.assertEqual(truths[0]["fields"], truths[1]["fields"])
        self.assertEqual(truths[0]["fields"][0]["label"], next(field for field in json.loads(original["source_mapping"])["fields"] if field["field_id"] == "1")["label"])
        manifest = next(item for item in store.artifacts(first) if item["name"] == "dataset.json")
        self.assertEqual(json.loads(store.artifact(first, manifest["id"])[0].read_text())["seed"], GenerationOptions.model_validate_json(first["options"]).seed)

    def test_standard_models_are_protected_in_the_interface(self):
        gallery = self.request("GET", "/modelos")
        self.assertIn("Padrão", gallery.text)
        self.assertNotIn("Catálogo", gallery.text)
        self.assertIn("Duplicar e editar", gallery.text)
        editor = self.client.get("/modelos/dnv/editor", headers={"Cookie": f"ocrsd_workspace={self.cookie}"}, follow_redirects=False)
        self.assertEqual(editor.status_code, 303)
        saved = self.request("POST", "/modelos/dnv/mapeamento", data={"definition": "{}", "layout": "null"})
        self.assertEqual(saved.status_code, 403)
        copy = self.request("POST", "/modelos/dnv/copia")
        self.assertRegex(copy.headers["hx-redirect"], r"^/modelos/[0-9a-f]{32}/editor$")

    def test_single_document_view_metadata_download_and_reuse(self):
        job = self.submit(1)
        self.run_worker()
        page = self.request("GET", f"/resultados/{job}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Documento 001", page.text)
        self.assertIn("reader-stage", page.text)
        self.assertNotIn("JSON bruto", page.text)
        self.assertNotIn("Exemplo sintético", page.text)
        image = self.request("GET", f"/resultados/{job}/documentos/1/paginas/1.png")
        self.assertEqual(image.status_code, 200)
        self.assertTrue(image.content.startswith(b"\x89PNG"))
        info = self.request("GET", f"/resultados/{job}/documentos/1/informacoes")
        self.assertIn("Ver JSON bruto", info.text)
        self.assertIn("Baixar JSON", info.text)
        artifact = self.app.state.store.artifacts(self.app.state.store.get(job))
        pdf = next(item for item in artifact if item["media_type"] == "application/pdf")
        ground_truth = next(item for item in artifact if item["name"] == "synthetic_0001.json")
        self.assertEqual((pdf["document_number"], ground_truth["document_number"]), (1, 1))
        self.assertTrue(self.request("GET", f'/resultados/{job}/arquivos/{pdf["id"]}').content.startswith(b"%PDF"))
        self.assertIn(b"document_id", self.request("GET", f'/resultados/{job}/arquivos/{ground_truth["id"]}').content)
        other = self.client.get("/modelos")
        other_cookie = re.search(r"ocrsd_workspace=([^;]+)", other.headers["set-cookie"]).group(1)
        self.assertEqual(self.client.get(f'/resultados/{job}/arquivos/{pdf["id"]}', headers={"Cookie": f"ocrsd_workspace={other_cookie}"}).status_code, 404)
        self.assertEqual(self.client.get(f"/resultados/{job}/documentos/1/informacoes", headers={"Cookie": f"ocrsd_workspace={other_cookie}"}).status_code, 404)
        self.assertEqual(self.client.post(f"/resultados/{job}/renomear", data={"nome": "Inválido"}, headers={"Cookie": f"ocrsd_workspace={other_cookie}", "HX-Request": "true"}).status_code, 404)
        self.assertIn("Baixar dataset", self.request("GET", f"/resultados/{job}/exportar").text)
        reused = self.request("POST", f"/resultados/{job}/reutilizar", data={})
        self.assertEqual(reused.status_code, 200)
        draft = re.search(r"draft=([0-9a-f]+)", reused.headers["hx-redirect"]).group(1)
        state = self.app.state.store.wizard_draft(draft, self.owner)
        self.assertEqual((state["document_type"], state["count"], state["fields"]), ("dnv", 1, ["1"]))
        self.assertEqual(state["preset"], "original_snapshot")
        self.assertIn("Documento base", self.request("GET", reused.headers["hx-redirect"]).text)
        self.assertIn("Configuração original", self.request("GET", f"/geracao?draft={draft}&etapa=4").text)

    def test_selective_zip_export_and_validation(self):
        job = self.submit(2)
        self.run_worker()
        for query, expected in (("pdfs=true", {".pdf"}), ("jsons=true", {".json"}),
                                ("pdfs=true&jsons=true", {".pdf", ".json"})):
            response = self.request("GET", f"/resultados/{job}/exportar/arquivo?{query}")
            self.assertEqual(response.status_code, 200)
            with ZipFile(io.BytesIO(response.content)) as archive:
                names = archive.namelist()
                self.assertEqual({Path(name).suffix for name in names}, expected)
                self.assertEqual(sum(name.endswith(".pdf") for name in names), 2 if ".pdf" in expected else 0)
                self.assertEqual(sum(name.startswith("synthetic_") and name.endswith(".json") for name in names), 2 if ".json" in expected else 0)
                self.assertEqual("dataset.json" in names, len(expected) == 2)
        self.assertEqual(self.request("GET", f"/resultados/{job}/exportar/arquivo").status_code, 422)

    def test_partial_documents_are_indexed_and_authorized(self):
        job = self.submit(3)
        store = self.app.state.store
        store.claim()
        directory = store.directory(job)
        directory.mkdir()
        pdf = directory / "synthetic_0001.pdf"
        ground_truth = directory / "synthetic_0001.json"
        pdf.write_bytes((Path(__file__).resolve().parents[1] / "documents/dnv/template.pdf").read_bytes())
        ground_truth.write_text(json.dumps({"document_id": "synthetic_0001", "fields": []}))
        store.publish_document(job, 1, pdf, ground_truth)
        result = self.request("GET", f"/resultados/{job}")
        self.assertIn("Documento 001", result.text)
        self.assertIn('id="result-live-update"', result.text)
        self.assertEqual(self.request("GET", f"/resultados/{job}/documentos/1/paginas/1.png").status_code, 200)
        self.assertEqual(self.request("GET", f"/resultados/{job}/documentos/1/informacoes").status_code, 200)
        self.assertEqual(self.request("GET", f"/resultados/{job}/exportar/arquivo?pdfs=true").status_code, 409)
        pdf2 = directory / "synthetic_0002.pdf"
        json2 = directory / "synthetic_0002.json"
        pdf2.write_bytes(pdf.read_bytes())
        json2.write_bytes(ground_truth.read_bytes())
        store.publish_document(job, 2, pdf2, json2)
        update = self.request("GET", f"/resultados/{job}/documentos/atualizacoes?documento=1&quantidade=1")
        self.assertIn("Documento 002", update.text)
        self.assertNotIn("Documento 001", update.text)
        self.assertIn('hx-swap-oob="beforeend:#result-tree-list"', update.text)
        self.assertIn("&amp;quantidade=2", update.text)
        unchanged = self.request("GET", f"/resultados/{job}/documentos/atualizacoes?documento=1&quantidade=2")
        self.assertNotIn("result-tree-row", unchanged.text)
        activity = self.request("GET", "/geracao/atividade")
        self.assertIn(f'data-notice-key="job:{job}:running"', activity.text)
        self.assertIn("2 de 3 documentos prontos", activity.text)
        store.finish(job, "failed", "Falha controlada")
        self.assertIn("Documento 002", self.request("GET", f"/resultados/{job}").text)
        finished = self.request("GET", "/geracao/atividade")
        self.assertIn(f'data-notice-key="job:{job}:finished"', finished.text)
        self.assertIn('data-notice-lifetime="0"', finished.text)
        self.assertNotIn(":running", finished.text)
        other = self.client.get("/modelos")
        other_cookie = re.search(r"ocrsd_workspace=([^;]+)", other.headers["set-cookie"]).group(1)
        self.assertEqual(self.client.get(f"/resultados/{job}/documentos/1/informacoes", headers={"Cookie": f"ocrsd_workspace={other_cookie}"}).status_code, 404)

    def test_elapsed_time_is_processing_time(self):
        job = self.submit(1)
        shorter = self.submit(2)
        store = self.app.state.store
        store.claim()
        store.finish(job, "failed", "Falha controlada")
        store.claim()
        store.finish(shorter, "failed", "Falha controlada")
        with store.connection() as db:
            db.execute("UPDATE jobs SET started_at=?,finished_at=? WHERE id=?", (100.0, 165.0, job))
            db.execute("UPDATE jobs SET started_at=?,finished_at=? WHERE id=?", (100.0, 105.0, shorter))
        listed = self.request("GET", "/resultados/lista?ordem=tempo_desc")
        self.assertIn("01:05", listed.text)
        self.assertIn(">Tempo<", listed.text)
        self.assertIn(">Abrir<", listed.text)
        self.assertIn(">Detalhes<", listed.text)
        self.assertLess(listed.text.index("DNV 1"), listed.text.index("DNV 2"))
        ascending = self.request("GET", "/resultados/lista?ordem=tempo_asc")
        self.assertLess(ascending.text.index("DNV 2"), ascending.text.index("DNV 1"))

    def test_multiple_documents_navigation_and_filters(self):
        job = self.submit(3)
        self.run_worker()
        page = self.request("GET", f"/resultados/{job}")
        self.assertEqual(page.text.count('class="result-tree-row'), 3)
        second = self.request("GET", f"/resultados/{job}/documentos/2")
        self.assertEqual(second.status_code, 200)
        self.assertIn("2 de 3", second.text)
        self.assertIn(f"/resultados/{job}/documentos/2/paginas/1.png", second.text)
        self.assertIn('aria-current="page"', second.text)
        self.assertEqual(self.request("GET", f"/resultados/{job}?documento=2").status_code, 200)
        self.assertEqual(self.request("GET", f"/resultados/{job}/documentos/4").status_code, 404)
        listed = self.request("GET", "/resultados/lista?busca=DNV&status=succeeded&modelo=dnv&ordem=antigos")
        self.assertIn("DNV 3", listed.text)
        self.assertEqual(listed.headers["hx-push-url"], "/resultados?busca=DNV&status=succeeded&modelo=dnv&ordem=antigos")
        self.assertIn("Nenhuma geração encontrada", self.request("GET", "/resultados/lista?busca=inexistente").text)

    def test_rename_sort_status_and_actions(self):
        completed = self.submit(1)
        self.run_worker()
        queued = self.submit(2)
        renamed = self.request("POST", f"/resultados/{completed}/renomear", data={"nome": "Lote revisado"})
        self.assertEqual(renamed.status_code, 200)
        self.assertIn("Lote revisado", renamed.text)
        self.assertEqual(self.app.state.store.detail(self.app.state.store.get(completed)).options.name, "DNV 1")
        self.assertEqual(self.request("POST", f"/resultados/{completed}/renomear", data={"nome": " "}).status_code, 422)
        listed = self.request("GET", "/resultados/lista?ordem=nome_desc&status=succeeded,queued")
        self.assertIn("Lote revisado", listed.text)
        self.assertIn("Baixar", listed.text)
        self.assertIn("Compartilhar", listed.text)
        self.assertLess(listed.text.index("Lote revisado"), listed.text.index("DNV 2"))
        self.assertIn(f'/resultados/{completed}/exportar', listed.text)
        self.assertNotIn(f'/resultados/{queued}/exportar', listed.text)
        self.assertIn("DNV 2", self.request("GET", "/resultados/lista?status=queued").text)
        self.assertNotIn("Lote revisado", self.request("GET", "/resultados/lista?status=queued").text)

    def test_pagination_for_many_jobs(self):
        for _ in range(27):
            self.submit(1)
        first = self.request("GET", "/resultados/lista")
        self.assertIn("27 datasets · página 1 de 2", first.text)
        self.assertEqual(first.text.count('class="result-name-cell"'), 25)
        second = self.request("GET", "/resultados/lista?pagina=2")
        self.assertEqual(second.text.count('class="result-name-cell"'), 2)
        self.assertEqual(second.headers["hx-push-url"], "/resultados?pagina=2")

    def test_running_failed_and_private_access(self):
        job = self.submit(1)
        self.assertIn("1 documentos", self.request("GET", f"/resultados/{job}").text)
        self.app.state.store.finish(job, "failed", "/private/secret.txt")
        failed = self.request("GET", f"/resultados/{job}")
        self.assertIn("Não foi possível concluir a geração", failed.text)
        self.assertIn("Detalhes técnicos", failed.text)
        self.assertNotIn("/private/secret.txt", failed.text)
        self.assertIn("Reutilizar configuração", failed.text)
        other = self.client.get("/modelos")
        other_cookie = re.search(r"ocrsd_workspace=([^;]+)", other.headers["set-cookie"]).group(1)
        self.assertEqual(self.client.get(f"/resultados/{job}", headers={"Cookie": f"ocrsd_workspace={other_cookie}"}).status_code, 404)
        self.assertEqual(self.client.get(f"/resultados/{job}/documentos/1/paginas/1.png", headers={"Cookie": f"ocrsd_workspace={other_cookie}"}).status_code, 404)
