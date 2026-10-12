import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx2
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.models import GenerationOptions, Writing
from src.web.services import Catalog
from src.web.settings import Settings
from tests.test_web import ASGITestClient


class WizardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        page = self.client.get("/geracao?modelo=dnv")
        self.cookie = re.search(r"ocrsd_workspace=([^;]+)", page.headers["set-cookie"]).group(1)
        self.draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', page.text).group(1)
        self.mapping = "mapeamento_seguindo_norma"
        self.assertNotIn('name="mapping_id"', page.text)

    def request(self, method, path, **kwargs):
        return self.client.request(method, path, headers={"Cookie": f"ocrsd_workspace={self.cookie}", "HX-Request": "true"}, **kwargs)

    def advance(self, step, **data):
        return self.request("POST", f"/geracao/etapa/{step}", data={"draft_id": self.draft, **data})

    def test_complete_generation_and_export(self):
        self.assertEqual(self.advance(1, document_type="dnv", mapping_id=self.mapping).status_code, 200)
        page = self.request("GET", f"/geracao?draft={self.draft}&etapa=2")
        self.assertIn('data-overlay-field=', page.text)
        fields = re.findall(r'name="fields" value="([^"]+)"', page.text)
        self.assertTrue(fields)
        self.assertEqual(self.advance(2, fields=fields[:2]).status_code, 200)
        self.assertEqual(self.advance(3, count="0").status_code, 422)
        self.assertEqual(self.advance(3, count="1").status_code, 200)
        refreshed = self.request("GET", f"/geracao?draft={self.draft}&etapa=3")
        self.assertIn('value="1"', refreshed.text)
        self.assertEqual(self.advance(4, preset="padrao_fonte_natural").status_code, 200)
        self.assertEqual(self.advance(5, filter_preset="none").status_code, 200)
        review = self.request("GET", f"/geracao?draft={self.draft}&etapa=6")
        self.assertIn("Gerar dataset", review.text)
        submitted = self.request("POST", "/geracao/gerar", data={"draft_id": self.draft})
        self.assertEqual(submitted.headers.get("hx-redirect"), "/modelos")
        self.assertEqual(self.request("POST", "/geracao/gerar", data={"draft_id": self.draft}).status_code, 409)
        activity = self.request("GET", "/geracao/atividade")
        self.assertIn("Gerando documentos", activity.text)
        owner = self.app.state.store.workspace(self.cookie)
        job = self.app.state.store.list_jobs(owner)[0]
        self.assertIsInstance(job["options"]["seed"], int)
        self.assertEqual(job["options"]["mapping_id"], "mapeamento_seguindo_norma")
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], env=environment,
                       cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90, check=True)
        completed = self.app.state.store.list_jobs(owner)[0]
        self.assertEqual(completed["status"], "succeeded")
        result = self.request("GET", f'/resultados/{job["id"]}')
        self.assertIn("Documento 001", result.text)
        self.assertIn("Baixar ZIP", self.request("GET", f'/resultados/{job["id"]}/exportar').text)
        archive = next(item for item in self.app.state.store.artifacts(self.app.state.store.get(job["id"])) if item["media_type"] == "application/zip")
        download = self.request("GET", f'/resultados/{job["id"]}/arquivos/{archive["id"]}?nome=Teste%20DNV')
        self.assertIn("Teste%20DNV.zip", download.headers["content-disposition"])

    def test_incompatible_writing_style_is_disabled_and_rejected(self):
        owner = self.app.state.store.workspace(self.cookie)
        writing = Writing(model="aaron1259", text_policy="strict").model_dump()
        preset = self.app.state.store.save_preset(owner, "Neural estrito", writing, "none", False, None, "")
        with patch.object(Catalog, "models", return_value=[{"id": "font", "available": True}, {"id": "aaron1259", "available": True}]):
            self.assertEqual(self.advance(1, document_type="dnv", mapping_id="mapeamento_seguindo_norma").status_code, 200)
            self.assertEqual(self.advance(2, fields=["1", "4"]).status_code, 200)
            self.assertEqual(self.advance(3, count="1").status_code, 200)
            page = self.request("GET", f"/geracao?draft={self.draft}&etapa=4")
            card = re.search(rf'<label class="wizard-style is-incompatible"[^>]*><input type="radio" name="preset" value="{preset["id"]}"[^>]*>', page.text)
            self.assertIsNotNone(card)
            self.assertIn("disabled", card.group(0))
            rejected = self.advance(4, preset=preset["id"])
            self.assertEqual(rejected.status_code, 422)
            self.assertIn("Normalizar", rejected.text)
            self.assertEqual(self.advance(2, fields=["4"]).status_code, 200)
            self.assertEqual(self.advance(4, preset=preset["id"]).status_code, 200)

    def test_isolated_drafts_and_failed_job(self):
        second = self.request("GET", "/geracao?modelo=certidao_nascimento")
        other = re.search(r'name="draft_id" value="([0-9a-f]+)"', second.text).group(1)
        self.assertNotEqual(self.draft, other)
        self.assertEqual(self.advance(1, document_type="dnv", mapping_id=self.mapping).status_code, 200)
        self.assertIn("CERTIDAO NASCIMENTO", self.request("GET", f"/geracao?draft={other}&etapa=1").text)
        fields = re.findall(r'name="fields" value="([^"]+)"', self.request("GET", f"/geracao?draft={self.draft}&etapa=2").text)
        self.assertEqual(self.advance(2, direction="back", fields=fields[:1]).status_code, 200)
        self.assertIn(f'name="fields" value="{fields[0]}" checked', self.request("GET", f"/geracao?draft={self.draft}&etapa=2").text)
        self.assertEqual(self.advance(2).status_code, 422)
        owner = self.app.state.store.workspace(self.cookie)
        self.assertEqual(self.request("GET", "/geracao?draft=bad&etapa=2").status_code, 404)
        self.assertIn("não possui mapeamento disponível", self.request("GET", "/geracao?modelo=sem_mapa").text)
        self.assertEqual(self.app.state.store.list_jobs(owner), [])
        options = GenerationOptions(document_type="dnv", mapping_id=self.mapping, fields=["campo_teste"], count=1)
        row, _ = self.app.state.store.submit(options, "wizard-test", owner)
        self.app.state.store.finish(row["id"], "failed", "Falha controlada")
        self.assertIn("Falha na geração", self.request("GET", "/geracao/atividade").text)
        self.assertIn("Não foi possível concluir a geração", self.request("GET", f'/resultados/{row["id"]}').text)

    def test_dnv_composite_field_overlays_restore_selected_state(self):
        self.assertEqual(self.advance(1, document_type="dnv", mapping_id="mapeamento_seguindo_norma").status_code, 200)
        page = self.request("GET", f"/geracao?draft={self.draft}&etapa=2")
        self.assertGreater(page.text.count('data-overlay-field="2"'), 1)
        self.assertIn('data-field-id="2"', page.text)
        self.assertEqual(self.advance(2, direction="back", fields=["2"]).status_code, 200)
        restored = self.request("GET", f"/geracao?draft={self.draft}&etapa=2")
        self.assertIn('name="fields" value="2" checked', restored.text)
        self.assertIn('class="wizard-box is-selected" data-overlay-field="2"', restored.text)

    def test_review_disables_invalid_mapping_configuration(self):
        self.advance(1, document_type="dnv")
        self.advance(2, fields=["1"])
        self.advance(3, count="1")
        self.advance(4, preset="padrao_fonte_natural")
        self.advance(5, filter_preset="none")
        owner = self.app.state.store.workspace(self.cookie)
        state = self.app.state.store.wizard_draft(self.draft, owner)
        self.app.state.store.save_wizard_draft(self.draft, owner, {**state, "fields": ["campo_removido"]})
        review = self.request("GET", f"/geracao?draft={self.draft}&etapa=6")
        self.assertIn("Selecione campos existentes", review.text)
        self.assertRegex(review.text, r'<button class="btn btn-primary" type="submit" disabled>Gerar dataset')
        self.assertEqual(self.request("POST", "/geracao/gerar", data={"draft_id": self.draft}).status_code, 422)

    def test_preset_gallery_returns_to_same_draft(self):
        self.advance(1, document_type="dnv", mapping_id=self.mapping)
        fields = re.findall(r'name="fields" value="([^"]+)"', self.request("GET", f"/geracao?draft={self.draft}&etapa=2").text)
        self.advance(2, fields=fields[:1])
        self.advance(3, count="1")
        outbound = self.request("GET", f"/geracao/editar-escrita?draft={self.draft}")
        self.assertEqual(outbound.headers["location"], f"/presets/escrita?retorno=geracao_{self.draft}")
        gallery = self.request("GET", outbound.headers["location"])
        self.assertIn(f"/geracao?draft={self.draft}&amp;preset=padrao_fonte_natural", gallery.text)
        returned = self.request("GET", f"/geracao?draft={self.draft}&preset=padrao_fonte_natural")
        self.assertIn("Escrita", returned.text)
        self.assertIn('name="preset" value="padrao_fonte_natural" checked', returned.text)

    def test_mapping_editor_preserves_draft(self):
        opened = self.request("POST", "/geracao/editar-mapeamento", data={"draft_id": self.draft})
        self.assertEqual(opened.status_code, 200)
        self.assertIn(f"?draft={self.draft}", opened.headers["hx-redirect"])
        editor = self.request("GET", opened.headers["hx-redirect"])
        self.assertIn(f"/geracao?draft={self.draft}&amp;etapa=1", editor.text)
        owner = self.app.state.store.workspace(self.cookie)
        state = self.app.state.store.wizard_draft(self.draft, owner)
        self.assertNotEqual(state["document_type"], "dnv")
        self.assertEqual(state["step"], 1)
        self.assertIn("ainda é um rascunho", self.request("GET", f"/geracao?draft={self.draft}").text)

    def test_mapping_preview_and_field_workspace(self):
        initial = self.request("GET", f"/geracao?draft={self.draft}")
        self.assertIn("Duplicar e editar", initial.text)
        self.assertIn("Próximo", initial.text)
        self.assertNotIn("test_w_datetime", initial.text)
        preview = self.request("GET", f"/geracao/mapeamento/previa?draft={self.draft}")
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.text.count('class="wizard-box"'), sum(max(1, len(field.get("elements") or [])) for field in
                         __import__("json").loads((Path(__file__).resolve().parents[1] / "documents/dnv/mappings/mapeamento_seguindo_norma.json").read_text())["fields"]))
        owner = self.app.state.store.workspace(self.cookie)
        self.assertEqual(self.app.state.store.wizard_draft(self.draft, owner)["mapping_id"], self.mapping)
        self.assertEqual(self.advance(1, document_type="dnv", mapping_id="test_w_datetime").status_code, 200)
        self.assertEqual(self.app.state.store.wizard_draft(self.draft, owner)["mapping_id"], self.mapping)
        fields = self.request("GET", f"/geracao?draft={self.draft}&etapa=2")
        self.assertIn("wizard-layout-fields", fields.text)
        self.assertIn("Identificação do recém-nascido", fields.text)
        self.assertIn("data-field-group-checkbox", fields.text)
        self.assertIn("data-selected-only", fields.text)
        self.assertIn('data-pdf-action="fit-width"', fields.text)
        self.assertIn('data-pdf-action="center"', fields.text)
