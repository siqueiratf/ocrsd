import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote

try:
    import httpx2  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.models import Writing
from src.web.settings import Settings
from tests.test_web import ASGITestClient

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf"


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0, neural_enabled=False)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        self.cookie = self.workspace()

    def workspace(self):
        response = self.client.get("/modelos")
        self.assertEqual(response.status_code, 200, response.text)
        return re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)

    def page(self, path, cookie=None):
        return self.client.get(path, headers={"Cookie": f"ocrsd_workspace={cookie or self.cookie}"})

    def hx(self, method, path, cookie=None, **kwargs):
        headers = {"HX-Request": "true", "Cookie": f"ocrsd_workspace={cookie or self.cookie}", **kwargs.pop("headers", {})}
        return self.client.request(method, path, headers=headers, **kwargs)

    def generation_form(self, **changes):
        form = {"name": "Dataset pela interface", "document_type": "certidao_nascimento", "mapping_id": "certidao_nascimento_fields",
                "fields": ["cpf", "matricula", "sexo"], "count": "1", "seed": "42", "model": "font",
                "handwriting": "on", "text_handwriting": "on", "intensity": "0.65", "writer": "", "bias": "0.75",
                "color": "#193787", "width": "1.5", "pressure": "0.25", "dropout": "0", "text_policy": "normalize",
                "filter_preset": "none"}
        form.update(changes)
        return form

    def run_worker(self):
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items() if value is not None},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT, env=environment,
                       capture_output=True, text=True, timeout=90, check=True)

    def test_gallery_deletes_only_custom_models_and_wizard_shows_description(self):
        gallery = self.page("/modelos").text
        self.assertIn("Declaração de Nascido Vivo (DNV)", gallery)
        self.assertNotIn("/modelos/dnv/excluir", gallery)
        wizard = self.page("/geracao?modelo=dnv").text
        self.assertIn("Formulário oficial do SINASC preenchido com dados fictícios", wizard)

        created = self.hx("POST", "/modelos/em-branco", data={"label": "Modelo temporário", "description": "Para excluir",
                                                            "size": "a4", "orientation": "portrait"})
        self.assertEqual(created.status_code, 200, created.text)
        model_id = re.search(r"/modelos/([0-9a-f]{32})/editor", created.headers["hx-redirect"]).group(1)
        self.assertIn(f"/modelos/{model_id}/excluir", self.page("/modelos").text)
        deleted = self.hx("POST", f"/modelos/{model_id}/excluir")
        self.assertEqual(deleted.headers.get("hx-redirect"), "/modelos")
        self.assertNotIn(f"/modelos/{model_id}/excluir", self.page("/modelos").text)
        self.assertEqual(self.hx("POST", "/modelos/dnv/excluir").status_code, 404)

    def test_unsupported_do_cause_is_visible_but_cannot_be_generated(self):
        first = self.page("/geracao?modelo=do").text
        draft = re.search(r'name="draft_id" value="([0-9a-f]+)"', first).group(1)
        self.hx("POST", "/geracao/etapa/1", data={"draft_id": draft, "document_type": "do"})
        page = self.page(f"/geracao?draft={draft}&etapa=2").text
        self.assertIn('value="40_a_causa"', page)
        self.assertIn('name="fields" value="40_a_causa" disabled', page)
        response = self.client.post("/api/v1/generation/validate", json={"document_type": "do", "fields": ["40_a_causa"]})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "unsupported_field")

    def test_pages_session_security_and_assets(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual((response.status_code, response.headers["location"]), (303, "/modelos"))
        first = self.client.get("/modelos")
        self.assertIn("HttpOnly", first.headers["set-cookie"])
        self.assertIn("SameSite=lax", first.headers["set-cookie"])
        self.assertIn("default-src 'self'", first.headers["content-security-policy"])
        self.assertEqual(first.headers["x-frame-options"], "SAMEORIGIN")
        self.assertIn("Galeria de Modelos", first.text)
        self.assertIn('href="/geracao?modelo=certidao_nascimento"', first.text)
        self.assertNotIn("set-cookie", self.page("/modelos").headers)
        for path, heading in (("/geracao", "Nova geração de dataset"), ("/resultados", "Resultados"),
                              ("/presets/escrita", "Presets de Escrita")):
            response = self.page(path)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn(heading, response.text)
            self.assertRegex(response.text, r'aria-current="page"|<summary class="is-current">')
        for asset in ("htmx.min.js", "app.js", "editor.js", "styles.css", "favicon.svg"):
            self.assertEqual(self.client.get(f"/static/{asset}").status_code, 200)
        missing = self.page("/inexistente")
        self.assertEqual(missing.status_code, 404)
        self.assertIn("Página ou recurso não encontrado", missing.text)
        unauthenticated = self.client.post("/geracao/resumo", headers={"HX-Request": "true"}, data={})
        self.assertEqual(unauthenticated.status_code, 401)
        self.assertEqual(unauthenticated.headers["hx-retarget"], "#avisos")
        self.assertIn("toast-erro", unauthenticated.text)
        forged = self.client.post("/geracao/jobs", headers={"Cookie": f"ocrsd_workspace={self.cookie}"}, data=self.generation_form())
        self.assertEqual(forged.status_code, 403)
        self.assertEqual(self.hx("POST", "/geracao/jobs", json={}).status_code, 415)
        paths = self.app.openapi()["paths"]
        self.assertTrue(all(path.startswith("/api/") for path in paths))
        self.assertNotIn("content-security-policy", self.client.get("/api/v1/health").headers)

    def test_branding_logo_favicon_and_asset_versions(self):
        page = self.page("/modelos").text
        for name in ("favicon.ico", "favicon.svg", "favicon-32.png", "favicon-16.png", "apple-touch-icon.png", "manifest.webmanifest", "labrec-logo.svg", "ocrsd-logo.svg", "styles.css"):
            self.assertRegex(page, rf"/static/{re.escape(name)}\?v=[0-9a-f]{{12}}")
        labrec = page.index('class="brand-logo brand-labrec" src="/static/labrec-logo.svg?v=')
        divider = page.index('class="brand-divider" aria-hidden="true"')
        ocrsd = page.index('class="brand-logo brand-ocrsd" src="/static/ocrsd-logo.svg?v=')
        self.assertLess(labrec, divider)
        self.assertLess(divider, ocrsd)
        self.assertNotIn("labrec-logo", page[page.index("<head>"):page.index("</head>")])
        self.assertEqual(page.count('rel="icon"'), 4)
        versioned = self.client.get(re.search(r'/static/favicon\.svg\?v=[0-9a-f]+', page).group(0))
        self.assertEqual(versioned.headers["cache-control"], "public, max-age=31536000, immutable")
        self.assertEqual(self.client.get("/static/favicon.svg").headers["cache-control"], "no-cache")
        icon = self.client.get("/favicon.ico")
        self.assertEqual((icon.status_code, icon.headers["content-type"]), (200, "image/x-icon"))
        self.assertTrue(icon.content.startswith(b"\x00\x00\x01\x00\x03\x00"))
        self.assertEqual(self.client.get("/apple-touch-icon.png").headers["content-type"], "image/png")
        self.assertEqual(json.loads(self.client.get("/static/manifest.webmanifest").text)["icons"][1]["sizes"], "192x192")
        static = ROOT / "src/web/static"
        self.assertEqual((static / "ocrsd-logo.svg").read_bytes(), (ROOT / "ocrsd-logo.svg").read_bytes())
        favicon = (static / "favicon.svg").read_text()
        starts = [float(value) for value in re.findall(r'<path[^>]*\bd="M\s*(-?[\d.]+)', favicon)]
        self.assertEqual(len(starts), 19)
        self.assertTrue(all(170 <= value < 660 for value in starts))
        logo = (ROOT / "ocrsd-logo.svg").read_text()
        for path in re.findall(r'<path[^>]*\bd="M\s*-?[\d.]+[^>]*>', favicon):
            self.assertIn(path.split(' d="')[1].split('"')[0], logo)

    def test_legacy_workspace_token_is_adopted(self):
        legacy = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        adopted = self.hx("POST", "/sessao", data={"token": legacy})
        self.assertEqual(adopted.status_code, 204)
        self.assertEqual(adopted.headers["hx-refresh"], "true")
        self.assertIn(f"ocrsd_workspace={legacy}", adopted.headers["set-cookie"])
        self.assertEqual(self.hx("POST", "/sessao", data={"token": "invalido"}).status_code, 401)

    def test_generation_results_and_revocable_sharing(self):
        page = self.page("/geracao?modelo=certidao_nascimento")
        self.assertIn("Documento base", page.text)
        self.assertNotIn('name="mapping_id"', page.text)
        sections = self.hx("POST", "/geracao/formulario", data={**self.generation_form(), "acao": "documento", "document_type": "dnv"})
        self.assertIn('id="previa" hx-swap-oob="innerHTML"', sections.text)
        self.assertIn("/modelos/dnv/previa.png", sections.text)
        self.assertEqual(sections.text.count('name="fields"'), sections.text.count("checked") - 3)
        cleared = self.hx("POST", "/geracao/campos", data={**self.generation_form(), "acao": "limpar"})
        self.assertIn("0 selecionados", cleared.text)
        filtered = self.hx("POST", "/geracao/campos", data={**self.generation_form(), "busca": "cpf"})
        self.assertEqual(filtered.text.count("<label class=\"check\" hidden>"), filtered.text.count('name="fields"') - 1)
        summary = self.hx("POST", "/geracao/resumo", data=self.generation_form(filter_preset="strong"))
        self.assertIn("Realismo: Forte", summary.text)
        self.assertIn('id="contagem-campos" hx-swap-oob="true">3 selecionados', summary.text)
        invalid = self.hx("POST", "/geracao/jobs", data=self.generation_form(count="abc"))
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("Quantidade de documentos", invalid.text)
        self.assertIn("ao menos um campo", self.hx("POST", "/geracao/jobs", data=self.generation_form(fields=[])).text)
        self.settings.max_count = 10
        self.assertIn("Limite de 10", self.hx("POST", "/geracao/jobs", data=self.generation_form(count="11")).text)
        self.settings.max_count = None

        queued = self.hx("POST", "/geracao/jobs", data=self.generation_form())
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = re.search(r'hx-get="/geracao/jobs/([0-9a-f]{32})"', queued.text).group(1)
        self.assertIn('hx-trigger="every 1500ms"', queued.text)
        self.assertRegex(queued.text, r'id="enviar"[^>]*disabled[^>]*hx-swap-oob="true"')
        self.run_worker()
        finished = self.hx("GET", f"/geracao/jobs/{job_id}")
        self.assertIn(f'href="/resultados?job={job_id}"', finished.text)
        self.assertNotIn("hx-trigger", finished.text)

        results = self.page(f"/resultados/{job_id}")
        self.assertIn('result-tree-row is-active', results.text)
        self.assertIn('class="reader-stage"', results.text)
        artifacts = self.app.state.store.artifacts(self.app.state.store.get(job_id))
        pdf = next(item for item in artifacts if item["media_type"] == "application/pdf")
        json_file = next(item for item in artifacts if item["name"] == "synthetic_0001.json")
        archive = next(item for item in artifacts if item["media_type"] == "application/zip")
        self.assertIn("Baixar", results.text)
        self.assertIn(f'/resultados/{job_id}/exportar/arquivo', self.hx("GET", f"/resultados/{job_id}/exportar").text)
        inline = self.page(f'/resultados/{job_id}/arquivos/{pdf["id"]}?download=false')
        self.assertTrue(inline.content.startswith(b"%PDF"))
        self.assertIn("inline", inline.headers["content-disposition"])
        self.assertNotIn("content-security-policy", inline.headers)
        viewer = self.hx("GET", f'/resultados/{job_id}/documentos/1/informacoes')
        self.assertIn("Ver JSON bruto", viewer.text)
        self.assertIn("Baixar JSON", viewer.text)
        self.assertIn(f'/resultados/{job_id}/arquivos/{json_file["id"]}', viewer.text)
        self.assertIn("Nenhuma geração encontrada", self.hx("GET", "/resultados/lista?busca=inexistente").text)
        self.assertIn("Dataset pela interface", self.hx("GET", "/resultados/lista?status=succeeded").text)
        detail = self.hx("GET", f"/resultados/{job_id}?busca=&status=")
        self.assertIn('class="result-toolbar"', detail.text)
        self.assertIn('id="result-workspace"', detail.text)
        other = self.workspace()
        self.assertEqual(self.hx("GET", f"/resultados/{job_id}", cookie=other).status_code, 404)
        self.assertNotIn(job_id, self.page("/resultados", cookie=other).text)

        reused = self.page(f"/geracao?job={job_id}")
        self.assertIn("Certidão de Nascimento", reused.text)
        reused_id = re.search(r'name="draft_id" value="([0-9a-f]+)"', reused.text).group(1)
        reused_state = self.app.state.store.wizard_draft(reused_id, self.app.state.store.workspace(self.cookie))
        self.assertEqual(reused_state["fields"], ["cpf", "matricula", "sexo"])

        dialog = self.hx("GET", f"/resultados/{job_id}/compartilhar")
        self.assertIn("Compartilhar resultado", dialog.text)
        created = self.hx("POST", f"/resultados/{job_id}/compartilhamentos", data={"expires_in_hours": "2"})
        self.assertEqual(created.status_code, 200, created.text)
        share_id, token = re.search(r"/compartilhado/([0-9a-f]{32})#token=([^\"]+)", created.text).groups()
        token = unquote(token)
        self.assertIn('id="compartilhamentos" hx-swap-oob="innerHTML"', created.text)
        self.assertIn("somente visualização", created.text)
        self.assertEqual(self.hx("POST", f"/resultados/{job_id}/compartilhamentos", data={"expires_in_hours": "0"}).status_code, 422)
        pending = self.client.get(f"/compartilhado/{share_id}")
        self.assertIn(f'data-compartilhamento="{share_id}"', pending.text)
        self.assertEqual(self.client.post(f"/compartilhado/{share_id}/acesso", headers={"HX-Request": "true"}, data={"token": "x"}).status_code, 404)
        access = self.client.post(f"/compartilhado/{share_id}/acesso", headers={"HX-Request": "true"}, data={"token": token})
        self.assertEqual(access.status_code, 204)
        self.assertIn(f"Path=/compartilhado/{share_id}", access.headers["set-cookie"])
        share_cookie = {"Cookie": f"ocrsd_share={token}"}
        shared = self.client.get(f"/compartilhado/{share_id}", headers=share_cookie)
        self.assertIn("Dataset pela interface", shared.text)
        self.assertIn(f'/compartilhado/{share_id}/arquivos/{pdf["id"]}?download=false', shared.text)
        self.assertNotIn("Baixar dataset", shared.text)
        self.assertTrue(self.client.get(f'/compartilhado/{share_id}/arquivos/{pdf["id"]}', headers=share_cookie).content.startswith(b"%PDF"))
        self.assertEqual(self.client.get(f'/compartilhado/{share_id}/arquivos/{archive["id"]}?download=true', headers=share_cookie).status_code, 401)
        revoked = self.hx("POST", f"/compartilhamentos/{share_id}/revogar")
        self.assertIn("Revogado", revoked.text)
        self.assertEqual(self.client.get(f"/compartilhado/{share_id}", headers=share_cookie).status_code, 410)

    def editor_data(self, model_id):
        editor = self.page(f"/modelos/{model_id}/editor")
        self.assertEqual(editor.status_code, 200, editor.text)
        return json.loads(re.search(r'<script type="application/json" id="dados-editor">(.*?)</script>', editor.text, re.S).group(1))

    def save(self, model_id, data, fields, layout=None, **extra):
        form = {"definition": json.dumps({**data["definition"], "fields": fields}), "revision": data["revision"],
                "layout": json.dumps(layout) if layout is not None else "", **extra}
        return self.hx("POST", f"/modelos/{model_id}/mapeamento", data=form)

    def test_editor_exposes_and_keeps_semantic_bindings(self):
        response = self.hx("POST", "/modelos", data={"label": "Semântico", "mapping_id": "default"},
                           files={"pdf": ("template.pdf", TEMPLATE.read_bytes(), "application/pdf")})
        model_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", response.headers["hx-redirect"]).group(1)
        data = self.editor_data(model_id)
        self.assertEqual(data["semantic_options"][0]["key"], "newborn.name")
        self.assertEqual(set(data["semantic_options"][0]), {"key", "label", "group", "kind", "description"})
        fields = [{"field_id": "mae", "label": "Nome da mãe", "data_type": "person_name", "bbox": [100, 100, 600, 160]},
                  {"field_id": "livre", "label": "Livre", "bbox": [100, 200, 600, 260], "semantic": {"key": "birth.date", "part": "year"}}]
        self.assertEqual(self.save(model_id, data, fields).status_code, 200)
        stored = self.hx("GET", f"/modelos/{model_id}/estado.json").json()
        self.assertEqual(stored["definition"]["fields"][1]["semantic"], {"key": "birth.date", "part": "year"})
        self.assertEqual(stored["semantic_inferred"], {"mae": "mother.name", "livre": None})
        rejected = self.save(model_id, {**data, "revision": stored["revision"]}, [{**fields[1], "semantic": "birth.nada"}])
        self.assertEqual(rejected.status_code, 422)
        self.assertIn("Chave semântica desconhecida", rejected.text)
        self.assertEqual(self.hx("GET", "/modelos/dnv/estado.json").json()["semantic_inferred"]["19"], "mother.age")

    def test_model_upload_editor_publication_and_catalog_copy(self):
        response = self.hx("POST", "/modelos", data={"label": "Modelo enviado", "description": "Pela interface", "mapping_id": "default"},
                           files={"pdf": ("template.pdf", TEMPLATE.read_bytes(), "application/pdf")})
        self.assertEqual(response.status_code, 200, response.text)
        model_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", response.headers["hx-redirect"]).group(1)
        rejected = self.hx("POST", "/modelos", data={"label": "Sem PDF", "mapping_id": "default"},
                           files={"pdf": ("notas.txt", b"texto", "text/plain")})
        self.assertEqual(rejected.status_code, 422)
        data = self.editor_data(model_id)
        self.assertEqual(data["definition"]["fields"], [])
        self.assertIsNone(data["layout"])
        self.assertTrue(data["preview"].startswith(f"/modelos/{model_id}/previa.png?largura=2400&v="))
        self.assertEqual(self.page(data["preview"]).headers["content-type"], "image/png")
        self.assertIn("digit_sequence", data["type_labels"])
        self.assertIn("municipality", data["type_labels"])
        self.assertEqual({group["id"] for group in data["palette"]}, {"basicos", "pessoais", "geograficos", "estruturados", "personalizados"})
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 422)
        self.assertNotIn(f'value="{model_id}"', self.page("/geracao").text)
        self.assertIn("ainda é um rascunho", self.page(f"/geracao?modelo={model_id}").text)

        fields = [{"field_id": "nome", "label": "Nome", "data_type": "person_name", "bloco": "Registrado", "official_variable": "NOME",
                   "bbox": [100, 100, 600, 160]},
                  {"field_id": "sexo", "label": "Sexo", "data_type": "checkbox", "selection_mode": "single", "bbox": [100, 200, 600, 260],
                   "elements": [{"element_id": "sexo_0", "label": "F", "value": "F", "bbox": [110, 210, 150, 250]},
                                {"element_id": "sexo_1", "label": "M", "value": "M", "bbox": [300, 210, 340, 250]}]}]
        saved = self.save(model_id, data, fields, label="Modelo revisado", description="Mapeado")
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertIn("Mapeamento salvo", saved.text)
        self.assertIn('id="publicacao" class="publication" hx-swap-oob="true"', saved.text)
        new_revision = json.loads(saved.headers["hx-trigger"])["editor-salvo"]["revision"]
        self.assertNotEqual(new_revision, data["revision"])
        stored = self.hx("GET", f"/modelos/{model_id}/estado.json").json()
        self.assertEqual(stored["revision"], new_revision)
        self.assertEqual(stored["definition"]["fields"][0]["official_variable"], "NOME")
        self.assertEqual(stored["definition"]["fields"][1]["elements"][1]["value"], "M")
        conflict = self.save(model_id, data, fields)
        self.assertEqual(conflict.status_code, 409)
        self.assertIn("outra janela", conflict.text)
        invalid_rules = [{**fields[0], "validation": {"min_value": 9, "max_value": 1}}]
        self.assertEqual(self.save(model_id, {**data, "revision": new_revision}, invalid_rules).status_code, 422)
        outside = [{"field_id": "nome", "bbox": [0, 0, 999999, 10]}]
        self.assertEqual(self.save(model_id, {**data, "revision": new_revision}, outside).status_code, 422)

        preview = self.hx("POST", f"/modelos/{model_id}/previa", data={"definition": json.dumps({**data["definition"], "fields": fields}), "seed": "5"})
        self.assertEqual(preview.status_code, 200, preview.text)
        self.assertIn('src="data:image/png;base64,', preview.text)
        self.assertIn('value="6"', preview.text)
        self.assertEqual(self.hx("GET", f"/modelos/{model_id}/estado.json").json()["revision"], new_revision)

        published = self.hx("POST", f"/modelos/{model_id}/publicar")
        self.assertEqual(published.status_code, 200, published.text)
        self.assertIn("Modelo publicado", published.text)
        self.assertIn(f'href="/geracao?modelo={model_id}"', published.text)
        gallery = self.page("/modelos").text
        self.assertIn("Modelo revisado", gallery)
        self.assertIn("2 campos", gallery)
        self.assertIn("Publicado", gallery)
        generation = self.page(f"/geracao?modelo={model_id}").text
        self.assertIn("Modelo revisado", generation)
        self.assertNotIn("ainda é um rascunho", generation)
        reverted = self.save(model_id, {**data, "revision": new_revision}, [fields[0], {**fields[1], "elements": []}])
        self.assertIn("voltou para rascunho", reverted.text)
        self.assertNotIn(f'value="{model_id}"', self.page("/geracao").text)

        self.assertEqual(self.client.get("/modelos/dnv/editor", headers={"Cookie": f"ocrsd_workspace={self.cookie}"},
                                         follow_redirects=False).status_code, 303)
        copied = self.hx("POST", "/modelos/certidao_nascimento/copia")
        copy_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", copied.headers["hx-redirect"]).group(1)
        self.assertIn("Certidão de Nascimento — cópia", self.page(f"/modelos/{copy_id}/editor").text)
        self.assertEqual(self.hx("POST", f"/modelos/{copy_id}/excluir").headers["hx-redirect"], "/modelos")
        self.assertEqual(self.page(f"/modelos/{copy_id}/editor").status_code, 404)

    def test_blank_template_composer_generates_pdf_and_dataset(self):
        import fitz

        new_page = self.page("/modelos/novo?modo=branco")
        self.assertIn('name="orientation" value="landscape"', new_page.text)
        self.assertEqual(self.hx("POST", "/modelos/em-branco", data={"label": "Composto", "size": "a5", "orientation": "portrait"}).status_code, 422)
        created = self.hx("POST", "/modelos/em-branco", data={"label": "Certidão composta", "size": "a4", "orientation": "landscape"})
        self.assertEqual(created.status_code, 200, created.text)
        model_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", created.headers["hx-redirect"]).group(1)
        data = self.editor_data(model_id)
        self.assertEqual(data["definition"]["page_size_px"], [3508, 2480])
        self.assertEqual(data["layout"], {"page": {"size": "a4", "orientation": "landscape"}, "elements": []})
        self.assertIsNone(data["preview"])
        self.assertIn("layout", {group["id"] for group in data["palette"]})
        self.assertIn("Documento composto", self.page("/modelos").text)
        layout = {"page": data["layout"]["page"], "elements": [
            {"id": "titulo", "kind": "text", "text": "CERTIDÃO DE TESTE\nSegunda linha", "font_size": 18, "bold": True, "align": "center", "bbox": [200, 100, 3300, 400]},
            {"id": "linha", "kind": "line", "stroke": 1.5, "bbox": [200, 450, 3300, 470]},
            {"id": "moldura", "kind": "box", "stroke": 1, "fill": True, "bbox": [200, 600, 1800, 800]}]}
        fields = [{"field_id": "nome", "label": "Nome", "data_type": "person_name", "bbox": [220, 620, 1780, 780]},
                  {"field_id": "cidade", "label": "Cidade", "data_type": "municipality", "bbox": [1900, 620, 3000, 780]},
                  {"field_id": "cpf", "label": "CPF", "data_type": "document", "validation": {"document_kind": "cpf"}, "bbox": [220, 900, 1200, 980]}]
        bad_layout = {**layout, "elements": [{"id": "x", "kind": "text", "text": "fora", "font_size": 10, "bbox": [0, 0, 9999, 10]}]}
        self.assertEqual(self.save(model_id, data, fields, bad_layout).status_code, 422)
        self.assertEqual(self.save(model_id, data, fields, None).status_code, 422)
        saved = self.save(model_id, data, fields, layout)
        self.assertEqual(saved.status_code, 200, saved.text)
        template = self.app.state.store.model_directory(model_id) / "template.pdf"
        with fitz.open(template) as document:
            self.assertIn("CERTIDÃO DE TESTE", document[0].get_text())
            self.assertEqual(round(document[0].rect.width), 842)
        self.assertEqual(self.hx("GET", f"/modelos/{model_id}/estado.json").json()["layout"]["elements"][0]["id"], "titulo")
        preview = self.hx("POST", f"/modelos/{model_id}/previa", data={"definition": json.dumps({**data["definition"], "fields": fields}),
                                                                       "layout": json.dumps(layout)})
        self.assertIn('src="data:image/png;base64,', preview.text)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 200)
        copied = self.hx("POST", f"/modelos/{model_id}/copia")
        copy_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", copied.headers["hx-redirect"]).group(1)
        self.assertEqual(self.editor_data(copy_id)["layout"]["elements"][2]["id"], "moldura")

        job = self.hx("POST", "/geracao/jobs", data=self.generation_form(document_type=model_id, mapping_id="default", fields=["nome", "cidade", "cpf"]))
        self.assertEqual(job.status_code, 200, job.text)
        self.run_worker()
        job_id = re.search(r'/geracao/jobs/([0-9a-f]{32})', job.text).group(1)
        row = self.app.state.store.get(job_id)
        self.assertEqual(row["status"], "succeeded", row["error"])
        truth = json.loads((self.app.state.store.directory(job_id) / "synthetic_0001.json").read_text())
        values = {entry["field_id"]: entry["value"] for entry in truth["fields"]}
        self.assertRegex(values["cpf"], r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
        self.assertTrue(values["cidade"])

    def test_custom_field_types_are_validated_reused_and_deleted(self):
        dialog = self.hx("GET", "/modelos/tipos/novo?base_type=integer&nome=Idade&validacao=%7B%22min_value%22%3A%201%2C%20%22max_value%22%3A%209%7D&aplicar=1")
        self.assertIn('value="Idade"', dialog.text)
        self.assertIn('<option value="intervalo" selected>', dialog.text)
        self.assertIn('name="aplicar" value="1"', dialog.text)
        base = {"name": "Matrícula", "slug": "matricula_civil", "description": "Número de matrícula", "base_type": "text",
                "rule": "mascara", "mask": "######.##.##.####.#.#####.###.#######-##", "blank_percent": "10", "escopo": "tipo"}
        self.assertEqual(self.hx("POST", "/modelos/tipos", data={**base, "rule": "intervalo"}).status_code, 422)
        self.assertEqual(self.hx("POST", "/modelos/tipos", data={**base, "slug": "Inválido!"}).status_code, 422)
        self.assertEqual(self.hx("POST", "/modelos/tipos", data={**base, "mask": "sem tokens"}).status_code, 422)
        self.assertEqual(self.hx("POST", "/modelos/tipos", data={**base, "base_type": "checkbox"}).status_code, 422)
        created = self.hx("POST", "/modelos/tipos", data=base)
        self.assertEqual(created.status_code, 200, created.text)
        events = json.loads(created.headers["hx-trigger-after-swap"])
        self.assertTrue(events["tipos-atualizados"])
        self.assertEqual(events["campo-personalizado"]["config"], {"data_type": "text", "custom_type": "matricula_civil", "label": "Matrícula",
                                                                   "validation": {"mask": base["mask"], "blank_probability": 0.1}})
        self.assertEqual(self.hx("POST", "/modelos/tipos", data=base).status_code, 409)
        single = self.hx("POST", "/modelos/tipos", data={**base, "slug": "uso_unico", "escopo": "campo", "rule": "lista", "allowed_values": "A\nB\nA"})
        self.assertNotIn("tipos-atualizados", json.loads(single.headers["hx-trigger-after-swap"]))
        self.assertEqual(json.loads(single.headers["hx-trigger-after-swap"])["campo-personalizado"]["config"]["validation"]["allowed_values"], ["A", "B"])
        palette = self.hx("GET", "/modelos/tipos/paleta").text
        self.assertIn("Matrícula", palette)
        self.assertNotIn("uso_unico", palette)
        listing = self.page("/presets/tipos").text
        self.assertIn("matricula_civil", listing)
        type_id = self.app.state.store.list_field_types(self.app.state.store.workspace(self.cookie))[0]["id"]
        self.assertNotIn("matricula_civil", self.page("/presets/tipos", cookie=self.workspace()).text)
        deleted = self.hx("POST", f"/presets/tipos/{type_id}/excluir")
        self.assertIn("Tipo personalizado excluído", deleted.text)
        self.assertIn("Nenhum tipo personalizado", deleted.text)

    def preset_form(self, **changes):
        form = {"preset_name": "Caneta azul", "description": "Teste", "model": "font", "intensity": "0.4", "handwriting": "on",
                "text_handwriting": "on", "bias": "0.75", "color": "#193787", "width": "1.5", "pressure": "0.25", "dropout": "0",
                "text_policy": "normalize", "writer_mode": "aleatorio", "variacoes": "1"}
        form.update(changes)
        return form

    def test_writing_preset_gallery_editor_and_previews(self):
        self.assertEqual(self.client.get("/presets", follow_redirects=False).headers["location"], "/presets/escrita")
        gallery = self.page("/presets/escrita")
        self.assertIn("Presets de Escrita", gallery.text)
        self.assertIn("Fonte manuscrita · natural", gallery.text)
        self.assertIn("Buscar presets...", gallery.text)
        self.assertNotIn("Laboratório de escrita", gallery.text)
        self.assertIn("/presets/escrita/padrao_fonte_natural/amostra.png?linhas=4&amp;v=", gallery.text)
        grid = self.hx("GET", "/presets/escrita?busca=regular", headers={"HX-Target": "grade-presets"})
        self.assertTrue(grid.text.lstrip().startswith('<div id="grade-presets"'))
        self.assertEqual(grid.text.count('class="preset-card '), 1)
        self.assertEqual(self.hx("GET", "/presets/escrita?modelo=sjvasquez", headers={"HX-Target": "grade-presets"}).text.count('class="preset-card '), 1)
        self.assertIn("Nenhum preset", self.hx("GET", "/presets/escrita?aba=meus", headers={"HX-Target": "grade-presets"}).text)
        thumbnail = self.page("/presets/escrita/padrao_fonte_natural/amostra.png?linhas=4")
        self.assertEqual(thumbnail.headers["content-type"], "image/png")
        self.assertEqual(self.page("/presets/escrita/padrao_fonte_natural/amostra.png?linhas=4").content, thumbnail.content)
        self.assertEqual(len(list((self.settings.storage / "previews").glob("*.png"))), 1)
        self.assertEqual(self.page("/presets/escrita/style_20260921_140258_43ec91/amostra.png").status_code, 422)

        editor = self.page("/presets/escrita/novo")
        self.assertIn("Novo preset", editor.text)
        self.assertIn("Atualizar prévia", editor.text)
        self.assertIn('name="intensity"', editor.text)
        self.assertNotIn('name="color" value', editor.text.split("Aparência da escrita")[1].split("Restaurar")[0])
        controls = self.hx("POST", "/presets/escrita/controles", data=self.preset_form(model="aaron1259", modelo_atual="font"))
        self.assertIn('name="bias" min="0" max="3" step="0.05" value="2.1"', controls.text)
        self.assertIn("Estilo" not in controls.text and 'max="511"', controls.text)
        self.assertIn('data-cor', controls.text)
        defaults = self.hx("POST", "/presets/escrita/controles", data=self.preset_form(model="sjvasquez", modelo_atual="sjvasquez", bias="0.1", acao="padroes"))
        self.assertIn('value="0.75"', defaults.text)
        self.assertIn("Estilo 12", defaults.text)
        preview = self.hx("POST", "/presets/escrita/previa", data=self.preset_form(variacoes="3"))
        self.assertEqual(preview.text.count('src="data:image/png;base64,'), 12)
        self.assertIn("Variação 3", preview.text)
        unavailable = self.hx("POST", "/presets/escrita/previa", data=self.preset_form(model="aaron1259"))
        self.assertIn("Modelo indisponível neste servidor", unavailable.text)
        self.assertEqual(self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(model="aaron1259", acao="novo")).status_code, 422)
        self.assertEqual(self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(preset_name="", acao="novo")).status_code, 422)

        created = self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(acao="novo", retorno="geracao"))
        preset_id = re.fullmatch(r"/presets/escrita/([0-9a-f]{32})\?retorno=geracao", created.headers["hx-redirect"]).group(1)
        page = self.page(f"/presets/escrita/{preset_id}")
        self.assertIn("versão 1", page.text)
        self.assertIn("Salvar alterações", page.text)
        self.assertIn("Criar link de compartilhamento", page.text)
        updated = self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(id=preset_id, acao="salvar", intensity="0.9"))
        self.assertEqual(json.loads(updated.headers["hx-trigger"])["preset-salvo"]["version"], 2)
        self.assertIn('value="0.9"', self.page(f"/presets/escrita/{preset_id}").text)
        copy = self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(id=preset_id, acao="novo"))
        copy_id = copy.headers["hx-redirect"].rsplit("/", 1)[1]
        self.assertIn('value="Cópia de Caneta azul"', self.page(f"/presets/escrita/{copy_id}").text)
        self.assertIn('value="0.9"', self.page(f"/presets/escrita/{preset_id}").text)
        standard = self.page("/presets/escrita/padrao_fonte_regular")
        self.assertIn("Preset padrão do OCRSD", standard.text)
        self.assertNotIn("Salvar alterações", standard.text)
        duplicated = self.hx("POST", "/presets/escrita/padrao_fonte_regular/duplicar", data={"retorno": ""})
        self.assertRegex(duplicated.headers["hx-redirect"], r"^/presets/escrita/[0-9a-f]{32}\?renomear=1$")
        self.assertIn('value="Cópia de Fonte manuscrita · regular"', self.page(duplicated.headers["hx-redirect"]).text)
        self.assertEqual(self.hx("POST", "/presets/escrita/padrao_fonte_regular/excluir").status_code, 409)
        deleted = self.hx("POST", f"/presets/escrita/{copy_id}/excluir")
        self.assertIn("Preset excluído", deleted.text)
        self.assertEqual(self.hx("POST", f"/presets/escrita/{copy_id}/excluir", data={"origem": "editor"}).status_code, 404)
        meus = self.hx("GET", "/presets/escrita?aba=meus&ordem=nome", headers={"HX-Target": "grade-presets"}).text
        self.assertLess(meus.index("Caneta azul"), meus.index("Cópia de Fonte manuscrita"))

        api = self.client.get("/api/v1/presets", headers={"X-Workspace-Token": self.cookie}).json()
        self.assertIn("Caneta azul", [item["label"] for item in api])
        self.assertIn("padrao_fonte_natural", [item["id"] for item in api])
        api_update = self.client.put(f"/api/v1/presets/{preset_id}", headers={"X-Workspace-Token": self.cookie},
                                     json={"name": "Caneta azul API", "writing": Writing().model_dump(), "filter_preset": "none", "shareable": False})
        self.assertEqual(api_update.status_code, 200, api_update.text)
        self.assertIn("versão 3", self.page(f"/presets/escrita/{preset_id}").text)
        models = self.page("/presets/escrita/modelos").text
        self.assertIn("recurso futuro", models)
        self.assertNotIn('type="file"', models)

    def test_writing_preset_sharing_and_generation_snapshot(self):
        created = self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(acao="novo", preset_name="Compartilhável"))
        preset_id = created.headers["hx-redirect"].rsplit("/", 1)[1]
        shared = self.hx("POST", f"/presets/escrita/{preset_id}/compartilhamento", data={"acao": "link"})
        link = re.search(r'value="(http://testserver/presets/escrita/convite/[0-9a-f]{32}#token=[^"]+)"', shared.text).group(1)
        token = unquote(link.split("#token=")[1])
        guest = self.workspace()
        self.assertEqual(self.page(f"/presets/escrita/{preset_id}", cookie=guest).status_code, 404)
        self.assertIn("data-convite", self.page(f"/presets/escrita/convite/{preset_id}", cookie=guest).text)
        self.assertEqual(self.hx("POST", f"/presets/escrita/convite/{preset_id}", cookie=guest, data={"token": "errado"}).status_code, 404)
        accepted = self.hx("POST", f"/presets/escrita/convite/{preset_id}", cookie=guest, data={"token": token})
        self.assertEqual((accepted.status_code, accepted.headers["hx-redirect"]), (204, f"/presets/escrita/{preset_id}"))
        guest_page = self.page(f"/presets/escrita/{preset_id}", cookie=guest).text
        self.assertIn("Compartilhado comigo", guest_page)
        self.assertNotIn("Salvar alterações", guest_page)
        self.assertEqual(self.hx("POST", "/presets/escrita/salvar", cookie=guest, data=self.preset_form(id=preset_id, acao="salvar")).status_code, 404)
        self.assertIn("Compartilhável", self.hx("GET", "/presets/escrita?aba=compartilhados", cookie=guest, headers={"HX-Target": "grade-presets"}).text)
        self.assertIn("1 espaço de trabalho tem acesso", self.page(f"/presets/escrita/{preset_id}").text)
        rotated = self.hx("POST", f"/presets/escrita/{preset_id}/compartilhamento", data={"acao": "novo-link"})
        self.assertNotIn(token, unquote(rotated.text))
        self.assertEqual(self.page(f"/presets/escrita/{preset_id}", cookie=guest).status_code, 200)
        self.hx("POST", f"/presets/escrita/{preset_id}/compartilhamento", data={"acao": "privado"})
        self.assertEqual(self.page(f"/presets/escrita/{preset_id}", cookie=guest).status_code, 404)
        self.assertNotIn("Compartilhável", self.page("/presets/escrita", cookie=guest).text)

        form = self.generation_form(name="Rascunho preservado")
        draft = self.hx("POST", "/geracao/rascunho", data=form)
        self.assertEqual(draft.headers["hx-redirect"], "/presets/escrita?retorno=geracao")
        returning = self.page("/presets/escrita?retorno=geracao").text
        self.assertIn("rascunho da geração está preservado", returning)
        self.assertIn(f"/geracao?rascunho=1&amp;preset={preset_id}", returning)
        restored = self.page(f"/geracao?rascunho=1&preset={preset_id}").text
        self.assertIn("Documento base", restored)
        restored_id = re.search(r'name="draft_id" value="([0-9a-f]+)"', restored).group(1)
        restored_state = self.app.state.store.wizard_draft(restored_id, self.app.state.store.workspace(self.cookie))
        self.assertEqual(restored_state["preset"], preset_id)
        self.assertEqual(restored_state["fields"], ["cpf", "matricula", "sexo"])
        applied = self.hx("POST", "/geracao/formulario", data={**form, "acao": "preset", "preset": "padrao_fonte_variada"})
        self.assertIn('name="preset" value="padrao_fonte_variada" checked', applied.text)
        self.assertIn('value="0.95"', applied.text)

        job = self.hx("POST", "/geracao/jobs", data={**form, "preset": preset_id, "intensity": "0.4"})
        self.assertEqual(job.status_code, 200, job.text)
        job_id = re.search(r'/geracao/jobs/([0-9a-f]{32})', job.text).group(1)
        reference = json.loads(self.app.state.store.get(job_id)["preset"])
        self.assertEqual(reference, {"id": preset_id, "name": "Compartilhável", "version": 1, "kind": "meu", "adjusted": False})
        self.hx("POST", "/presets/escrita/salvar", data=self.preset_form(id=preset_id, acao="salvar", intensity="0.1"))
        row = self.app.state.store.get(job_id)
        self.assertEqual(json.loads(row["preset"])["version"], 1)
        self.assertEqual(json.loads(row["options"])["writing"]["intensity"], 0.4)
        self.run_worker()
        self.assertIn("Compartilhável · v1", self.hx("GET", f"/resultados/{job_id}/detalhes").text)
        adjusted = self.hx("POST", "/geracao/jobs", data={**form, "preset": preset_id, "intensity": "0.7"})
        adjusted_id = re.search(r'/geracao/jobs/([0-9a-f]{32})', adjusted.text).group(1)
        self.assertTrue(json.loads(self.app.state.store.get(adjusted_id)["preset"])["adjusted"])

    def test_realism_tools_and_saved_configurations(self):
        realism_page = self.page("/ferramentas/realismo")
        self.assertIn("Comparação de realismo", realism_page.text)
        self.assertEqual(self.hx("POST", "/ferramentas/realismo", data={"document_type": "dnv", "seed": "1", "filter_preset": "medium"}).text.count("data:image/jpeg"), 2)
        self.assertEqual(self.hx("POST", "/ferramentas/realismo", data={"document_type": "dnv", "seed": "1", "filter_preset": "none"}).status_code, 422)
        realism = self.hx("POST", "/geracao/realismo", data={"document_type": "dnv", "seed": "7", "filter_preset": "none"})
        self.assertEqual(realism.status_code, 200, realism.text)
        self.assertEqual(realism.text.count('src="data:image/jpeg;base64,'), 2)
        self.assertIn('value="light" checked', realism.text)
        apply = self.hx("POST", "/geracao/realismo/aplicar", data={"filter_preset": "strong"})
        self.assertIn('hx-swap-oob="true"', apply.text)
        self.assertIn('value="strong" selected', apply.text)
        self.assertEqual(apply.headers["hx-trigger-after-swap"], "fechar-modal, realismo-aplicado")
        model_change = self.hx("POST", "/geracao/formulario", data={**self.generation_form(writer="7"), "acao": "modelo"})
        self.assertIn('name="writer" min="0" max="511" value=""', model_change.text)

        self.assertEqual(self.hx("POST", "/geracao/configuracoes", data=self.generation_form()).status_code, 422)
        stored = self.hx("POST", "/geracao/configuracoes", data={**self.generation_form(seed="99"), "nome_configuracao": "Seed 99", "preset": "padrao_fonte_natural"})
        self.assertIn("Seed 99", stored.text)
        self.assertIn("Configuração salva", stored.text)
        config_id = re.search(r'<option value="([0-9a-f]{32})">Seed 99', stored.text).group(1)
        loaded = self.hx("POST", "/geracao/formulario", data={"acao": "configuracao", "configuracao": config_id})
        self.assertIn('name="seed" min="0" max="4294967295" value="99"', loaded.text)
        self.assertEqual(self.app.state.store.config(config_id, self.app.state.store.workspace(self.cookie))["options"]["seed"], 99)
        self.assertIn("Documento base", self.page(f"/geracao?configuracao={config_id}").text)


if __name__ == "__main__":
    unittest.main()
