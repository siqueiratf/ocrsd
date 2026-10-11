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
        environment = {**os.environ, **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT, env=environment,
                       capture_output=True, text=True, timeout=90, check=True)

    def test_pages_session_security_and_assets(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual((response.status_code, response.headers["location"]), (303, "/modelos"))
        first = self.client.get("/modelos")
        self.assertIn("HttpOnly", first.headers["set-cookie"])
        self.assertIn("SameSite=lax", first.headers["set-cookie"])
        self.assertIn("default-src 'self'", first.headers["content-security-policy"])
        self.assertEqual(first.headers["x-frame-options"], "SAMEORIGIN")
        self.assertIn("Modelos de documentos", first.text)
        self.assertIn('href="/geracao?modelo=certidao_nascimento"', first.text)
        self.assertNotIn("set-cookie", self.page("/modelos").headers)
        for path, heading in (("/geracao", "Nova geração de dataset"), ("/resultados", "Datasets gerados"),
                              ("/presets", "Presets de escrita e realismo")):
            response = self.page(path)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn(heading, response.text)
            self.assertIn('aria-current="page"', response.text)
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

    def test_legacy_workspace_token_is_adopted(self):
        legacy = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        adopted = self.hx("POST", "/sessao", data={"token": legacy})
        self.assertEqual(adopted.status_code, 204)
        self.assertEqual(adopted.headers["hx-refresh"], "true")
        self.assertIn(f"ocrsd_workspace={legacy}", adopted.headers["set-cookie"])
        self.assertEqual(self.hx("POST", "/sessao", data={"token": "invalido"}).status_code, 401)

    def test_generation_results_and_revocable_sharing(self):
        page = self.page("/geracao?modelo=certidao_nascimento")
        self.assertIn('value="data_nascimento_por_extenso"', page.text)
        sections = self.hx("POST", "/geracao/formulario", data={**self.generation_form(), "acao": "documento", "document_type": "dnv"})
        self.assertIn('id="previa" hx-swap-oob="innerHTML"', sections.text)
        self.assertIn("/modelos/dnv/previa.png", sections.text)
        self.assertEqual(sections.text.count('name="fields"'), sections.text.count("checked") - 2)
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
        self.assertIn("Limite de 10", self.hx("POST", "/geracao/jobs", data=self.generation_form(count="11")).text)

        queued = self.hx("POST", "/geracao/jobs", data=self.generation_form())
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = re.search(r'hx-get="/geracao/jobs/([0-9a-f]{32})"', queued.text).group(1)
        self.assertIn('hx-trigger="every 1500ms"', queued.text)
        self.assertRegex(queued.text, r'id="enviar"[^>]*disabled[^>]*hx-swap-oob="true"')
        self.run_worker()
        finished = self.hx("GET", f"/geracao/jobs/{job_id}")
        self.assertIn(f'href="/resultados?job={job_id}"', finished.text)
        self.assertNotIn("hx-trigger", finished.text)

        results = self.page(f"/resultados?job={job_id}")
        self.assertIn('class="is-selected"', results.text)
        self.assertIn("Visualizador do documento sintético", results.text)
        artifacts = self.app.state.store.artifacts(self.app.state.store.get(job_id))
        pdf = next(item for item in artifacts if item["media_type"] == "application/pdf")
        json_file = next(item for item in artifacts if item["name"] == "synthetic_0001.json")
        archive = next(item for item in artifacts if item["media_type"] == "application/zip")
        self.assertIn(f'/resultados/{job_id}/arquivos/{archive["id"]}', results.text)
        inline = self.page(f'/resultados/{job_id}/arquivos/{pdf["id"]}?download=false')
        self.assertTrue(inline.content.startswith(b"%PDF"))
        self.assertIn("inline", inline.headers["content-disposition"])
        self.assertNotIn("content-security-policy", inline.headers)
        viewer = self.hx("GET", f'/resultados/{job_id}/visualizar/{json_file["id"]}')
        self.assertIn('<pre class="json-view">', viewer.text)
        self.assertIn("&#34;synthetic&#34;: true", viewer.text)
        self.assertIn('id="arquivos" hx-swap-oob="innerHTML"', viewer.text)
        self.assertIn("Nenhuma geração encontrada", self.hx("GET", "/resultados/lista?busca=inexistente").text)
        self.assertIn("Dataset pela interface", self.hx("GET", "/resultados/lista?status=succeeded").text)
        detail = self.hx("GET", f"/resultados/{job_id}?busca=&status=")
        self.assertIn('id="tabela-jobs"', detail.text)
        self.assertIn('id="filtro-job" name="job" value="' + job_id, detail.text)
        other = self.workspace()
        self.assertEqual(self.hx("GET", f"/resultados/{job_id}", cookie=other).status_code, 404)
        self.assertNotIn(job_id, self.page("/resultados", cookie=other).text)

        reused = self.page(f"/geracao?job={job_id}")
        self.assertIn("Dataset pela interface — nova execução", reused.text)
        self.assertEqual(reused.text.count('name="fields" value="cpf" checked'), 1)
        self.assertNotIn('name="fields" value="filiacao" checked', reused.text)

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

    def test_model_upload_editor_and_catalog_copy(self):
        response = self.hx("POST", "/modelos", data={"label": "Modelo enviado", "description": "Pela interface", "mapping_id": "default"},
                           files={"pdf": ("template.pdf", TEMPLATE.read_bytes(), "application/pdf")})
        self.assertEqual(response.status_code, 200, response.text)
        model_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", response.headers["hx-redirect"]).group(1)
        rejected = self.hx("POST", "/modelos", data={"label": "Sem PDF", "mapping_id": "default"},
                           files={"pdf": ("notas.txt", b"texto", "text/plain")})
        self.assertEqual(rejected.status_code, 422)
        editor = self.page(f"/modelos/{model_id}/editor")
        data = json.loads(re.search(r'<script type="application/json" id="dados-editor">(.*?)</script>', editor.text, re.S).group(1))
        self.assertEqual(data["definition"]["fields"], [])
        self.assertEqual(data["preview"], f"/modelos/{model_id}/previa.png")
        self.assertEqual(self.page(data["preview"]).headers["content-type"], "image/png")
        definition = {**data["definition"], "fields": [{"field_id": "nome", "label": "Nome", "data_type": "person_name", "bbox": [100, 100, 600, 160]}]}
        saved = self.hx("POST", f"/modelos/{model_id}/mapeamento", data={"definition": json.dumps(definition), "label": "Modelo revisado",
                                                                         "description": "Mapeado", "status": "published"})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertIn("Mapeamento salvo", saved.text)
        self.assertEqual(self.hx("GET", f"/modelos/{model_id}/mapeamento.json").json()["fields"][0]["field_id"], "nome")
        outside = {**definition, "fields": [{"field_id": "nome", "bbox": [0, 0, 999999, 10]}]}
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/mapeamento", data={"definition": json.dumps(outside), "label": "x"}).status_code, 422)
        gallery = self.page("/modelos").text
        self.assertIn("Modelo revisado", gallery)
        self.assertIn("1 campos", gallery)
        self.assertIn('value="nome"', self.page(f"/geracao?modelo={model_id}").text)

        self.assertEqual(self.client.get("/modelos/dnv/editor", headers={"Cookie": f"ocrsd_workspace={self.cookie}"},
                                         follow_redirects=False).status_code, 303)
        copied = self.hx("POST", "/modelos/certidao_nascimento/copia")
        copy_id = re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", copied.headers["hx-redirect"]).group(1)
        self.assertIn("CERTIDAO NASCIMENTO — cópia", self.page(f"/modelos/{copy_id}/editor").text)
        self.assertEqual(self.hx("POST", f"/modelos/{copy_id}/excluir").headers["hx-redirect"], "/modelos")
        self.assertEqual(self.page(f"/modelos/{copy_id}/editor").status_code, 404)

    def test_presets_realism_and_saved_configurations(self):
        writing = {key: value for key, value in self.generation_form().items()
                   if key in {"model", "handwriting", "text_handwriting", "intensity", "writer", "bias", "color", "width", "pressure", "dropout", "text_policy"}}
        samples = self.hx("POST", "/presets/amostras", data=writing)
        self.assertEqual(samples.status_code, 200, samples.text)
        self.assertEqual(samples.text.count('src="data:image/png;base64,'), 3)
        self.assertEqual(self.hx("POST", "/presets/amostras", data={**writing, "model": "sjvasquez"}).status_code, 422)
        self.assertIn("Laboratório de escrita", self.hx("GET", "/presets/laboratorio").text)
        saved = self.hx("POST", "/presets", data={**writing, "color": "#aa0000", "preset_name": "Caneta vermelha"})
        self.assertEqual(saved.headers["hx-trigger"], "presets-atualizados")
        self.assertIn("Caneta vermelha", saved.text)
        preset_id = self.app.state.store.list_presets(self.app.state.store.workspace(self.cookie))[0]["id"]
        loaded = self.hx("POST", "/presets/formulario", data={"preset": preset_id})
        self.assertIn('value="#aa0000"', loaded.text)
        self.assertIn("Caneta vermelha", self.hx("GET", "/geracao/presets").text)
        applied = self.hx("POST", "/geracao/formulario", data={**self.generation_form(), "acao": "preset", "preset": preset_id})
        self.assertIn('value="#aa0000"', applied.text)
        model_change = self.hx("POST", "/geracao/formulario", data={**self.generation_form(writer="7"), "acao": "modelo"})
        self.assertIn('name="writer" min="0" max="511" value=""', model_change.text)
        self.assertNotIn("Caneta vermelha", self.hx("POST", f"/presets/{preset_id}/excluir").text)

        realism = self.hx("POST", "/geracao/realismo", data={"document_type": "dnv", "seed": "7", "filter_preset": "none"})
        self.assertEqual(realism.status_code, 200, realism.text)
        self.assertEqual(realism.text.count('src="data:image/jpeg;base64,'), 2)
        self.assertIn('value="light" checked', realism.text)
        apply = self.hx("POST", "/geracao/realismo/aplicar", data={"filter_preset": "strong"})
        self.assertIn('hx-swap-oob="true"', apply.text)
        self.assertIn('value="strong" selected', apply.text)
        self.assertEqual(apply.headers["hx-trigger-after-swap"], "fechar-modal, realismo-aplicado")
        self.assertEqual(self.hx("POST", "/presets/realismo", data={"document_type": "dnv", "seed": "1", "filter_preset": "medium"}).text.count("data:image/jpeg"), 2)

        self.assertEqual(self.hx("POST", "/geracao/configuracoes", data=self.generation_form()).status_code, 422)
        stored = self.hx("POST", "/geracao/configuracoes", data={**self.generation_form(seed="99"), "nome_configuracao": "Seed 99"})
        self.assertIn("Seed 99", stored.text)
        self.assertIn("Configuração salva", stored.text)
        config_id = re.search(r'<option value="([0-9a-f]{32})">Seed 99', stored.text).group(1)
        loaded = self.hx("POST", "/geracao/formulario", data={"acao": "configuracao", "configuracao": config_id})
        self.assertIn('name="seed" min="0" max="4294967295" value="99"', loaded.text)
        self.assertIn('value="99"', self.page(f"/geracao?configuracao={config_id}").text)


if __name__ == "__main__":
    unittest.main()
