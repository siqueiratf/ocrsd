import base64
import json
from pathlib import Path
import re
import tempfile
import unittest

try:
    import fitz
    import httpx2  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.settings import Settings
from tests.test_web import ASGITestClient


def _pdf(width: int = 600, height: int = 800) -> bytes:
    with fitz.open() as document:
        document.new_page(width=width, height=height)
        return document.tobytes()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_documents_root(root: Path) -> None:
    sample = root / "sample"
    (sample / "mappings").mkdir(parents=True)
    (sample / "template.pdf").write_bytes(_pdf())
    _write_json(sample / "mappings" / "sample_v1.json",
               {"template_id": "sample", "dpi": 300, "page_size_px": [600, 800],
                "fields": [{"field_id": "nome", "label": "Nome", "data_type": "text", "bbox": [20, 20, 300, 60]}]})
    _write_json(sample / "modelo.json", {"active_mapping": "sample_v1", "label": "Modelo Base", "description": "Modelo de teste"})

    legacy = root / "legacy"
    legacy.mkdir()
    (legacy / "legacy.pdf").write_bytes(_pdf())
    _write_json(legacy / "legacy_fields.json",
               {"template_id": "legacy", "dpi": 300, "page_size_px": [600, 800],
                "fields": [{"field_id": "campo", "label": "Campo", "data_type": "text", "bbox": [20, 20, 300, 60]}]})


class StandardModelsTestCase(unittest.TestCase):
    standard_edits = False

    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.addCleanup(self.storage.cleanup)
        self.documents = tempfile.TemporaryDirectory()
        self.addCleanup(self.documents.cleanup)
        self.documents_root = Path(self.documents.name)
        _build_documents_root(self.documents_root)
        self.settings = Settings(storage=Path(self.storage.name), documents=self.documents_root,
                                 min_free_bytes=0, neural_enabled=False, standard_edits=self.standard_edits)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)

    def workspace(self) -> dict:
        token = self.client.post("/api/v1/workspaces", json={}).json()["token"]
        return {"X-Workspace-Token": token}

    def create_custom_model(self, headers, *, field: dict | None = None, label: str = "Meu Modelo Personalizado") -> dict:
        field = field or {"field_id": "nome", "label": "Nome", "data_type": "text", "bbox": [20, 20, 300, 60]}
        mapping = {"template_id": "modelo_web", "dpi": 300, "page_size_px": [600, 800], "fields": [field]}
        payload = {"label": label, "description": "Descrição do modelo", "filename": "modelo.pdf",
                  "pdf_base64": base64.b64encode(_pdf()).decode(), "mapping_id": "default", "mapping": mapping}
        response = self.client.post("/api/v1/models", json=payload, headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()


class FlagOffTests(StandardModelsTestCase):
    standard_edits = False

    def test_promote_is_rejected(self):
        headers = self.workspace()
        model = self.create_custom_model(headers)
        response = self.client.post(f'/api/v1/models/{model["id"]}/standard',
                                     json={"key": "novo_padrao", "label": "Novo Padrão", "description": ""}, headers=headers)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(response.json()["error"]["code"], "standard_edits_disabled")
        self.assertFalse((self.documents_root / "novo_padrao").exists())

    def test_save_builtin_mapping_is_rejected_and_files_untouched(self):
        headers = self.workspace()
        mapping_path = self.documents_root / "sample" / "mappings" / "sample_v1.json"
        original = mapping_path.read_bytes()
        definition = json.loads(original)
        definition["fields"][0]["label"] = "Editado"
        response = self.client.put("/api/v1/models/sample/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(response.status_code, 403, response.text)
        self.assertEqual(response.json()["error"]["code"], "builtin_protected")
        self.assertEqual(mapping_path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in (self.documents_root / "sample" / "mappings").iterdir()), ["sample_v1.json"])


class FlagOnTests(StandardModelsTestCase):
    standard_edits = True

    def test_promote_creates_standard_and_catalog_updates_without_restart(self):
        headers = self.workspace()
        model = self.create_custom_model(headers, label="Certidão Especial")
        response = self.client.post(f'/api/v1/models/{model["id"]}/standard',
                                     json={"key": "certidao_especial", "label": "Certidão Especial", "description": "Teste"},
                                     headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        body = response.json()
        self.assertTrue(body["builtin"])
        self.assertTrue(body["editable"])
        self.assertEqual(body["mapping_id"], "certidao_especial_v1")

        folder = self.documents_root / "certidao_especial"
        self.assertTrue((folder / "template.pdf").is_file())
        self.assertTrue((folder / "modelo.json").is_file())
        self.assertTrue((folder / "mappings" / "certidao_especial_v1.json").is_file())
        self.assertTrue((folder / "dataset" / ".gitkeep").is_file())
        metadata = json.loads((folder / "modelo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["active_mapping"], "certidao_especial_v1")
        self.assertIn("promovido de modelo personalizado", metadata["source"])

        types = {item["id"] for item in self.client.get("/api/v1/document-types").json()}
        self.assertIn("certidao_especial", types)

        validated = self.client.post("/api/v1/generation/validate",
                                     json={"document_type": "certidao_especial", "mapping_id": "certidao_especial_v1",
                                           "fields": ["nome"], "count": 1, "seed": 1})
        self.assertEqual(validated.status_code, 200, validated.text)

    def test_promote_duplicate_key_conflicts(self):
        headers = self.workspace()
        model = self.create_custom_model(headers)
        first = self.client.post(f'/api/v1/models/{model["id"]}/standard',
                                 json={"key": "duplicado", "label": "Duplicado", "description": ""}, headers=headers)
        self.assertEqual(first.status_code, 201, first.text)
        other = self.create_custom_model(headers, label="Outro modelo")
        second = self.client.post(f'/api/v1/models/{other["id"]}/standard',
                                  json={"key": "duplicado", "label": "Duplicado outra vez", "description": ""}, headers=headers)
        self.assertEqual(second.status_code, 409, second.text)
        self.assertEqual(second.json()["error"]["code"], "standard_exists")
        metadata = json.loads((self.documents_root / "duplicado" / "modelo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["label"], "Duplicado")

    def test_promote_bad_key_is_sanitized_and_stays_inside_documents_root(self):
        headers = self.workspace()
        model = self.create_custom_model(headers)
        response = self.client.post(f'/api/v1/models/{model["id"]}/standard',
                                     json={"key": "../evil", "label": "Chave perigosa", "description": ""}, headers=headers)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["id"], "evil")
        self.assertTrue((self.documents_root / "evil").is_dir())
        self.assertFalse((self.documents_root.parent / "evil").exists())

    def test_promote_with_mapping_issues_is_rejected(self):
        headers = self.workspace()
        checkbox_field = {"field_id": "chk", "label": "Chk", "data_type": "checkbox", "bbox": [1, 1, 50, 50]}
        model = self.create_custom_model(headers, field=checkbox_field)
        response = self.client.post(f'/api/v1/models/{model["id"]}/standard',
                                     json={"key": "invalido", "label": "Inválido", "description": ""}, headers=headers)
        self.assertEqual(response.status_code, 422, response.text)
        self.assertFalse((self.documents_root / "invalido").exists())

    def test_save_builtin_mapping_versions_and_preserves_old_files(self):
        headers = self.workspace()
        definition = json.loads((self.documents_root / "sample" / "mappings" / "sample_v1.json").read_text(encoding="utf-8"))
        definition["fields"][0]["label"] = "Nome completo"
        first = self.client.put("/api/v1/models/sample/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["id"], "sample_v2")
        metadata = json.loads((self.documents_root / "sample" / "modelo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["active_mapping"], "sample_v2")

        definition["fields"][0]["label"] = "Nome completo 2"
        second = self.client.put("/api/v1/models/sample/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["id"], "sample_v3")
        metadata = json.loads((self.documents_root / "sample" / "modelo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["active_mapping"], "sample_v3")

        mappings_dir = self.documents_root / "sample" / "mappings"
        self.assertEqual(sorted(p.name for p in mappings_dir.iterdir()),
                         ["sample_v1.json", "sample_v2.json", "sample_v3.json"])
        original = json.loads((mappings_dir / "sample_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(original["fields"][0]["label"], "Nome")

    def test_save_builtin_mapping_with_issues_is_rejected_and_untouched(self):
        headers = self.workspace()
        mapping_path = self.documents_root / "sample" / "mappings" / "sample_v1.json"
        original = mapping_path.read_bytes()
        definition = json.loads(original)
        definition["fields"] = []
        response = self.client.put("/api/v1/models/sample/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(mapping_path.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in (self.documents_root / "sample" / "mappings").iterdir()), ["sample_v1.json"])

    def test_fallback_folder_gets_mappings_directory_and_both_mappings_stay_discoverable(self):
        headers = self.workspace()
        definition = json.loads((self.documents_root / "legacy" / "legacy_fields.json").read_text(encoding="utf-8"))
        definition["fields"][0]["label"] = "Campo editado"
        response = self.client.put("/api/v1/models/legacy/mapping", json={"definition": definition}, headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["id"], "legacy_fields_v2")

        mappings_dir = self.documents_root / "legacy" / "mappings"
        self.assertTrue(mappings_dir.is_dir())
        self.assertTrue((mappings_dir / "legacy_fields_v2.json").is_file())
        self.assertTrue((self.documents_root / "legacy" / "legacy_fields.json").is_file())

        document = next(item for item in self.client.get("/api/v1/document-types").json() if item["id"] == "legacy")
        ids = {item["id"] for item in document["mappings"]}
        self.assertEqual(ids, {"legacy_fields", "legacy_fields_v2"})

        metadata = json.loads((self.documents_root / "legacy" / "modelo.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["active_mapping"], "legacy_fields_v2")
        self.assertEqual(metadata["label"], "LEGACY")


class BuiltinEditorUiTests(unittest.TestCase):
    def build_app(self, standard_edits: bool):
        storage = tempfile.TemporaryDirectory()
        self.addCleanup(storage.cleanup)
        documents = tempfile.TemporaryDirectory()
        self.addCleanup(documents.cleanup)
        root = Path(documents.name)
        _build_documents_root(root)
        settings = Settings(storage=Path(storage.name), documents=root, min_free_bytes=0, neural_enabled=False,
                            standard_edits=standard_edits)
        app = create_app(settings)
        client = ASGITestClient(app)
        self.addCleanup(client.close)
        return client

    def cookie(self, client) -> str:
        response = client.get("/modelos")
        return re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)

    def test_builtin_editor_shows_standard_save_label_only_when_flag_on(self):
        on_client = self.build_app(True)
        cookie = self.cookie(on_client)
        page = on_client.get("/modelos/sample/editor", headers={"Cookie": f"ocrsd_workspace={cookie}"})
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn("Salvar nova versão do padrão", page.text)

        off_client = self.build_app(False)
        cookie = self.cookie(off_client)
        redirected = off_client.get("/modelos/sample/editor", headers={"Cookie": f"ocrsd_workspace={cookie}"},
                                    follow_redirects=False)
        self.assertEqual(redirected.status_code, 303)


if __name__ == "__main__":
    unittest.main()
