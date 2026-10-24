import json
import tempfile
import unittest
from pathlib import Path

from src.document_catalog import catalog_issues, catalog_path, load_catalog, save_catalog
from src.document_types import DOCUMENTS_ROOT, DocumentType, discover_document_types


class LoadSaveTests(unittest.TestCase):
    def test_load_missing_catalog_returns_empty_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_catalog(Path(directory)), {"orgaos": {}, "documentos": {}})

    def test_load_ignores_malformed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalogo.json").write_text("not json", encoding="utf-8")
            self.assertEqual(load_catalog(root), {"orgaos": {}, "documentos": {}})

    def test_save_then_load_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"orgaos": {"cnj": {"sigla": "CNJ", "nome": "Conselho Nacional de Justiça", "site": "https://www.cnj.jus.br"}},
                    "documentos": {"exemplo": {"orgao": "cnj", "label": "Exemplo", "description": "Um exemplo",
                                               "documentacao": "https://exemplo.gov.br", "referencia": "Ref", "categoria": "Teste",
                                               "prioridade": "alta", "status": "disponivel"}}}
            save_catalog(root, data)
            self.assertTrue(catalog_path(root).is_file())
            self.assertEqual(json.loads(catalog_path(root).read_text(encoding="utf-8")), data)
            self.assertTrue(catalog_path(root).read_text(encoding="utf-8").endswith("\n"))
            self.assertEqual(load_catalog(root), data)

    def test_save_is_atomic_and_drops_unknown_top_level_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_catalog(root, {"orgaos": {}, "documentos": {}, "lixo": True})
            self.assertEqual(load_catalog(root), {"orgaos": {}, "documentos": {}})
            self.assertFalse(catalog_path(root).with_suffix(".tmp").exists())


class CatalogIssuesTests(unittest.TestCase):
    def test_valid_catalog_has_no_issues(self):
        catalog = {"orgaos": {"cnj": {"sigla": "CNJ", "nome": "CNJ", "site": "https://www.cnj.jus.br"}},
                   "documentos": {"a": {"orgao": "cnj", "status": "disponivel", "prioridade": "alta",
                                        "documentacao": "https://exemplo.gov.br"}}}
        self.assertEqual(catalog_issues(catalog, {"a": object()}), [])

    def test_unknown_orgao_is_flagged(self):
        catalog = {"orgaos": {}, "documentos": {"a": {"orgao": "fantasma", "status": "disponivel", "prioridade": "alta"}}}
        issues = catalog_issues(catalog, {"a": object()})
        self.assertTrue(any("órgão desconhecido" in issue for issue in issues))

    def test_folder_without_entry_is_flagged(self):
        issues = catalog_issues({"orgaos": {}, "documentos": {}}, {"orfao": object()})
        self.assertTrue(any("documents/orfao" in issue for issue in issues))

    def test_disponivel_without_folder_and_without_variante_is_flagged(self):
        catalog = {"orgaos": {"x": {"sigla": "X", "nome": "X"}},
                   "documentos": {"fantasma": {"orgao": "x", "status": "disponivel", "prioridade": "alta"}}}
        issues = catalog_issues(catalog, {})
        self.assertTrue(any("fantasma" in issue and "disponível" in issue for issue in issues))

    def test_disponivel_without_folder_but_with_variante_is_not_flagged(self):
        catalog = {"orgaos": {"x": {"sigla": "X", "nome": "X"}},
                   "documentos": {"base": {"orgao": "x", "status": "disponivel", "prioridade": "alta"},
                                 "variante": {"orgao": "x", "status": "disponivel", "prioridade": "alta", "variante_de": "base"}}}
        issues = catalog_issues(catalog, {"base": object()})
        self.assertEqual(issues, [])

    def test_bad_status_and_priority_and_url_are_flagged(self):
        catalog = {"orgaos": {"x": {"sigla": "X", "nome": "X", "site": "ftp://nope"}},
                   "documentos": {"a": {"orgao": "x", "status": "rascunho", "prioridade": "altíssima",
                                        "documentacao": "exemplo.gov.br"}}}
        issues = catalog_issues(catalog, {"a": object()})
        self.assertEqual(len(issues), 4)

    def test_real_documents_folder_catalog_has_no_issues(self):
        discovered = discover_document_types(DOCUMENTS_ROOT)
        catalog = load_catalog(DOCUMENTS_ROOT)
        self.assertEqual(catalog_issues(catalog, discovered), [])


class DocumentTypeCatalogFallbackTests(unittest.TestCase):
    def _folder(self, root: Path, key: str, modelo: dict | None = None) -> Path:
        folder = root / key
        (folder / "mappings").mkdir(parents=True)
        (folder / "template.pdf").touch()
        (folder / "mappings" / f"{key}_v1.json").write_text("{}", encoding="utf-8")
        if modelo is not None:
            (folder / "modelo.json").write_text(json.dumps(modelo), encoding="utf-8")
        return folder

    def test_label_and_description_prefer_catalog_over_modelo_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._folder(root, "a", {"label": "Do modelo.json", "description": "Descrição do modelo.json"})
            save_catalog(root, {"orgaos": {"cnj": {"sigla": "CNJ", "nome": "CNJ"}},
                                "documentos": {"a": {"orgao": "cnj", "label": "Do catálogo", "description": "Descrição do catálogo",
                                                     "documentacao": "https://exemplo.gov.br", "categoria": "Teste",
                                                     "status": "disponivel", "prioridade": "alta"}}})
            document = discover_document_types(root)["a"]
            self.assertEqual(document.label, "Do catálogo")
            self.assertEqual(document.description, "Descrição do catálogo")
            self.assertEqual(document.orgao["key"], "cnj")
            self.assertEqual(document.documentacao, "https://exemplo.gov.br")
            self.assertEqual(document.categoria, "Teste")

    def test_falls_back_to_modelo_json_when_no_catalog_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._folder(root, "b", {"label": "Do modelo.json", "description": "Descrição do modelo.json"})
            document = discover_document_types(root)["b"]
            self.assertEqual(document.label, "Do modelo.json")
            self.assertEqual(document.description, "Descrição do modelo.json")
            self.assertEqual(document.orgao, {"key": "outros", "sigla": "Outros", "nome": "Outros"})
            self.assertEqual(document.documentacao, "")

    def test_falls_back_to_key_when_no_catalog_and_no_modelo_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._folder(root, "sem_metadados")
            document = discover_document_types(root)["sem_metadados"]
            self.assertEqual(document.label, "SEM METADADOS")
            self.assertEqual(document.description, "")

    def test_default_document_type_without_catalog_field_uses_fallback_orgao(self):
        document = DocumentType(key="x", root=Path("."), template=Path("x.pdf"), mappings=(), presets=(), dataset=Path("."))
        self.assertEqual(document.orgao, {"key": "outros", "sigla": "Outros", "nome": "Outros"})


if __name__ == "__main__":
    unittest.main()
