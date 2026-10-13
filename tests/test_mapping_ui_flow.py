from datetime import date
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

try:
    import httpx2
except ImportError as exc:
    raise unittest.SkipTest("Instale os extras web e test") from exc

from src.web.api import create_app
from src.web.settings import Settings
from src.semantic.data import LOCATION_PROFILES
from tests.test_web import ASGITestClient


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf"
DEATH = ROOT / "documents/do/do.pdf"
PROVENANCE = ("method", "provider", "model", "registry_version", "analysis_id", "analysis_source", "page", "created_at")
CITIES = {(profile["city"], profile["uf"]) for profile in LOCATION_PROFILES}
ACCEPTED = ("deceased.name", "deceased.sex", "deceased.birth_date", "death.date", "deceased.age",
            "deceased.residence.municipality", "deceased.residence.uf")


class AssistedMappingFlowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Settings(storage=Path(temporary.name), min_free_bytes=0, neural_enabled=False)
        self.app = create_app(self.settings)
        self.client = ASGITestClient(self.app)
        self.addCleanup(self.client.close)
        self.owner = self.workspace()

    def workspace(self):
        response = self.client.get("/modelos")
        self.assertEqual(response.status_code, 200, response.text)
        return re.search(r"ocrsd_workspace=([^;]+)", response.headers["set-cookie"]).group(1)

    def hx(self, method, path, *, cookie=None, **kwargs):
        headers = {"HX-Request": "true", "Cookie": f"ocrsd_workspace={cookie or self.owner}",
                   **kwargs.pop("headers", {})}
        return self.client.request(method, path, headers=headers, **kwargs)

    def import_model(self, label="Certidão para análise", template=TEMPLATE):
        response = self.hx("POST", "/modelos", data={"label": label, "mapping_id": "default"},
                           files={"pdf": (template.name, template.read_bytes(), "application/pdf")})
        self.assertEqual(response.status_code, 200, response.text)
        return re.fullmatch(r"/modelos/([0-9a-f]{32})/editor", response.headers["hx-redirect"]).group(1)

    def editor_state(self, model_id):
        response = self.hx("GET", f"/modelos/{model_id}/estado.json")
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def save(self, model_id, state, definition):
        return self.hx("POST", f"/modelos/{model_id}/mapeamento",
                       data={"definition": json.dumps(definition), "revision": state["revision"], "layout": ""})

    def run_worker(self):
        environment = {**os.environ,
                       **{f"OCRSD_{key.upper()}": str(value) for key, value in self.settings.model_dump().items()},
                       "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        result = subprocess.run([sys.executable, "-m", "src.web.worker", "--once"], cwd=ROOT,
                                env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_background_analysis_review_and_workspace_isolation(self):
        model_id = self.import_model()
        page = self.hx("GET", f"/modelos/{model_id}/editor")
        self.assertIn("Mapear com IA", page.text)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/mapear-ia", json={"page": 2}).status_code, 422)
        queued = self.hx("POST", f"/modelos/{model_id}/mapear-ia", json={"page": 1})
        self.assertEqual(queued.status_code, 202, queued.text)
        job = queued.json()
        self.assertEqual(job["status"], "queued")
        self.assertIsNone(job["proposal"])
        job_id = job["id"]
        other = self.workspace()
        self.assertEqual(self.hx("GET", f"/modelos/{model_id}/mapear-ia/{job_id}", cookie=other).status_code, 404)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/mapear-ia", cookie=other,
                                 json={"page": 1}).status_code, 404)
        self.run_worker()
        complete = self.hx("GET", f"/modelos/{model_id}/mapear-ia/{job_id}")
        self.assertEqual(complete.status_code, 200, complete.text)
        result = complete.json()
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        proposal = result["proposal"]
        self.assertTrue(proposal["fields"])
        self.assertEqual(proposal["provenance"]["reviewed"], False)
        self.assertEqual(proposal["provenance"]["provider"], "none")
        width, height = proposal["page_size_px"]
        for item in proposal["fields"]:
            x1, y1, x2, y2 = item["bbox"]
            self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
            self.assertEqual(item["suggestion"]["status"], "suggested")
        item = next(field for field in proposal["fields"] if field["data_type"] in {"text", "person_name"})
        pending = {**proposal, "fields": [item]}
        state = self.editor_state(model_id)
        saved = self.save(model_id, state, pending)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 422)
        reviewed = {**pending, "fields": [{key: value for key, value in item.items() if key != "suggestion"}],
                    "provenance": {**pending["provenance"], "reviewed": True}}
        state = self.editor_state(model_id)
        self.assertEqual(self.save(model_id, state, reviewed).status_code, 200)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 200)
        generated = self.client.post("/api/v1/jobs", headers={"X-Workspace-Token": self.owner},
                                     json={"document_type": model_id, "mapping_id": "default",
                                           "fields": [item["field_id"]], "count": 1, "seed": 41,
                                           "writing": {"model": "font", "handwriting": False,
                                                       "text_handwriting": False}})
        self.assertEqual(generated.status_code, 202, generated.text)
        generation_id = generated.json()["id"]
        self.run_worker()
        generation = self.app.state.store.get(generation_id)
        self.assertEqual(generation["status"], "succeeded", generation["error"])
        output = self.app.state.store.directory(generation_id)
        truth = json.loads((output / "synthetic_0001.json").read_text())
        self.assertEqual(truth["generation"]["seed"], 41)
        self.assertEqual(truth["fields"][0]["field_id"], item["field_id"])
        self.assertTrue((output / "synthetic_0001.pdf").read_bytes().startswith(b"%PDF"))
        self.assertEqual(self.hx("POST", "/modelos/certidao_nascimento/mapear-ia", json={"page": 1}).status_code, 403)

    def test_custom_semantic_review_publication_and_generation(self):
        model_id = self.import_model("Valores compartilhados")
        key = "custom.codigo_local"
        spec = {"key": key, "label": "Código local", "description": "Identificador sintético do cartório",
                "data_type": "text", "generator": {"type": "choices", "values": ["A1", "B2"]}}
        created = self.hx("POST", f"/modelos/{model_id}/definicoes-semanticas", json=spec)
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["status"], "approved")
        other = self.workspace()
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/definicoes-semanticas", cookie=other,
                                 json={**spec, "key": "custom.outra"}).status_code, 404)
        state = self.editor_state(model_id)
        fields = [{"field_id": field_id, "label": label, "data_type": "text", "semantic": key, "bbox": box}
                  for field_id, label, box in (("codigo", "Código", [100, 100, 600, 160]),
                                                ("codigo_repetido", "Código repetido", [100, 200, 600, 260]))]
        fields[0]["suggestion"] = {"status": "suggested", "candidate_id": "c001"}
        pending = {**state["definition"], "fields": fields,
                   "provenance": {"method": "ocrsd-hybrid", "reviewed": False}}
        saved = self.save(model_id, state, pending)
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 422)
        state = self.editor_state(model_id)
        self.assertEqual(state["definition"]["semantic_definitions"][key], spec)
        fields[0].pop("suggestion")
        approved = {**state["definition"], "fields": fields,
                    "provenance": {"method": "ocrsd-hybrid", "reviewed": True}}
        self.assertEqual(self.save(model_id, state, approved).status_code, 200)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 200)
        generated = self.client.post("/api/v1/jobs", headers={"X-Workspace-Token": self.owner},
                                     json={"document_type": model_id, "mapping_id": "default",
                                           "fields": ["codigo", "codigo_repetido"], "count": 1, "seed": 123,
                                           "writing": {"model": "font", "handwriting": False,
                                                       "text_handwriting": False}})
        self.assertEqual(generated.status_code, 202, generated.text)
        job_id = generated.json()["id"]
        self.run_worker()
        row = self.app.state.store.get(job_id)
        self.assertEqual(row["status"], "succeeded", row["error"])
        truth = json.loads((self.app.state.store.directory(job_id) / "synthetic_0001.json").read_text())
        self.assertEqual(truth["fields"][0]["value"], truth["fields"][1]["value"])
        self.assertEqual(truth["semantic_record"][key], truth["fields"][0]["value"])
        self.assertEqual(truth["generation"]["seed"], 123)
        self.assertTrue((self.app.state.store.directory(job_id) / "synthetic_0001.pdf").read_bytes().startswith(b"%PDF"))

    def test_analysis_provenance_uses_enqueue_mapping_revision(self):
        model_id = self.import_model("Certidão com revisão concorrente")
        queued_state = self.editor_state(model_id)
        queued_version = queued_state["model"]["version"]
        queued_revision = queued_state["mapping_revision"]
        queued = self.hx("POST", f"/modelos/{model_id}/mapear-ia", json={"page": 1})
        self.assertEqual(queued.status_code, 202, queued.text)
        job_id = queued.json()["id"]

        changed = {**queued_state["definition"], "fields": [
            {"field_id": "campo_novo", "label": "Campo novo", "data_type": "text", "bbox": [100, 100, 600, 160]}]}
        saved = self.save(model_id, queued_state, changed)
        self.assertEqual(saved.status_code, 200, saved.text)
        current_state = self.editor_state(model_id)
        self.assertGreater(current_state["model"]["version"], queued_version)
        self.assertNotEqual(current_state["mapping_revision"], queued_revision)

        self.run_worker()
        response = self.hx("GET", f"/modelos/{model_id}/mapear-ia/{job_id}")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        provenance = result["proposal"]["provenance"]
        self.assertEqual(provenance["model_version"], queued_version)
        self.assertEqual(provenance["mapping_revision"], queued_revision)
        self.assertNotEqual(provenance["model_version"], current_state["model"]["version"])
        self.assertNotEqual(provenance["mapping_revision"], current_state["mapping_revision"])

    def analyze(self, model_id):
        queued = self.hx("POST", f"/modelos/{model_id}/mapear-ia", json={"page": 1})
        self.assertEqual(queued.status_code, 202, queued.text)
        job = queued.json()
        self.assertEqual(self.editor_state(model_id)["analysis"],
                         {name: job[name] for name in ("id", "status", "progress", "stage", "error")})
        self.run_worker()
        response = self.hx("GET", f"/modelos/{model_id}/mapear-ia/{job['id']}")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["status"], "succeeded", result.get("error"))
        self.assertEqual(result["stage"], "Proposta pronta para revisão")
        self.assertFalse(result["stale"])
        self.assertEqual(self.editor_state(model_id)["analysis"]["status"], "succeeded")
        return result

    @staticmethod
    def reviewed_provenance(proposal, accepted, rejected, reviewed=True):
        return {**{name: proposal["provenance"].get(name) for name in PROVENANCE},
                "reviewed": reviewed, "accepted": accepted, "rejected": rejected}

    def generate(self, model_id, fields, seed, count=1):
        response = self.client.post("/api/v1/jobs", headers={"X-Workspace-Token": self.owner},
                                    json={"document_type": model_id, "mapping_id": "default", "fields": fields,
                                          "count": count, "seed": seed,
                                          "writing": {"model": "font", "handwriting": False, "text_handwriting": False}})
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["id"]
        self.run_worker()
        row = self.app.state.store.get(job_id)
        self.assertEqual(row["status"], "succeeded", row["error"])
        output = self.app.state.store.directory(job_id)
        manifest = json.loads((output / "dataset.json").read_text())
        return [json.loads((output / f"synthetic_{index:04d}.json").read_text()) for index in range(1, count + 1)], manifest

    def test_death_declaration_review_publication_and_consistent_generation(self):
        self.assertNotIn("Mapear com IA", self.hx("GET", "/modelos/certidao_nascimento/editor").text)
        self.assertEqual(self.hx("POST", "/modelos/certidao_nascimento/mapear-ia", json={"page": 1}).status_code, 403)
        blank = self.hx("POST", "/modelos/em-branco", data={"label": "Composto", "size": "a4", "orientation": "portrait"})
        blank_id = blank.headers["hx-redirect"].split("/")[2]
        self.assertNotIn("Mapear com IA", self.hx("GET", f"/modelos/{blank_id}/editor").text)
        refused = self.hx("POST", f"/modelos/{blank_id}/mapear-ia", json={"page": 1})
        self.assertEqual(refused.status_code, 422)
        self.assertIn("Documentos compostos", refused.text)

        model_id = self.import_model("Declaração de óbito", DEATH)
        self.assertIn("Mapear com IA", self.hx("GET", f"/modelos/{model_id}/editor").text)
        self.assertIsNone(self.editor_state(model_id)["analysis"])
        result = self.analyze(model_id)
        self.assertIsNotNone(self.app.state.store.analysis_row(result["id"])["client"])
        proposal = result["proposal"]
        state = self.editor_state(model_id)
        self.assertEqual(proposal["page_size_px"], state["definition"]["page_size_px"])
        self.assertEqual(proposal["provenance"]["provider"], "none")
        by_key = {field["semantic"] if isinstance(field.get("semantic"), str) else None: field
                  for field in proposal["fields"] if field["suggestion"]["match"] == "matched"}
        self.assertTrue(set(ACCEPTED) <= set(by_key), sorted(key for key in by_key if key))
        accepted = [{name: value for name, value in by_key[key].items() if name != "suggestion"} for key in ACCEPTED]
        pending = next(field for field in proposal["fields"] if field["suggestion"]["match"] == "new")
        rejected = len(proposal["fields"]) - len(accepted) - 1
        draft = {**state["definition"], "fields": [*accepted, pending],
                 "provenance": self.reviewed_provenance(proposal, len(accepted), rejected, reviewed=False)}
        self.assertEqual(self.save(model_id, state, draft).status_code, 200)
        blocked = self.hx("POST", f"/modelos/{model_id}/publicar")
        self.assertEqual(blocked.status_code, 422)
        self.assertIn("Revise todas as sugestões", blocked.text)
        state = self.editor_state(model_id)
        self.assertEqual(state["definition"]["fields"][-1]["suggestion"]["status"], "suggested")

        reviewed = {**state["definition"], "fields": accepted,
                    "provenance": self.reviewed_provenance(proposal, len(accepted), rejected + 1)}
        self.assertEqual(self.save(model_id, state, reviewed).status_code, 200)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/publicar").status_code, 200)
        saved = self.editor_state(model_id)["definition"]
        self.assertFalse(any("suggestion" in field for field in saved["fields"]))
        self.assertEqual(saved["provenance"]["analysis_id"], result["id"])
        self.assertEqual(saved["provenance"]["method"], "ocrsd-hybrid")
        self.assertIs(saved["provenance"]["reviewed"], True)
        self.assertEqual(self.hx("POST", f"/modelos/{model_id}/mapear-ia", json={"page": 1}).status_code, 422)

        wizard = self.hx("GET", f"/geracao?modelo={model_id}")
        self.assertEqual(wizard.status_code, 200, wizard.text)
        self.assertIn(model_id, wizard.text)
        field_ids = {key: by_key[key]["field_id"] for key in ACCEPTED}
        truths, manifest = self.generate(model_id, list(field_ids.values()), 314, count=2)
        provenance = manifest["reproduction"]["mapping_snapshot"]["provenance"]
        self.assertEqual((provenance["analysis_id"], provenance["reviewed"]), (result["id"], True))
        for truth in truths:
            values = {item["field_id"]: item for item in truth["fields"]}
            record = truth["semantic_record"]
            for key, field_id in field_ids.items():
                self.assertEqual(values[field_id]["semantic"]["key"], key)
                self.assertEqual(values[field_id]["semantic"]["value"], record[key])
            born, died = (date.fromisoformat(record[key]) for key in ("deceased.birth_date", "death.date"))
            self.assertEqual(int(record["deceased.age"]), died.year - born.year - ((died.month, died.day) < (born.month, born.day)))
            self.assertIn(record["deceased.sex"], {"M", "F"})
            self.assertIn((record["deceased.residence.municipality"], record["deceased.residence.uf"]), CITIES)
            sex = values[field_ids["deceased.sex"]]["elements"]
            self.assertEqual([element["value"] for element in sex if element["checked"]], [record["deceased.sex"]])

    def test_new_semantic_definition_from_proposal_is_reused_by_next_analysis(self):
        first = self.import_model("Declaração de óbito", DEATH)
        proposal = self.analyze(first)["proposal"]
        new = next(field for field in proposal["fields"] if field["label"] == "Naturalidade")
        self.assertEqual(new["suggestion"]["match"], "new")
        suggested = new["suggestion"]["proposal"]
        spec = {"key": suggested["key"], "label": suggested["label"], "description": suggested["description"],
                "data_type": "text", "generator": {"type": "choices", "values": ["Florianópolis/SC", "Joinville/SC"]}}
        created = self.hx("POST", f"/modelos/{first}/definicoes-semanticas", json=spec)
        self.assertEqual(created.status_code, 201, created.text)
        state = self.editor_state(first)
        field = {**{name: value for name, value in new.items() if name != "suggestion"}, "semantic": spec["key"]}
        definition = {**state["definition"], "fields": [field],
                      "provenance": self.reviewed_provenance(proposal, 1, len(proposal["fields"]) - 1)}
        self.assertEqual(self.save(first, state, definition).status_code, 200)
        self.assertEqual(self.hx("POST", f"/modelos/{first}/publicar").status_code, 200)
        self.assertEqual(self.editor_state(first)["definition"]["semantic_definitions"][spec["key"]], spec)

        second = self.import_model("Outra declaração de óbito", DEATH)
        self.assertIn(spec["key"], {option["key"] for option in self.editor_state(second)["semantic_options"]})
        reused = next(item for item in self.analyze(second)["proposal"]["fields"] if item["label"] == "Naturalidade")
        self.assertEqual(reused["semantic"], spec["key"])
        self.assertEqual(reused["suggestion"]["match"], "matched")
        state = self.editor_state(second)
        accepted = {name: value for name, value in reused.items() if name != "suggestion"}
        self.assertEqual(self.save(second, state, {**state["definition"], "fields": [accepted]}).status_code, 200)
        self.assertEqual(self.hx("POST", f"/modelos/{second}/publicar").status_code, 200)
        truth = self.generate(second, [accepted["field_id"]], 7)[0][0]
        self.assertIn(truth["fields"][0]["value"], spec["generator"]["values"])


if __name__ == "__main__":
    unittest.main()
