import json
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field as dataclass_field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.mapping_assist.matching import MEDICAL_GENERATOR, allowed_keys, match_candidates, registry_index
from src.mapping_assist.proposal import accept_suggestion, build_proposal
from src.mapping_assist.providers import (AssistantError, NullAssistant, OpenAICompatibleAssistant, ProviderStatus,
                                          assistant_from_settings)
from src.semantic import VERSION
from src.field_rules import FIELD_TYPES
from src.semantic.custom import validate_custom_definition
from src.semantic.registry import KINDS
from src.semantic.resolve import Binding, normalize, resolve_field
from src.web.services import mapping_issues, validate_mapping
from src.web.settings import Settings

try:
    from src.mapping_assist.analysis import FieldCandidate, PageAnalysis
except ImportError:
    @dataclass(frozen=True)
    class FieldCandidate:
        id: str
        label: str
        label_bbox: tuple[int, int, int, int] | None
        region: tuple[int, int, int, int]
        kind: str
        elements: tuple[dict, ...] = ()
        section: str | None = None
        section_title: str | None = None
        neighbors: tuple[str, ...] = ()
        number: str | None = None
        source: str = "text_layer"
        confidence: float | None = None

    @dataclass(frozen=True)
    class PageAnalysis:
        page: int
        dpi: int
        page_size_px: tuple[int, int]
        unit: float
        source: str
        candidates: tuple[FieldCandidate, ...]
        warnings: tuple[str, ...] = ()
        timings: dict = dataclass_field(default_factory=dict)

ROOT = Path(__file__).resolve().parent.parent
DNV = ROOT / "documents" / "dnv" / "mappings"
CERTIDAO = ROOT / "documents" / "certidao_nascimento" / "certidao_nascimento_fields.json"
DO = "declaracao_obito"
NEW = None
DNV_EXPECTED = {
    "1": "newborn.name", "5": "newborn.sex", "raca_cor": NEW, "2": "birth.date", "12": NEW, "7": NEW, "comprimento": NEW,
    "8": NEW, "9": NEW, "anomalia": NEW, "10": NEW, "11": "birth.establishment.name", "14": "birth.place.street",
    "15": "birth.place.number", "uf": "birth.place.uf", "17": "birth.place.municipality", "16": "birth.place.neighborhood",
    "61": "mother.cns", "18": "mother.name", "66": NEW, "19": NEW, "64": "mother.occupation", "22": "mother.naturalidade",
    "23": NEW, "raca_cor_22": NEW, "20": "mother.birth_date", "21": "mother.age", "25": "mother.residence.street",
    "59": "mother.residence.complement", "26": "mother.residence.number", "60": "mother.residence.cep",
    "27": "mother.residence.neighborhood", "municipio": "mother.residence.municipality", "29": "mother.residence.uf",
    "57": "guardian.name", "idade": "guardian.age", "n_perdas_fetais_abortos": "mother.obstetric.fetal_losses",
    "n_partos_vaginais": "mother.obstetric.vaginal_births", "n_cesareas": "mother.obstetric.cesareans",
    "n_nascidos_vivos": "mother.obstetric.live_births", "gestacoes_anteriores": "mother.obstetric.previous_pregnancies",
    "40": NEW, "nascimento_assistido_por": NEW, "tipo_gravidez": NEW, "tipo_parto": NEW, "parto_induzido": NEW,
    "cesarea_ocorreu_antes_do_trabalho_de_parto_iniciar": NEW, "n_semanas_gestacao": "pregnancy.weeks",
    "n_consultas_pre_natal": "pregnancy.prenatal_visits", "mes_que_comecou_pre_natal": "pregnancy.prenatal_start_month",
    "metodo_utilizado_para_estimar": NEW, "47": "filling.responsible.name", "48": NEW, "46": "filling.date",
    "51": "filling.responsible.issuer", "50": "filling.responsible.document_number", "49": NEW,
    "52": "registration.office.name", "registro": "registration.number", "54": "registration.date",
    "55": "registration.office.municipality", "56": "registration.office.uf", "67": "birth.time",
    "codigo_cnes": "birth.establishment.cnes", "data_ultima_menstruacao": "pregnancy.last_menstrual_period", "cpf": "mother.cpf",
    "CEP": "birth.place.cep",
}
CERTIDAO_EXPECTED = {
    "cpf": "newborn.cpf", "matricula": "registration.matricula", "data_nascimento_por_extenso": "birth.date",
    "dia_nascimento": "birth.date", "mes_nascimento": "birth.date", "ano_nascimento": "birth.date", "hora_nascimento": "birth.time",
    "naturalidade": "newborn.naturalidade", "municipio_registro_uf": "registration.place.summary",
    "local_municipio_nascimento_uf": "birth.place.summary", "sexo": "newborn.sex", "filiacao": "parents.names",
    "avos": "grandparents.names", "data_registro_por_extenso": "registration.date", "numero_dnv": "registration.dnv_number",
    "gemeos": "birth.twins", "nome_matricula_gemeos": "twin.name_matricula", "averbacoes_anotacoes_acrescer": NEW,
    "anotacoes_cadastro": NEW,
}
SEX = (("M - Masc.", "M"), ("F - Fem.", "F"), ("I - Ignorado", "I"))
YES_NO = (("Sim", "1"), ("Não", "2"), ("Ignorado", "9"))
DO_LABELS = [
    ("Tipo de óbito", "I", "checkbox", (("Fetal", "1"), ("Não Fetal", "2")), NEW),
    ("Data do óbito", "I", "digit_sequence", 8, "death.date"), ("Hora", "I", "digit_sequence", 4, "death.time"),
    ("Cartão SUS", "I", "digit_sequence", 15, NEW), ("Naturalidade", "I", "text", 0, NEW),
    ("Nome do Falecido", "I", "text", 0, "deceased.name"), ("Nome do Pai", "I", "text", 0, "deceased.father_name"),
    ("Nome da Mãe", "I", "text", 0, "deceased.mother_name"),
    ("Data de nascimento", "I", "digit_sequence", 8, "deceased.birth_date"), ("Idade", "I", "text", 0, "deceased.age"),
    ("Sexo", "I", "checkbox", SEX, "deceased.sex"),
    ("Raça/Cor", "I", "checkbox", (("Branca", "1"), ("Preta", "2"), ("Amarela", "3"), ("Parda", "4"), ("Indígena", "5")), NEW),
    ("Situação conjugal", "I", "checkbox", (("Solteiro", "1"), ("Casado", "2"), ("Viúvo", "3")), NEW),
    ("Escolaridade (última série concluída)", "I", "checkbox", (("Sem escolaridade", "0"), ("Fundamental I", "1")), NEW),
    ("Ocupação habitual", "I", "text", 0, "deceased.occupation"),
    ("Logradouro (rua, praça, avenida, etc.)", "II", "text", 0, "deceased.residence.street"),
    ("Número", "II", "text", 0, "deceased.residence.number"), ("Complemento", "II", "text", 0, NEW),
    ("CEP", "II", "digit_sequence", 8, "deceased.residence.cep"),
    ("Bairro/Distrito", "II", "text", 0, "deceased.residence.neighborhood"),
    ("Município de residência", "II", "text", 0, "deceased.residence.municipality"),
    ("UF", "II", "text", 0, "deceased.residence.uf"),
    ("Local de ocorrência do óbito", "III", "checkbox", (("Hospital", "1"), ("Domicílio", "3"), ("Via pública", "4")), NEW),
    ("Estabelecimento", "III", "text", 0, "death.place.establishment.name"), ("Código CNES", "III", "digit_sequence", 7, NEW),
    ("Endereço da ocorrência, se fora do estabelecimento ou da residência (rua, praça, avenida, etc)", "III", "text", 0, NEW),
    ("CEP", "III", "digit_sequence", 8, NEW), ("Bairro/Distrito", "III", "text", 0, NEW),
    ("Município de ocorrência", "III", "text", 0, "death.place.municipality"), ("UF", "III", "text", 0, "death.place.uf"),
    ("Idade (anos)", "IV", "text", 0, NEW), ("Número de filhos tidos", "IV", "digit_sequence", 2, NEW),
    ("Nº de semanas de gestação", "IV", "digit_sequence", 2, NEW),
    ("Tipo de gravidez", "IV", "checkbox", (("Única", "1"), ("Dupla", "2")), NEW),
    ("Peso ao nascer", "IV", "digit_sequence", 4, NEW),
    ("Número da Declaração de Nascido Vivo", "IV", "digit_sequence", 11, NEW),
    ("A morte ocorreu", "V", "checkbox", (("Na gravidez", "1"), ("No parto", "2")), NEW),
    ("Necrópsia?", "V", "checkbox", YES_NO, NEW), ("Causas da morte - Parte I", "V", "text", 0, NEW),
    ("Parte II", "V", "text", 0, NEW), ("Nome do Médico", "VI", "text", 0, "certifier.name"),
    ("CRM", "VI", "text", 0, "certifier.registration"),
    ("Óbito atestado por Médico", "VI", "checkbox", (("Assistente", "1"), ("Substituto", "2"), ("IML", "3")), NEW),
    ("Município e UF do SVO ou IML", "VI", "text", 0, NEW), ("Meio de contato (telefone, fax, e-mail, etc.)", "VI", "text", 0, NEW),
    ("Data do atestado", "VI", "digit_sequence", 8, NEW),
    ("Tipo", "VII", "checkbox", (("Acidente", "1"), ("Suicídio", "2"), ("Homicídio", "3")), NEW),
    ("Acidente do trabalho", "VII", "checkbox", YES_NO, NEW),
    ("Descrição sumária do evento, incluindo o tipo de local de ocorrência", "VII", "text", 0, NEW),
    ("Logradouro (rua, praça, avenida, etc.)", "VII", "text", 0, NEW),
    ("Cartório", "VIII", "text", 0, "death_registration.office.name"),
    ("Registro", "VIII", "digit_sequence", 6, "death_registration.number"),
    ("Data", "VIII", "digit_sequence", 8, "death_registration.date"),
    ("Município", "VIII", "text", 0, "death_registration.office.municipality"),
    ("UF", "VIII", "text", 0, "death_registration.office.uf"), ("Declarante", "IX", "text", 0, NEW),
    ("Testemunhas", "IX", "text", 0, NEW),
]
CLINICAL = {"Número de filhos tidos", "Nº de semanas de gestação", "Tipo de gravidez", "Peso ao nascer", "A morte ocorreu",
            "Necrópsia?", "Causas da morte - Parte I", "Parte II", "Tipo", "Acidente do trabalho"}


def _boxes(count: int, x: int = 100, y: int = 100) -> tuple[dict, ...]:
    return tuple({"index": index, "bbox": (x + index * 40, y, x + index * 40 + 30, y + 40)} for index in range(count))


def _options(options, x: int = 100, y: int = 100) -> tuple[dict, ...]:
    return tuple({"label": label, "code": code, "bbox": (x + index * 120, y, x + index * 120 + 30, y + 30)}
                 for index, (label, code) in enumerate(options))


def candidate(identifier: str, label: str, section: str | None = None, kind: str = "text", elements=(), region=None,
              **extra) -> FieldCandidate:
    elements = _boxes(elements) if kind == "digit_sequence" and isinstance(elements, int) else elements
    elements = _options(elements) if kind == "checkbox" else () if isinstance(elements, int) else elements
    return FieldCandidate(id=identifier, label=label, label_bbox=None, region=region or (80, 80, 1500, 160), kind=kind,
                          elements=tuple(elements), section=section, **extra)


def _dnv_candidates() -> tuple[list[FieldCandidate], dict]:
    printed = json.loads((DNV / "20260831.json").read_text(encoding="utf-8"))
    norma = json.loads((DNV / "mapeamento_seguindo_norma.json").read_text(encoding="utf-8"))["fields"]

    def overlap(first, second) -> int:
        return max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))

    found = []
    for field in printed["fields"]:
        if field["field_id"] not in DNV_EXPECTED:
            continue
        section = max(norma, key=lambda item: overlap(field["bbox"], item["bbox"]))["bloco"]
        kind = field.get("data_type") if field.get("data_type") in {"checkbox", "digit_sequence"} else "text"
        elements = tuple({"index": element.get("index", position), "bbox": tuple(element["bbox"])} if kind == "digit_sequence"
                         else {"label": element.get("label"), "code": None, "bbox": tuple(element["bbox"])}
                         for position, element in enumerate(field.get("elements") or []))
        label = field.get("description") or field["label"]
        found.append(FieldCandidate(field["field_id"], label, None, tuple(field["bbox"]), kind,
                                    elements if kind != "text" else (), section))
    return found, printed


def _score(matches, expected: dict) -> tuple[float, list, list]:
    known = [match for match in matches if expected[match.candidate_id] is not None]
    recall = sum(match.status == "matched" and match.key == expected[match.candidate_id] for match in known) / len(known)
    wrong = [(match.candidate_id, match.key) for match in matches
             if match.status == "matched" and match.key != expected[match.candidate_id]]
    return recall, wrong, known


class ResolverRegressionTests(unittest.TestCase):
    def test_death_certificate_sections(self):
        for label, block in (("Número de filhos tidos", "IV"), ("Número de filhos tidos", None), ("Idade", "IV"),
                             ("Número", "IV"), ("Município", "VII"), ("Logradouro", "VII"), ("Declarante", "IX"),
                             ("Município e UF do SVO ou IML", "VI"), ("Bairro/Distrito", "III"), ("Peso ao nascer", "IV")):
            with self.subTest(label=label, block=block):
                self.assertIsNone(resolve_field({"label": label, "bloco": block}, DO))
        for label, block, key in (("Hora", "I", "death.time"), ("Data do óbito", "I", "death.date"), ("Número", "II", "deceased.residence.number"),
                                  ("Data de nascimento", "I", "deceased.birth_date"), ("Nome do Médico", "VI", "certifier.name"),
                                  ("Data", "VIII", "death_registration.date")):
            self.assertEqual(resolve_field({"label": label, "bloco": block}, DO), Binding(key), label)
        self.assertEqual(resolve_field({"label": "N nascidos vivos", "bloco": "V"}, "dnv"), Binding("mother.obstetric.live_births"))


class MatchingTests(unittest.TestCase):
    def test_dnv_printed_labels(self):
        candidates, _ = _dnv_candidates()
        self.assertEqual(len(candidates), len(DNV_EXPECTED))
        matches = match_candidates(candidates, "dnv")
        recall, wrong, _ = _score(matches, DNV_EXPECTED)
        self.assertGreaterEqual(recall, 0.95)
        self.assertLessEqual(len(wrong), 2, wrong)
        by_id = {match.candidate_id: match for match in matches}
        self.assertEqual(by_id["5"].method, "rules")
        self.assertEqual(by_id["raca_cor"].status, "new")
        self.assertTrue(by_id["raca_cor"].proposal["generator"].startswith("lista de opções detectadas"))
        self.assertTrue(by_id["8"].proposal["generator"].startswith(MEDICAL_GENERATOR))

    def test_certidao_labels(self):
        definition = json.loads(CERTIDAO.read_text(encoding="utf-8"))
        candidates = [candidate(field["field_id"], field["label"]) for field in definition["fields"]]
        matches = match_candidates(candidates, "certidao_nascimento")
        recall, wrong, _ = _score(matches, CERTIDAO_EXPECTED)
        self.assertGreaterEqual(recall, 0.95)
        self.assertEqual(wrong, [])

    def test_death_certificate_labels(self):
        candidates = [candidate(str(index), label, section, kind, elements)
                      for index, (label, section, kind, elements, _) in enumerate(DO_LABELS)]
        expected = {str(index): key for index, (*_, key) in enumerate(DO_LABELS)}
        matches = match_candidates(candidates, "DO")
        recall, wrong, _ = _score(matches, expected)
        self.assertEqual((recall, wrong), (1.0, []))
        for match, (label, section, *_ , key) in zip(matches, DO_LABELS):
            with self.subTest(label=label, section=section):
                if key is None:
                    self.assertNotEqual(match.status, "matched")
                if label in CLINICAL:
                    self.assertEqual(match.status, "new")
                    self.assertTrue(match.proposal["generator"].startswith(MEDICAL_GENERATOR), match.proposal)
        by_label = {(label, section): match for match, (label, section, *_) in zip(matches, DO_LABELS)}
        cause = by_label[("Causas da morte - Parte I", "V")].proposal
        self.assertEqual(cause["key"], "custom.death.cause.parte_i")
        self.assertEqual(cause["kind"], "text")
        self.assertIn("filhos", by_label[("Número de filhos tidos", "IV")].reason)
        self.assertIn("seção IX não tem chaves", by_label[("Declarante", "IX")].reason)
        self.assertIn("raca", normalize(by_label[("Raça/Cor", "I")].reason))
        self.assertEqual(by_label[("Complemento", "II")].proposal["key"], "custom.deceased.residence.complemento")
        self.assertEqual(by_label[("Tipo de óbito", "I")].status, "new")

    def test_thresholds_and_ambiguity(self):
        uf = match_candidates([candidate("uf", "UF")], DO)[0]
        self.assertEqual((uf.status, uf.key), ("ambiguous", None))
        self.assertEqual({key for key, _ in uf.alternatives},
                         {"deceased.residence.uf", "death.place.uf", "death_registration.office.uf"})
        birth = candidate("nasc", "Data de Nascimento")
        self.assertEqual(match_candidates([birth], "dnv")[0].status, "ambiguous")
        relaxed = match_candidates([birth], "dnv", min_margin=0.1)[0]
        self.assertEqual((relaxed.status, relaxed.key, relaxed.method), ("matched", "birth.date", "ngram"))
        unknown = match_candidates([candidate("x", "Quantidade de parafusos")], "dnv")[0]
        self.assertEqual(unknown.status, "new")
        self.assertIn("limiar", unknown.reason)
        checkbox = match_candidates([candidate("c", "Sexo", "I", "checkbox", SEX)], DO)[0]
        self.assertEqual((checkbox.status, checkbox.key), ("matched", "deceased.sex"))
        comb = match_candidates([candidate("d", "Nome do Falecido", "I", "digit_sequence", 8)], DO)[0]
        self.assertNotEqual(comb.status, "matched")
        self.assertEqual(match_candidates([candidate("e", "   ")], DO)[0].status, "new")


class FakeModel(BaseHTTPRequestHandler):
    mode = "valid"
    delay = 0.0
    get_delay = 0.0
    posts = 0

    def log_message(self, *args):
        pass

    def _send(self, status: int, payload) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        time.sleep(type(self).get_delay)
        self._send(200, {"data": [{"id": "fake"}]})

    def do_POST(self):
        type(self).posts += 1
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        time.sleep(type(self).delay)
        if type(self).mode == "error":
            return self._send(500, {"error": "boom"})
        if type(self).mode == "drip":
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "4096")
                self.end_headers()
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        payload = json.loads(request["messages"][1]["content"])
        if type(self).mode == "invalid":
            content = "isto não é JSON"
        elif "proposals" in request["messages"][0]["content"]:
            content = json.dumps({"proposals": [{"id": item["id"], "key": "custom.death.cause.part_i", "label": "Causa — parte I",
                                                 "generator": "inventar doença", "kind": "nonsense", "data_type": "text"}
                                                for item in payload["items"]]})
        elif type(self).mode == "outside":
            content = json.dumps({"answers": [{"id": item["id"], "key": "newborn.name", "reason": "x"} for item in payload["items"]]})
        else:
            content = "<think>ok</think>" + json.dumps({"answers": [
                {"id": item["id"], "key": item["options"][-1]["key"] if item["label"] != "Nada" else None,
                 "reason": "contexto da seção"} for item in payload["items"]] + [{"id": "intruso", "key": "deceased.name"}]})
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}]})


class ProviderTests(unittest.TestCase):
    def setUp(self):
        FakeModel.mode, FakeModel.delay, FakeModel.get_delay, FakeModel.posts = "valid", 0.0, 0.0, 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.cache = tempfile.TemporaryDirectory()
        self.addCleanup(self.cache.cleanup)

    def assistant(self, **options) -> OpenAICompatibleAssistant:
        return OpenAICompatibleAssistant(self.url, "fake", **{"timeout": 2, **options})

    def items(self):
        return [candidate("uf", "UF"), candidate("nada", "Nada")]

    def alternatives(self):
        return {"uf": (("deceased.residence.uf", 1.0), ("death.place.uf", 1.0)), "nada": (("deceased.name", 0.4),)}

    def test_null_assistant(self):
        null = NullAssistant()
        self.assertEqual(null.status(), ProviderStatus(False, "none", None, "Assistente de IA desativado"))
        self.assertEqual(null.disambiguate(DO, self.items(), self.alternatives()), {})
        self.assertEqual(match_candidates(self.items(), DO, null), match_candidates(self.items(), DO))
        self.assertIsInstance(assistant_from_settings(Settings(), None), NullAssistant)

    def test_valid_answers_are_restricted_to_candidates(self):
        assistant = self.assistant()
        self.assertTrue(assistant.status().available)
        self.assertEqual(assistant.disambiguate(DO, self.items(), self.alternatives()),
                         {"uf": ("death.place.uf", "contexto da seção"), "nada": (None, "contexto da seção")})
        FakeModel.mode = "outside"
        self.assertEqual(assistant.disambiguate(DO, self.items(), self.alternatives()), {})

    def test_invalid_json_and_timeout_degrade(self):
        FakeModel.mode = "invalid"
        assistant = self.assistant()
        with self.assertRaises(AssistantError):
            assistant.disambiguate(DO, self.items(), self.alternatives())
        self.assertEqual(FakeModel.posts, 2)
        matches = match_candidates([candidate("uf", "UF")], DO, assistant)
        self.assertEqual((matches[0].status, matches[0].key), ("ambiguous", None))
        self.assertIn("Assistente falhou", matches[0].reason)
        FakeModel.mode, FakeModel.delay = "valid", 0.6
        slow = self.assistant(timeout=0.2)
        started = time.monotonic()
        with self.assertRaises(AssistantError):
            slow.disambiguate(DO, self.items(), self.alternatives())
        self.assertLess(time.monotonic() - started, 2.0)

    def test_unavailable_server(self):
        closed = OpenAICompatibleAssistant("http://127.0.0.1:9/v1", "fake", timeout=1)
        status = closed.status()
        self.assertFalse(status.available)
        matches = match_candidates([candidate("uf", "UF")], DO, closed)
        self.assertEqual(matches[0].status, "ambiguous")
        self.assertIn("Assistente indisponível", matches[0].reason)

    def test_circuit_breaker(self):
        now = [0.0]
        FakeModel.mode = "error"
        assistant = self.assistant(clock=lambda: now[0])
        for _ in range(3):
            with self.assertRaises(AssistantError):
                assistant.disambiguate(DO, self.items(), self.alternatives())
        self.assertEqual(FakeModel.posts, 6)
        with self.assertRaisesRegex(AssistantError, "pausado"):
            assistant.disambiguate(DO, self.items(), self.alternatives())
        self.assertFalse(assistant.status().available)
        self.assertEqual(FakeModel.posts, 6)
        now[0] = 301.0
        FakeModel.mode = "valid"
        self.assertEqual(len(assistant.disambiguate(DO, self.items(), self.alternatives())), 2)

    def test_cache_hit(self):
        first = self.assistant(cache_dir=self.cache.name)
        answer = first.disambiguate(DO, self.items(), self.alternatives())
        self.assertEqual(FakeModel.posts, 1)
        second = self.assistant(cache_dir=self.cache.name)
        self.assertEqual(second.disambiguate(DO, self.items(), self.alternatives()), answer)
        self.assertEqual(FakeModel.posts, 1)
        self.assertEqual(len(list(Path(self.cache.name).glob("*.json"))), 1)

    def test_batching(self):
        items = [candidate(f"uf{index}", "UF") for index in range(45)]
        alternatives = {item.id: (("deceased.residence.uf", 1.0), ("death.place.uf", 1.0)) for item in items}
        self.assertEqual(len(self.assistant().disambiguate(DO, items, alternatives)), 45)
        self.assertEqual(FakeModel.posts, 3)

    def test_remote_urls_are_refused_unless_allowed(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleAssistant("http://10.1.2.3:11434/v1", "fake")
        with self.assertRaises(ValueError):
            OpenAICompatibleAssistant("https://api.example.com/v1", "fake")
        OpenAICompatibleAssistant("https://api.example.com/v1", "fake", allow_remote=True)
        OpenAICompatibleAssistant("http://localhost:8080/v1", "fake")
        refused = assistant_from_settings(Settings(ai_provider="openai", ai_base_url="http://192.168.0.2/v1"), None)
        self.assertIsInstance(refused, NullAssistant)
        self.assertIn("loopback", refused.status().detail)
        allowed = assistant_from_settings(Settings(ai_provider="openai", ai_base_url=self.url, ai_model="fake"), self.cache.name)
        self.assertIsInstance(allowed, OpenAICompatibleAssistant)

    def test_matching_with_assistant(self):
        items = [candidate("uf", "UF"), candidate("causa", "Causas da morte - Parte I", "V")]
        matches = {match.candidate_id: match for match in match_candidates(items, DO, self.assistant())}
        self.assertEqual((matches["uf"].status, matches["uf"].method), ("matched", "assistant"))
        self.assertIn(matches["uf"].key, {key for key, _ in matches["uf"].alternatives})
        proposal = matches["causa"].proposal
        self.assertEqual((proposal["key"], proposal["label"], proposal["kind"]), ("custom.death.cause.part_i", "Causa — parte I", "text"))
        self.assertTrue(proposal["generator"].startswith(MEDICAL_GENERATOR))

    def test_budget_skips_slow_provider_and_keeps_deterministic_results(self):
        FakeModel.delay = 1.5
        assistant = self.assistant(timeout=5)
        status = assistant.status()
        report = {}
        started = time.monotonic()
        matches = match_candidates([candidate("uf", "UF"), candidate("x", "Quantidade de parafusos")], DO, assistant,
                                   status=status, budget=0.4, report=report)
        self.assertLess(time.monotonic() - started, 1.4)
        self.assertEqual([match.status for match in matches], ["ambiguous", "new"])
        self.assertNotIn("assistant", matches[1].proposal)
        self.assertEqual(report["status"], "degraded")
        self.assertTrue(any("tempo" in warning for warning in report["warnings"]), report)

    def test_total_response_deadline(self):
        FakeModel.mode = "drip"
        assistant = self.assistant(timeout=0.5)
        started = time.monotonic()
        with self.assertRaisesRegex(AssistantError, "tempo esgotado"):
            assistant._http("POST", "/chat/completions", {})
        self.assertLess(time.monotonic() - started, 1.5)

    def test_status_is_reused_and_failure_cause_recorded(self):
        class Counting(OpenAICompatibleAssistant):
            calls = 0

            def status(self):
                type(self).calls += 1
                return super().status()

        assistant = Counting(self.url, "fake", timeout=2)
        status = assistant.status()
        match_candidates([candidate("uf", "UF")], DO, assistant, status=status)
        self.assertEqual(Counting.calls, 1)
        closed = OpenAICompatibleAssistant("http://127.0.0.1:9/v1", "fake", timeout=1)
        with self.assertRaisesRegex(AssistantError, "recusada"):
            closed._http("GET", "/models")
        report = {}
        match_candidates([candidate("uf", "UF")], DO, closed, report=report)
        self.assertEqual(report["status"], "unavailable")
        self.assertIn("recusada", report["detail"])
        FakeModel.mode = "error"
        report = {}
        match_candidates([candidate("uf", "UF")], DO, self.assistant(), report=report)
        self.assertTrue(any("HTTP 500" in warning for warning in report["warnings"]), report)

    def test_cache_key_includes_base_url_and_remote_requires_https(self):
        first = self.assistant(cache_dir=self.cache.name)
        other = OpenAICompatibleAssistant("http://localhost:1/v1", "fake", cache_dir=self.cache.name)
        messages = first._messages("x", {"a": 1})
        self.assertNotEqual(first._cache_path(messages), other._cache_path(messages))
        with self.assertRaisesRegex(ValueError, "https"):
            OpenAICompatibleAssistant("http://api.example.com/v1", "fake", allow_remote=True)

    def test_assistant_rejects_out_of_section_and_invented_keys(self):
        class Chooser:
            offered = {}

            def status(self):
                return ProviderStatus(True, "fake", "fake", "ok")

            def disambiguate(self, document, items, candidates):
                type(self).offered = dict(candidates)
                return {item.id: ("death.date", "fora da seção") for item in items}

            def propose_categories(self, document, items):
                return {item.id: {"key": "death.cause.parte_i", "kind": "nonsense", "data_type": "integer"} for item in items}

        item = candidate("d", "Data do óbito", "II")
        valid = set(allowed_keys(DO, "II", "text"))
        match = match_candidates([item], DO, Chooser())[0]
        self.assertNotEqual(match.key, "death.date")
        self.assertNotEqual(match.status, "matched")
        self.assertTrue(Chooser.offered)
        self.assertTrue(all(key in valid for options in Chooser.offered.values() for key, _ in options))
        scores = dict(registry_index().scores(item.label, valid))
        self.assertTrue(all(score == round(scores[key], 3) for key, score in match.alternatives))
        new = match_candidates([candidate("p", "Quantidade de parafusos", "II")], DO, Chooser())[0]
        self.assertTrue(new.proposal["key"].startswith("custom.deceased.residence."))
        self.assertEqual(new.proposal["kind"], "text")


class ProposalTests(unittest.TestCase):
    def analysis(self, dpi: int = 300) -> PageAnalysis:
        factor = dpi / 300
        scaled = lambda box: tuple(round(value * factor) for value in box)
        items = [
            candidate("c1", "Nome do Falecido", "I", region=scaled((100, 100, 1500, 180))),
            candidate("c2", "Sexo", "I", "checkbox", SEX, region=scaled((100, 200, 600, 260))),
            candidate("c3", "Data do óbito", "I", "digit_sequence", 8, region=scaled((90, 300, 500, 360))),
            candidate("c4", "Data", "VIII", "digit_sequence", 6, region=scaled((90, 400, 400, 460))),
            candidate("c5", "Número de filhos tidos", "IV", "digit_sequence", 2, region=scaled((90, 500, 300, 560))),
            candidate("c6", "Causas da morte - Parte I", "V", region=scaled((100, 600, 1500, 660))),
            candidate("c7", "UF", region=scaled((100, 700, 200, 760))),
            candidate("c8", "Necrópsia?", "V", "checkbox", YES_NO, region=scaled((100, 800, 600, 860))),
            candidate("c9", "Nome", "IX", region=scaled((100, 900, 600, 960))),
            candidate("c10", "Nome", "IX", region=scaled((100, 1000, 600, 1060))),
            candidate("c11", "Fora", "IX", region=(5000, 5000, 5100, 5100)),
        ]
        if dpi != 300:
            items = [replace(item, elements=tuple({**element, "bbox": scaled(element["bbox"])} for element in item.elements))
                     for item in items]
        return PageAnalysis(page=1, dpi=dpi, page_size_px=(round(2480 * factor), round(3508 * factor)), unit=12.0,
                            source="text_layer", candidates=tuple(items), warnings=("aviso de teste",), timings={"total": 0.5})

    def test_build_and_validate(self):
        analysis = self.analysis()
        matches = match_candidates(analysis, DO)
        definition = build_proposal(analysis, matches, DO, {"provider": "none", "model": None})
        validated = validate_mapping(definition, (2480, 3508))
        self.assertTrue(any("Revise todas as sugestões" in issue for issue in mapping_issues(validated)))
        fields = {field["suggestion"]["candidate_id"]: field for field in definition["fields"]}
        self.assertNotIn("c11", fields)
        self.assertEqual((definition["template_id"], definition["dpi"], definition["page_size_px"]), (DO, 300, [2480, 3508]))
        self.assertEqual((fields["c1"]["field_id"], fields["c1"]["data_type"], fields["c1"]["semantic"]),
                         ("deceased_name", "person_name", "deceased.name"))
        self.assertEqual((fields["c2"]["data_type"], fields["c2"]["semantic"]), ("checkbox", "deceased.sex"))
        self.assertEqual([element["value"] for element in fields["c2"]["elements"]], ["M", "F", "I"])
        self.assertEqual((fields["c3"]["data_type"], len(fields["c3"]["elements"]), fields["c3"]["semantic"]),
                         ("digit_sequence", 8, "death.date"))
        self.assertEqual([element["index"] for element in fields["c3"]["elements"]], list(range(8)))
        self.assertEqual((fields["c4"]["semantic"], len(fields["c4"]["elements"])), ("death_registration.date", 6))
        for identifier in ("c5", "c6", "c7", "c8", "c9"):
            self.assertNotIn("semantic", fields[identifier])
        self.assertEqual(fields["c7"]["suggestion"]["match"], "ambiguous")
        self.assertEqual(fields["c6"]["suggestion"]["proposal"]["generator"], MEDICAL_GENERATOR)
        self.assertEqual((fields["c9"]["field_id"], fields["c10"]["field_id"]), ("nome", "nome_2"))
        self.assertEqual(fields["c8"]["bloco"], "V")
        suggestion = fields["c1"]["suggestion"]
        self.assertEqual((suggestion["status"], suggestion["match"], suggestion["method"]), ("suggested", "matched", "rules"))
        self.assertEqual(set(suggestion), {"status", "candidate_id", "match", "method", "score", "alternatives", "reason", "proposal"})
        provenance = definition["provenance"]
        self.assertEqual((provenance["method"], provenance["analysis_source"], provenance["registry_version"], provenance["reviewed"]),
                         ("ocrsd-hybrid", "text_layer", VERSION, False))
        self.assertIn("aviso de teste", provenance["warnings"])
        self.assertTrue(any("c11" in warning for warning in provenance["warnings"]))
        self.assertEqual(provenance["timings"], {"total": 0.5})
        json.dumps(definition)
        reviewed = [accept_suggestion(field) for field in definition["fields"]]
        self.assertFalse(any("suggestion" in field for field in reviewed))
        chosen = accept_suggestion(fields["c7"], "death.place.uf")
        self.assertEqual(chosen["semantic"], "death.place.uf")
        self.assertNotIn("suggestion", chosen)
        with self.assertRaises(ValueError):
            accept_suggestion(fields["c7"], "nada.disso")

    def test_dpi_scaling(self):
        analysis = self.analysis(150)
        definition = build_proposal(analysis, match_candidates(analysis, DO), DO, {})
        validate_mapping(definition)
        self.assertEqual(definition["page_size_px"], [2480, 3508])
        first = next(field for field in definition["fields"] if field["suggestion"]["candidate_id"] == "c1")
        self.assertEqual(first["bbox"], [100, 100, 1500, 180])

    def test_dnv_reference_proposal(self):
        candidates, printed = _dnv_candidates()
        analysis = PageAnalysis(1, 300, tuple(printed["page_size_px"]), 12.0, "text_layer", tuple(candidates))
        definition = build_proposal(analysis, match_candidates(analysis, "dnv"), "dnv", {"provider": "openai", "model": "fake"})
        validated = validate_mapping(definition)
        self.assertTrue(any("Revise todas as sugestões" in issue for issue in mapping_issues(validated)))
        self.assertEqual((definition["provenance"]["provider"], definition["provenance"]["model"]), ("openai", "fake"))
        fields = {field["suggestion"]["candidate_id"]: field for field in definition["fields"]}
        self.assertEqual((fields["2"]["semantic"], fields["2"]["data_type"]), ("birth.date", "digit_sequence"))
        self.assertEqual((fields["46"]["semantic"], len(fields["46"]["elements"])), ("filling.date", 6))
        self.assertEqual(fields["5"]["semantic"], "newborn.sex")
        accepted = {**definition, "fields": [accept_suggestion(field) for field in definition["fields"]]}
        self.assertEqual(mapping_issues(validate_mapping({**accepted, "provenance": {**accepted["provenance"], "reviewed": True}})), [])

    def test_certidao_date_parts(self):
        definition = json.loads(CERTIDAO.read_text(encoding="utf-8"))
        items = tuple(candidate(field["field_id"], field["label"], region=(100, 100 + 100 * index, 900, 180 + 100 * index))
                      for index, field in enumerate(definition["fields"]))
        analysis = PageAnalysis(1, 300, (2481, 3508), 12.0, "text_layer", items)
        proposal = build_proposal(analysis, match_candidates(analysis, "certidao_nascimento"), "certidao_nascimento", {})
        fields = {field["suggestion"]["candidate_id"]: field for field in proposal["fields"]}
        self.assertEqual(fields["dia_nascimento"]["semantic"], {"key": "birth.date", "part": "day"})
        self.assertEqual(fields["dia_nascimento"]["data_type"], "integer")
        self.assertTrue(any("Revise todas as sugestões" in issue for issue in mapping_issues(validate_mapping(proposal))))


PREGNANCY = (("Única", "1"), ("Dupla", "2"), ("Tripla ou mais", "3"), ("Ignorado", "9"))
DELIVERY = (("Vaginal", "1"), ("Cesáreo", "2"), ("Ignorado", "9"))
CESAREAN = (("Sim", "1"), ("Não", "2"), ("Não se aplica", "3"), ("Ignorado", "9"))


class GateRegressionTests(unittest.TestCase):
    def test_dnv_categorical_checkboxes_bind_and_generate(self):
        from src.synthetic_dataset import generate_dataset

        items = [candidate("g", "Tipo de gravidez", "V", "checkbox", PREGNANCY, region=(100, 100, 700, 140)),
                 candidate("p", "Tipo de parto", "V", "checkbox", DELIVERY, region=(100, 200, 700, 240)),
                 candidate("c", "Cesárea ocorreu antes do trabalho de parto iniciar?", "V", "checkbox", CESAREAN,
                           region=(100, 300, 700, 340)),
                 candidate("a", "Detectada alguma anomalia congênita?", "VI", "checkbox", YES_NO, region=(100, 400, 700, 440)),
                 candidate("d", "Descrever todas as anomalias congênitas observadas", "VI", region=(100, 500, 1500, 560)),
                 candidate("x", "Tipo de parto", "V", "checkbox", (("Opção 1", None), ("Opção 2", None)), region=(100, 600, 700, 640))]
        matches = {match.candidate_id: match for match in match_candidates(items, "dnv")}
        expected = {"g": "birth.pregnancy_type", "p": "birth.delivery_type", "c": "birth.cesarean_before_labor",
                    "a": "newborn.anomaly_status", "d": "newborn.anomaly_description"}
        for identifier, key in expected.items():
            self.assertEqual((matches[identifier].status, matches[identifier].key, matches[identifier].method),
                             ("matched", key, "rules"), matches[identifier].reason)
        self.assertEqual(matches["x"].status, "ambiguous")
        self.assertIn("não representam", matches["x"].reason)
        analysis = PageAnalysis(1, 300, (4330, 5717), 12.0, "text_layer", tuple(items[:5]))
        proposal = build_proposal(analysis, list(matches.values()), "dnv", {})
        self.assertEqual({field["semantic"] for field in proposal["fields"]}, set(expected.values()))
        definition = {**proposal, "fields": [accept_suggestion(field) for field in proposal["fields"]],
                      "provenance": {**proposal["provenance"], "reviewed": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(definition), encoding="utf-8")
            documents = generate_dataset(ROOT / "documents" / "dnv" / "template.pdf", path, {field["field_id"] for field in definition["fields"]},
                                         Path(directory) / "saida", 2, seed=5, handwriting=False, text_handwriting=False)
            for document in documents:
                truth = json.loads(document.json_path.read_text(encoding="utf-8"))
                by_id = {field["field_id"]: field for field in truth["fields"]}
                record = truth["semantic_record"]
                for field_id, key in (("birth_delivery_type", "birth.delivery_type"), ("birth_pregnancy_type", "birth.pregnancy_type"),
                                      ("newborn_anomaly_status", "newborn.anomaly_status")):
                    checked = [element["label"] for element in by_id[field_id]["elements"] if element.get("checked")]
                    self.assertEqual(checked, [record[key]])

    def test_uf_and_death_certificate_bindings(self):
        items = [candidate("r", "UF", "II", "digit_sequence", 2), candidate("o", "UF", "III", "digit_sequence", 2),
                 candidate("v", "UF", "VIII", "digit_sequence", 2), candidate("m", "UF", "VI", "digit_sequence", 2),
                 candidate("d", "Data", "VIII", "digit_sequence", 8), candidate("i", "Idade", "I"),
                 candidate("b", "Data de nascimento", "I", "digit_sequence", 8), candidate("b11", "Data de nascimento", "I", "digit_sequence", 11),
                 candidate("iso", "Idade Anos completos", "I", "checkbox", (("Ignorado", "9"),))]
        matches = {match.candidate_id: match for match in match_candidates(items[:-1], DO) + match_candidates(items[-1:], DO)}
        for identifier, key in (("r", "deceased.residence.uf"), ("o", "death.place.uf"), ("v", "death_registration.office.uf"),
                                ("d", "death_registration.date"), ("i", "deceased.age"), ("b", "deceased.birth_date"),
                                ("iso", "deceased.age")):
            self.assertEqual((matches[identifier].status, matches[identifier].key), ("matched", key), identifier)
        self.assertEqual(matches["b11"].status, "ambiguous")
        self.assertEqual(matches["b11"].alternatives[0][0], "deceased.birth_date")
        uf = matches["m"].proposal
        self.assertEqual((uf["data_type"], uf["kind"]), ("uf", "uf"))
        self.assertNotIn("inteiro", uf["generator"])
        analysis = PageAnalysis(1, 300, (2550, 3300), 12.0, "text_layer",
                                tuple(replace(item, region=(100, 100 + 100 * index, 900, 160 + 100 * index)) for index, item in enumerate(items)))
        fields = {field["suggestion"]["candidate_id"]: field for field in build_proposal(analysis, list(matches.values()), DO, {})["fields"]}
        for identifier in ("r", "o", "v", "m"):
            self.assertEqual(fields[identifier]["data_type"], "uf")
            self.assertNotIn("elements", fields[identifier])
        self.assertEqual(fields["r"]["semantic"], "deceased.residence.uf")
        self.assertNotIn("semantic", fields["iso"])
        self.assertEqual(fields["iso"]["suggestion"]["match"], "ambiguous")
        self.assertEqual(fields["iso"]["suggestion"]["alternatives"][0]["key"], "deceased.age")

    def test_single_valued_keys_are_not_duplicated(self):
        items = [candidate("cbo", "Código CBO 2002", "III", "digit_sequence", 6), candidate("cod", "Código", "III", "digit_sequence", 3),
                 candidate("s1", "Sexo", "I", "checkbox", SEX), candidate("s2", "Sexo", "I", "checkbox", SEX),
                 candidate("n1", "Nome", "III"), candidate("n2", "Nome da mãe", "III")]
        matches = {match.candidate_id: match for match in match_candidates(items, "dnv")}
        self.assertEqual((matches["cbo"].status, matches["cbo"].key), ("matched", "mother.occupation_cbo"))
        self.assertEqual((matches["cod"].status, matches["cod"].key), ("ambiguous", None))
        self.assertEqual(matches["cod"].alternatives[0][0], "mother.occupation_cbo")
        self.assertIn("já está no campo", matches["cod"].reason)
        self.assertEqual([matches[name].key for name in ("s1", "s2")], ["newborn.sex", "newborn.sex"])
        self.assertEqual(sum(match.key == "mother.name" for match in matches.values()), 1)

    def test_invalid_stored_custom_definitions_are_skipped(self):
        valid = {"key": "custom.death.circumstance", "label": "Circunstância do óbito", "description": "Campo aprovado",
                 "data_type": "text", "generator": {"type": "unavailable"}}
        broken = [{"key": "custom.quebrada", "label": "", "description": "x", "data_type": "text", "generator": {"type": "x"}},
                  {"label": "sem chave"}, "lixo"]
        analysis = PageAnalysis(1, 300, (2480, 3508), 12.0, "text_layer", (candidate("c", "Circunstância do óbito", region=(100, 100, 900, 160)),))
        matches = match_candidates(analysis, DO, custom_definitions=[*broken, valid])
        proposal = build_proposal(analysis, matches, DO, {}, custom_definitions=[*broken, valid])
        self.assertEqual(proposal["fields"][0]["semantic"], valid["key"])
        self.assertEqual(set(proposal["semantic_definitions"]), {valid["key"]})

    def test_shared_medical_terms_and_custom_key_format(self):
        for key, label in (("custom.apgar_1", "Apgar 1º minuto"), ("custom.peso_nascer", "Peso ao nascer"),
                           ("custom.newborn.perimetro", "Perímetro cefálico"), ("custom.death.cause", "Causa básica")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_custom_definition(key, label, "Valor numérico", "integer", {"type": "integer_range", "min": 0, "max": 10})
            validate_custom_definition(key, label, "Valor numérico", "integer", {"type": "unavailable"})
        validate_custom_definition("custom.cidade_natal", "Cidade natal", "Município", "text", {"type": "choices", "values": ["Recife"]})
        for label, section, kind, elements in (("Apgar 1º minuto", "I", "digit_sequence", 2), ("Peso ao nascer", "I", "digit_sequence", 4),
                                               ("Raça/Cor", "I", "checkbox", (("Branca", "1"), ("Preta", "2"))),
                                               ("Quantidade de parafusos", "II", "text", 0), ("1º campo", None, "text", 0)):
            match = match_candidates([candidate("n", label, section, kind, elements)], "dnv")[0]
            proposal = match.proposal
            with self.subTest(label=label):
                generator = {"type": "unavailable"}
                definition = validate_custom_definition(proposal["key"], proposal["label"], proposal["description"], "text", generator)
                self.assertEqual(definition["key"], proposal["key"])
                self.assertIn(proposal["data_type"], FIELD_TYPES)
                self.assertIn(proposal["kind"], KINDS)
                if label.startswith(("Apgar", "Peso")):
                    self.assertTrue(proposal["generator"].startswith(MEDICAL_GENERATOR))


if __name__ == "__main__":
    unittest.main()
