from src.semantic import REGISTRY
import json
import unittest
from datetime import date, time
from pathlib import Path

from src.field_rules import DATE_FORMATS
from src.semantic.present import checkbox_choice, kind_compatible, present_elements, present_text
from src.semantic.registry import REGISTRY
from src.semantic.resolve import EXTENSO, KEYS, Binding, normalize, resolve_field
from src.synthetic_dataset import apply_shared_rules

ROOT = Path(__file__).resolve().parent.parent
DNV_MAPPINGS = ROOT / "documents" / "dnv" / "mappings"
CERTIDAO = ROOT / "documents" / "certidao_nascimento" / "certidao_nascimento_fields.json"
LEAP_DAY = date(2024, 2, 29)
DNV_EXPECTED = {
    "6": "newborn.anomaly_status", "17_codigo_cbo": "mother.occupation_cbo",
    "35": "birth.pregnancy_type", "38": "birth.delivery_type", "39": "birth.cesarean_before_labor",
    "41": "newborn.anomaly_description",
    "1": "newborn.name", "2": "birth.date", "2_hora": "birth.time", "3": "newborn.sex", "8": "birth.establishment.name",
    "8_codigo_cnes": "birth.establishment.cnes", "9": "birth.place.street", "9_numero": "birth.place.number",
    "10": "birth.place.cep", "11": "birth.place.neighborhood", "12": "birth.place.municipality", "12_codigo": "birth.place.ibge",
    "13": "birth.place.uf", "14": "mother.name", "15": "mother.cns", "15_cpf": "mother.cpf", "17": "mother.occupation",
    "18": "mother.birth_date", "19": "mother.age", "20": "mother.naturalidade", "23": "mother.residence.street",
    "23_numero": "mother.residence.number", "23_complemento": "mother.residence.complement", "24": "mother.residence.cep",
    "25": "mother.residence.neighborhood", "26": "mother.residence.municipality", "27": "mother.residence.uf",
    "28": "guardian.name", "29": "guardian.age", "30_gestacoes_anteriores": "mother.obstetric.previous_pregnancies",
    "30_partos_vaginais": "mother.obstetric.vaginal_births", "30_cesareas": "mother.obstetric.cesareans",
    "30_nascidos_vivos": "mother.obstetric.live_births", "30_perdas_fetais_abortos": "mother.obstetric.fetal_losses",
    "31": "pregnancy.last_menstrual_period", "32": "pregnancy.weeks", "33": "pregnancy.prenatal_visits",
    "34": "pregnancy.prenatal_start_month", "42": "filling.date", "43": "filling.responsible.name",
    "46": "filling.responsible.document_number", "47": "filling.responsible.issuer", "48": "registration.office.name",
    "49": "registration.number", "50": "registration.date", "51": "registration.office.municipality",
    "52": "registration.office.uf",
}
CERTIDAO_EXPECTED = {
    "cpf": Binding("newborn.cpf"), "matricula": Binding("registration.matricula"),
    "data_nascimento_por_extenso": Binding("birth.date", format=EXTENSO), "dia_nascimento": Binding("birth.date", "day"),
    "mes_nascimento": Binding("birth.date", "month"), "ano_nascimento": Binding("birth.date", "year"),
    "hora_nascimento": Binding("birth.time"), "naturalidade": Binding("newborn.naturalidade"),
    "municipio_registro_uf": Binding("registration.place.summary"),
    "local_municipio_nascimento_uf": Binding("birth.place.summary"), "sexo": Binding("newborn.sex"),
    "filiacao": Binding("parents.names"), "avos": Binding("grandparents.names"),
    "data_registro_por_extenso": Binding("registration.date", format=EXTENSO),
    "numero_dnv": Binding("registration.dnv_number"), "gemeos": Binding("birth.twins"),
    "nome_matricula_gemeos": Binding("twin.name_matricula"),
    "averbacoes_anotacoes_acrescer": None, "anotacoes_cadastro": None,
}
CLINICAL_TERMS = ("apgar", "peso", "comprimento", "perimetro", "raca", "escolaridade", "situacao conjugal", "funcao",
                  "tipo de documento", "apresentacao", "induzido", "metodo")
SAMPLE = {
    "birth.pregnancy_type": "Única", "birth.delivery_type": "Vaginal", "birth.cesarean_before_labor": "Não se aplica",
    "newborn.anomaly_status": "Ignorado", "newborn.anomaly_description": None, "mother.occupation_cbo": "411010",
    "newborn.name": "Ana Souza Lima", "newborn.sex": "F", "newborn.cpf": "52998224725",
    "newborn.naturalidade": "Recife/PE", "birth.date": LEAP_DAY, "birth.time": time(7, 5),
    "birth.establishment.name": "Hospital Municipal", "birth.establishment.cnes": "2077485",
    "birth.place.street": "Rua do Imperador", "birth.place.number": "45", "birth.place.cep": "50030000",
    "birth.place.neighborhood": "Santo Antônio", "birth.place.municipality": "Recife", "birth.place.ibge": "2611606",
    "birth.place.uf": "PE", "birth.place.summary": "Hospital Municipal, Recife, PE", "mother.name": "Maria Souza Lima",
    "mother.cns": "700000000000005", "mother.cpf": "11144477735", "mother.occupation": "Professora",
    "mother.birth_date": date(1994, 7, 3), "mother.age": 29, "mother.naturalidade": "Olinda/PE",
    "mother.residence.street": "Rua da Aurora", "mother.residence.number": "7", "mother.residence.complement": "Apto 101",
    "mother.residence.cep": "50050000", "mother.residence.neighborhood": "Boa Vista",
    "mother.residence.municipality": "Recife", "mother.residence.uf": "PE", "guardian.name": "José Lima", "guardian.age": 41,
    "mother.obstetric.previous_pregnancies": 2, "mother.obstetric.vaginal_births": 1, "mother.obstetric.cesareans": 1,
    "mother.obstetric.live_births": 2, "mother.obstetric.fetal_losses": 0, "pregnancy.last_menstrual_period": date(2023, 5, 24),
    "pregnancy.weeks": 39, "pregnancy.prenatal_visits": 8, "pregnancy.prenatal_start_month": 2,
    "filling.date": date(2024, 3, 1), "filling.responsible.name": "Carla Ribeiro",
    "filling.responsible.document_number": "123456", "filling.responsible.issuer": "COREN",
    "registration.office.name": "Cartório do 1º Ofício", "registration.number": "4321",
    "registration.date": date(2024, 3, 5), "registration.office.municipality": "Recife", "registration.office.uf": "PE",
    "registration.matricula": "12345601552024100001001000000110", "registration.dnv_number": "12345678901",
    "registration.place.summary": "Recife/PE", "parents.names": "Maria Souza Lima e José Lima",
    "grandparents.names": "Paterno: João Lima; materno: Rita Souza", "birth.twins": True,
    "twin.name_matricula": "Clara Souza Lima, 123456.01.55.2024.1.00001.001.0000002-11",
}


def _fields(path: Path) -> dict[str, dict]:
    return {str(field["field_id"]): field for field in apply_shared_rules(json.loads(path.read_text(encoding="utf-8")))["fields"]}


class ResolutionTests(unittest.TestCase):
    def test_active_dnv_mapping_bindings(self):
        fields = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        self.assertTrue(set(DNV_EXPECTED) <= set(fields))
        for field_id, field in fields.items():
            with self.subTest(field_id=field_id):
                expected = DNV_EXPECTED.get(field_id)
                self.assertEqual(resolve_field(field, "dnv"), Binding(expected) if expected else None)

    def test_certidao_bindings(self):
        fields = _fields(CERTIDAO)
        self.assertEqual(set(CERTIDAO_EXPECTED), set(fields))
        for field_id, field in fields.items():
            with self.subTest(field_id=field_id):
                self.assertEqual(resolve_field(field, "certidao_nascimento"), CERTIDAO_EXPECTED[field_id])

    def test_other_dnv_mappings_resolve_safely(self):
        for path in sorted(DNV_MAPPINGS.glob("*.json")):
            for field_id, field in _fields(path).items():
                for document in ("dnv", None):
                    with self.subTest(mapping=path.name, field_id=field_id, document=document):
                        binding = resolve_field(field, document)
                        if binding is None:
                            continue
                        text = f' {normalize(" ".join(str(field.get(key) or "") for key in ("field_id", "label", "description")))} '
                        self.assertIn(binding.key, KEYS)
                        self.assertFalse(any(f" {term} " in text for term in CLINICAL_TERMS), binding)
                        if field.get("data_type") == "checkbox":
                            self.assertTrue(binding.key.endswith(".sex") or binding.key in {"birth.pregnancy_type", "birth.delivery_type", "birth.cesarean_before_labor", "newborn.anomaly_status"})

    def test_other_dnv_mappings_known_bindings(self):
        fields = _fields(DNV_MAPPINGS / "20260831.json")
        expected = {"1": "newborn.name", "5": "newborn.sex", "57": "guardian.name", "idade": "guardian.age",
                    "data_ultima_menstruacao": "pregnancy.last_menstrual_period", "52": "registration.office.name"}
        for field_id, key in expected.items():
            self.assertEqual(resolve_field(fields[field_id], "dnv"), Binding(key))
        for field_id in ("2", "18", "20", "cpf", "CEP", "uf", "67", "10", "50"):
            self.assertIsNone(resolve_field(fields[field_id], "dnv"), field_id)

    def test_document_context(self):
        fields = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        self.assertEqual(resolve_field(fields["14"]), Binding("mother.name"))
        self.assertIsNone(resolve_field(fields["15"]))
        self.assertEqual(resolve_field(fields["15"], "dnv_v1"), Binding("mother.cns"))
        death = {"nome": ("Nome do falecido", "deceased.name"), "mae": ("Nome da mãe", "deceased.mother_name"),
                 "pai": ("Nome do pai", "deceased.father_name"), "data": ("Data do óbito", "death.date"),
                 "hora": ("Hora do óbito", "death.time"), "nasc": ("Data de nascimento", "deceased.birth_date"),
                 "crm": ("CRM", "certifier.registration"), "medico": ("Nome do médico", "certifier.name"),
                 "mun": ("Município de ocorrência", "death.place.municipality")}
        for field_id, (label, key) in death.items():
            self.assertEqual(resolve_field({"field_id": field_id, "label": label}, "do"), Binding(key), label)

    def test_date_fields_keep_the_full_date_and_contexts_restrict_inference(self):
        self.assertEqual(resolve_field({"field_id": "d", "label": "Dia do nascimento", "data_type": "date"}, "certidao_nascimento"), Binding("birth.date"))
        self.assertEqual(resolve_field({"field_id": "d", "label": "Dia do nascimento", "data_type": "integer"}, "certidao_nascimento"),
                         Binding("birth.date", "day"))
        self.assertIsNone(resolve_field({"field_id": "c", "label": "CPF da mãe"}, "certidao_nascimento"))
        self.assertIsNone(resolve_field({"field_id": "n", "label": "Semanas de gestação"}, "do"))
        self.assertEqual(resolve_field({"field_id": "c", "label": "CPF da mãe"}, "dnv"), Binding("mother.cpf"))
        death = {"data": ("Data do registro", None, "death_registration.date"), "cart": ("Cartório", None, "death_registration.office.name"),
                 "mun": ("Município", "VIII", "death_registration.office.municipality"), "uf": ("UF", "VIII", "death_registration.office.uf"),
                 "num": ("Registro", "VIII", "death_registration.number")}
        for field_id, (label, block, key) in death.items():
            self.assertEqual(resolve_field({"field_id": field_id, "label": label, "bloco": block}, "declaracao_obito"), Binding(key), label)
        self.assertEqual(KEYS, frozenset(REGISTRY))

    def test_twins_are_presented_from_booleans(self):
        certidao = _fields(CERTIDAO)
        self.assertEqual(present_text(certidao["gemeos"], Binding("birth.twins"), True), "SIM")
        self.assertEqual(present_text(certidao["gemeos"], Binding("birth.twins"), False), "NÃO")
        self.assertEqual(present_text({"data_type": "text"}, Binding("birth.twins"), False), "Não")
        boxes = {"data_type": "checkbox", "elements": [{"label": "Sim"}, {"label": "Não"}]}
        self.assertEqual(checkbox_choice(boxes, Binding("birth.twins"), False), [{"label": "Não"}])

    def test_explicit_binding_precedence_and_validation(self):
        field = {"field_id": "1", "description": "Nome do Recem-nascido (RN)", "data_type": "person_name"}
        self.assertEqual(resolve_field({**field, "semantic": "mother.name"}), Binding("mother.name", source="explicit"))
        self.assertEqual(resolve_field({**field, "semantic": {"key": "birth.date", "part": "year", "format": EXTENSO}}),
                         Binding("birth.date", "year", EXTENSO, "explicit"))
        self.assertEqual(resolve_field(field), Binding("newborn.name"))
        for semantic in ("newborn.unknown", {"key": "nada"}, {"part": "day"}, {"key": "birth.date", "part": "week"}, 7):
            with self.subTest(semantic=semantic), self.assertRaises(ValueError):
                resolve_field({**field, "semantic": semantic})


class PresentationTests(unittest.TestCase):
    def test_dates_in_all_formats(self):
        expected = {"DD/MM/AAAA": "29/02/2024", "DD-MM-AAAA": "29-02-2024", "DD/MM/AA": "29/02/24", "AAAA-MM-DD": "2024-02-29",
                    "DDMMAAAA": "29022024", "DDMMYYYY": "29022024", EXTENSO: "29 de fevereiro de 2024",
                    "EXTENSO_PALAVRAS": "vinte e nove de fevereiro de dois mil e vinte e quatro", "SEPARADO": "29/02/2024"}
        self.assertEqual(set(expected), set(DATE_FORMATS))
        for fmt, text in expected.items():
            field = {"data_type": "date", "validation": {"format": fmt}}
            self.assertEqual(present_text(field, Binding("birth.date"), LEAP_DAY), text)
            self.assertTrue(kind_compatible(field, Binding("birth.date"), LEAP_DAY))
        self.assertEqual(present_text({"data_type": "text"}, Binding("birth.date", format=EXTENSO), LEAP_DAY),
                         "29 de fevereiro de 2024")
        self.assertEqual(present_text({"data_type": "date", "validation": {"format": "AAAA-MM-DD"}},
                                      Binding("birth.date", format="DD/MM/AAAA"), LEAP_DAY), "29/02/2024")
        elements = [{"part": "day"}, {"part": "month"}, {"part": "year"}]
        field = {"data_type": "date", "validation": {"format": "SEPARADO"}, "elements": elements}
        self.assertEqual(present_elements(field, Binding("birth.date"), LEAP_DAY, elements),
                         list(zip(elements, ("29", "02", "2024"))))

    def test_digit_sequence_dates(self):
        fields = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        for field_id, value, text in (("2", LEAP_DAY, "29022024"), ("31", LEAP_DAY, "290224"), ("42", date(2024, 3, 1), "010324")):
            field = fields[field_id]
            ordered = sorted(field["elements"], key=lambda item: item["index"])
            self.assertEqual(present_text(field, Binding("birth.date"), value), text)
            self.assertEqual("".join(digit for _, digit in present_elements(field, Binding("birth.date"), value, ordered)), text)
            self.assertTrue(kind_compatible(field, Binding("birth.date"), value))
        self.assertEqual(present_text({"data_type": "digit_sequence", "validation": {"digit_count": 8}}, Binding("x.date"),
                                      LEAP_DAY), "29022024")

    def test_date_parts(self):
        fields = _fields(CERTIDAO)
        value = date(2024, 3, 5)
        self.assertEqual(present_text(fields["dia_nascimento"], Binding("birth.date", "day"), value), "05")
        self.assertEqual(present_text(fields["mes_nascimento"], Binding("birth.date", "month"), value), "03")
        self.assertEqual(present_text(fields["ano_nascimento"], Binding("birth.date", "year"), value), "2024")
        self.assertEqual(present_text(fields["dia_nascimento"], Binding("birth.date", "day"), LEAP_DAY), "29")
        strict = {"data_type": "integer", "validation": {"pattern": "^[1-9][0-9]?$"}}
        self.assertEqual(present_text(strict, Binding("birth.date", "day"), value), "5")
        boxes = {"data_type": "digit_sequence", "elements": [{"index": 0}, {"index": 1}]}
        self.assertEqual(present_text(boxes, Binding("birth.date", "year"), value), "24")
        for field_id, part in (("dia_nascimento", "day"), ("mes_nascimento", "month"), ("ano_nascimento", "year")):
            self.assertTrue(kind_compatible(fields[field_id], Binding("birth.date", part), value))
        self.assertEqual(present_text(fields["data_nascimento_por_extenso"], CERTIDAO_EXPECTED["data_nascimento_por_extenso"],
                                      LEAP_DAY), "vinte e nove de fevereiro de dois mil e vinte e quatro")

    def test_times(self):
        fields = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        hour = fields["2_hora"]
        ordered = sorted(hour["elements"], key=lambda item: item["index"])
        self.assertEqual(present_elements(hour, Binding("birth.time"), time(7, 5), ordered),
                         list(zip(ordered, "0705")))
        self.assertEqual(present_text(_fields(CERTIDAO)["hora_nascimento"], Binding("birth.time"), time(7, 5)), "07:05")
        self.assertEqual(present_text({"data_type": "time"}, Binding("birth.time"), time(23, 59)), "23:59")

    def test_integers_are_zero_padded_in_boxes(self):
        fields = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        self.assertEqual(present_text(fields["19"], Binding("mother.age"), 7), "07")
        self.assertEqual(present_text(fields["30_perdas_fetais_abortos"], Binding("mother.obstetric.fetal_losses"), 0), "00")
        self.assertEqual(present_text(fields["34"], Binding("pregnancy.prenatal_start_month"), 2), "2")
        self.assertEqual(present_text(fields["49"], Binding("registration.number"), "4321"), "004321")
        self.assertEqual(present_text({"data_type": "integer"}, Binding("mother.age"), 7), "7")
        self.assertEqual(present_text({"data_type": "text"}, Binding("mother.age"), 31), "31")

    def test_sex_presentations(self):
        self.assertEqual(present_text({"data_type": "sex", "validation": {"format": "completo"}}, Binding("newborn.sex"), "M"),
                         "Masculino")
        self.assertEqual(present_text({"data_type": "sex", "validation": {"format": "abreviado"}}, Binding("newborn.sex"), "F"), "F")
        sexo = _fields(CERTIDAO)["sexo"]
        self.assertEqual(present_text(sexo, Binding("newborn.sex"), "M"), "MASCULINO")
        self.assertEqual(present_text(sexo, Binding("newborn.sex"), "F"), "FEMININO")
        self.assertTrue(kind_compatible(sexo, Binding("newborn.sex"), "F"))
        self.assertEqual(present_text({"data_type": "enum", "validation": {"allowed_values": ["M", "F"]}}, Binding("newborn.sex"), "F"), "F")
        self.assertEqual(present_text({"data_type": "enum", "allowed_values": ["Masc.", "Fem."]}, Binding("newborn.sex"), "M"), "Masc.")
        self.assertEqual(present_text({"data_type": "text", "validation": {"pattern": "^[MF]$"}}, Binding("newborn.sex"), "M"), "M")
        checkbox = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")["3"]
        for code, label in (("M", "Masculino"), ("F", "Feminino")):
            chosen = checkbox_choice(checkbox, Binding("newborn.sex"), code)
            self.assertEqual([element["label"] for element in chosen], [label])
            self.assertTrue(kind_compatible(checkbox, Binding("newborn.sex"), code))
        coded = {"data_type": "checkbox", "elements": [{"label": "1", "value": "M"}, {"label": "2", "value": "F"}]}
        self.assertEqual(checkbox_choice(coded, Binding("newborn.sex"), "F"), [{"label": "2", "value": "F"}])

    def test_identifier_formats(self):
        certidao = _fields(CERTIDAO)
        dnv = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        self.assertEqual(present_text(certidao["cpf"], Binding("newborn.cpf"), "52998224725"), "529.982.247-25")
        self.assertEqual(present_text(dnv["15_cpf"], Binding("mother.cpf"), "52998224725"), "52998224725")
        self.assertEqual(present_text({"data_type": "document"}, Binding("mother.cpf"), "52998224725"), "529.982.247-25")
        self.assertEqual(present_text({"data_type": "text", "validation": {"pattern": "^[0-9]{11}$"}}, Binding("mother.cpf"),
                                      "52998224725"), "52998224725")
        self.assertEqual(present_text(certidao["matricula"], Binding("registration.matricula"), SAMPLE["registration.matricula"]),
                         "123456.01.55.2024.1.00001.001.0000001-10")
        self.assertEqual(present_text({"data_type": "text"}, Binding("birth.place.cep"), "01001000"), "01001-000")
        self.assertEqual(present_text({"data_type": "cep"}, Binding("birth.place.cep"), "01001000"), "01001-000")
        self.assertEqual(present_text(dnv["10"], Binding("birth.place.cep"), "01001000"), "01001000")
        self.assertEqual(present_text({"data_type": "text", "validation": {"pattern": "^[0-9]{8}$"}}, Binding("birth.place.cep"),
                                      "01001000"), "01001000")
        self.assertEqual(present_text({"data_type": "dnv_number", "validation": {"format": "agrupado"}},
                                      Binding("registration.dnv_number"), "12345678901"), "12-34567890-1")
        self.assertEqual(present_text({"data_type": "dnv_number", "validation": {"format": "digitos"}},
                                      Binding("registration.dnv_number"), "12345678901"), "12345678901")
        boxes = [{"index": index} for index in range(11)]
        self.assertEqual(present_elements({"data_type": "dnv_number", "validation": {"format": "caixas"}, "elements": boxes},
                                          Binding("registration.dnv_number"), "12345678901", boxes), list(zip(boxes, "12345678901")))
        self.assertEqual(present_text(certidao["numero_dnv"], Binding("registration.dnv_number"), "12345678901"), "12345678901")
        self.assertEqual(present_text(dnv["15"], Binding("mother.cns"), "700000000000005"), "700000000000005")
        self.assertEqual(present_text(dnv["13"], Binding("birth.place.uf"), "PE"), "PE")
        for field_id in ("cpf", "matricula", "numero_dnv"):
            binding = CERTIDAO_EXPECTED[field_id]
            self.assertTrue(kind_compatible(certidao[field_id], binding, SAMPLE[binding.key]), field_id)

    def test_every_active_binding_accepts_a_contract_value(self):
        for path, expected, document in ((DNV_MAPPINGS / "mapeamento_seguindo_norma.json", DNV_EXPECTED, "dnv"),
                                         (CERTIDAO, CERTIDAO_EXPECTED, "certidao_nascimento")):
            for field_id, field in _fields(path).items():
                binding = resolve_field(field, document)
                if binding is None:
                    continue
                with self.subTest(field_id=field_id, key=binding.key):
                    value = SAMPLE[binding.key]
                    if value is None:
                        self.assertTrue(REGISTRY[binding.key].optional)
                        continue
                    self.assertTrue(kind_compatible(field, binding, value))
                    ordered = sorted(field.get("elements") or [], key=lambda item: item.get("index", 10_000))
                    if field.get("data_type") == "digit_sequence":
                        pairs = present_elements(field, binding, value, ordered)
                        self.assertEqual(len(pairs), len(ordered))
                        self.assertTrue(all(digit.isdigit() for _, digit in pairs))
                    elif field.get("data_type") != "checkbox":
                        self.assertTrue(present_text(field, binding, value))
        self.assertEqual(present_text(_fields(CERTIDAO)["sexo"], Binding("newborn.sex"), "F"), "FEMININO")

    def test_incompatible_values_fall_back(self):
        dnv = _fields(DNV_MAPPINGS / "mapeamento_seguindo_norma.json")
        certidao = _fields(CERTIDAO)
        cases = (
            (dnv["15_cpf"], Binding("mother.name"), "Maria Souza"),
            (dnv["3"], Binding("mother.name"), "Maria Souza"),
            (dnv["3"], Binding("birth.date"), LEAP_DAY),
            (dnv["a"], Binding("newborn.sex"), "F"),
            (dnv["19"], Binding("mother.age"), 123),
            (dnv["19"], Binding("mother.age"), -1),
            (dnv["9_numero"], Binding("birth.place.number"), "1234"),
            (dnv["9_numero"], Binding("birth.place.number"), "S/N"),
            (dnv["10"], Binding("birth.place.cep"), "0100100099"),
            (dnv["34"], Binding("pregnancy.weeks"), 39),
            ({"data_type": "digit_sequence", "elements": [{"index": i} for i in range(4)]}, Binding("birth.date"), LEAP_DAY),
            (certidao["gemeos"], Binding("newborn.sex"), "M"),
            (certidao["gemeos"], Binding("birth.twins"), None),
            (certidao["dia_nascimento"], Binding("birth.date"), LEAP_DAY),
            (certidao["sexo"], Binding("newborn.name"), "Ana"),
            (certidao["filiacao"], Binding("birth.date"), LEAP_DAY),
            (certidao["hora_nascimento"], Binding("birth.date"), LEAP_DAY),
            ({"data_type": "date"}, Binding("birth.date"), "29/02/2024"),
            ({"data_type": "uf"}, Binding("newborn.name"), "Ana"),
            ({"data_type": "text"}, Binding("newborn.name", "day"), "Ana"),
            ({"data_type": "enum", "validation": {"allowed_values": ["A", "B"]}}, Binding("birth.place.uf"), "PE"),
            ({"data_type": "text"}, Binding("newborn.name"), None),
        )
        for field, binding, value in cases:
            with self.subTest(field=field.get("field_id") or field.get("data_type"), key=binding.key, value=value):
                self.assertFalse(kind_compatible(field, binding, value))
        self.assertIsNone(checkbox_choice(dnv["a"], Binding("newborn.sex"), "M"))
        self.assertIsNone(checkbox_choice(dnv["3"], Binding("newborn.name"), "Ana"))


if __name__ == "__main__":
    unittest.main()
