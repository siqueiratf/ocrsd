from __future__ import annotations

from dataclasses import dataclass

VERSION = "2"
KINDS = frozenset({
    "name", "sex", "date", "time", "integer", "uf", "municipality", "cep", "ibge", "cnes", "cpf", "cns", "dnv_number",
    "matricula", "document_number", "issuer", "street", "house_number", "complement", "neighborhood", "occupation", "text",
    "boolean",
})
CONTEXTS = ("dnv", "certidao_nascimento", "declaracao_obito")
DNV, CERT, DO = ("dnv",), ("certidao_nascimento",), ("declaracao_obito",)
BIRTH_DOCS = DNV + CERT


@dataclass(frozen=True)
class SemanticField:
    key: str
    label: str
    description: str
    kind: str
    entity: str
    group: str
    depends_on: tuple[str, ...] = ()
    contexts: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    optional: bool = False


def _field(key: str, label: str, kind: str, group: str, description: str, contexts: tuple[str, ...],
           synonyms: tuple[str, ...], depends_on: tuple[str, ...] = (), optional: bool = False) -> SemanticField:
    return SemanticField(key, label, description, kind, key.split(".", 1)[0], group, depends_on, contexts, synonyms, optional)


def _address(prefix: str, group: str, owner: str, contexts: tuple[str, ...], terms: tuple[str, ...],
             parts: tuple[str, ...]) -> list[SemanticField]:
    specs = {
        "street": ("Logradouro", "street", ("logradouro", "endereco", "rua", "avenida")),
        "number": ("Número do endereço", "house_number", ("numero do endereco", "numero da residencia", "numero")),
        "complement": ("Complemento", "complement", ("complemento",)),
        "neighborhood": ("Bairro/Distrito", "neighborhood", ("bairro", "distrito", "bairro/distrito")),
        "cep": ("CEP", "cep", ("cep", "codigo postal")),
        "municipality": ("Município", "municipality", ("municipio", "cidade")),
        "uf": ("UF", "uf", ("uf", "estado", "unidade da federacao")),
        "ibge": ("Código IBGE do município", "ibge", ("codigo ibge", "codigo (ibge)", "codigo do municipio")),
    }
    fields = []
    for part in parts:
        label, kind, base = specs[part]
        synonyms = tuple(f"{word} {term}" for term in terms for word in base) + base
        depends = () if part in {"number", "complement"} else tuple(f"{prefix}.{other}" for other in parts
                                                                    if other not in {part, "number", "complement"})
        fields.append(_field(f"{prefix}.{part}", f"{label} ({owner})", kind, group,
                             f"{label} de {owner}, sempre do mesmo município que os demais campos do endereço",
                             contexts, synonyms, depends))
    return fields


_ADDRESS = ("street", "number", "complement", "neighborhood", "cep", "municipality", "uf", "ibge")
FIELDS = [
    _field("newborn.name", "Nome do recém-nascido", "name", "Recém-nascido", "Nome completo do recém-nascido ou registrado",
           BIRTH_DOCS, ("nome do recem-nascido", "nome do recem nascido", "nome_rn", "nome do rn", "nome do registrado",
                        "nome do nascido", "nome da crianca"), ("mother.name", "father.name")),
    _field("newborn.sex", "Sexo do recém-nascido", "sex", "Recém-nascido", "Sexo do recém-nascido (M ou F)", BIRTH_DOCS,
           ("sexo", "sexo do recem-nascido", "sexo do rn", "sexo do registrado")),
    _field("newborn.cpf", "CPF do recém-nascido", "cpf", "Recém-nascido", "CPF sintético do registrado, com dígitos verificadores válidos",
           CERT, ("cpf", "cpf do registrado", "cpf do recem-nascido")),
    _field("newborn.naturalidade", "Naturalidade do recém-nascido", "municipality", "Recém-nascido",
           "Município do nascimento ou da residência da mãe, conforme a lei", CERT,
           ("naturalidade", "naturalidade do registrado"), ("birth.place.municipality", "mother.residence.municipality")),
    _field("mother.name", "Nome da mãe", "name", "Mãe", "Nome completo da mãe (parturiente)", BIRTH_DOCS,
           ("nome da mae", "nome da parturiente", "parturiente", "mae", "nome")),
    _field("mother.birth_date", "Data de nascimento da mãe", "date", "Mãe", "Data de nascimento da mãe, coerente com a idade no parto",
           DNV, ("data de nascimento da mae", "nascimento da mae", "data de nascimento da parturiente"), ("birth.date",)),
    _field("mother.age", "Idade da mãe", "integer", "Mãe", "Idade da mãe em anos completos na data do nascimento", DNV,
           ("idade da mae", "idade da parturiente", "idade"), ("mother.birth_date", "birth.date")),
    _field("mother.cpf", "CPF da mãe", "cpf", "Mãe", "CPF sintético da mãe, com dígitos verificadores válidos", DNV,
           ("cpf da mae", "cpf da parturiente", "cpf")),
    _field("mother.cns", "Cartão SUS da mãe", "cns", "Mãe", "Número do Cartão Nacional de Saúde da mãe (15 algarismos)", DNV,
           ("cartao sus", "cartao nacional de saude", "cns", "cartao sus da mae")),
    _field("mother.occupation", "Ocupação da mãe", "occupation", "Mãe", "Ocupação habitual da mãe", DNV,
           ("ocupacao habitual", "ocupacao da mae", "profissao da mae", "ocupacao")),
    _field("mother.naturalidade", "Naturalidade da mãe", "municipality", "Mãe", "Município onde a mãe nasceu", DNV,
           ("naturalidade da mae", "naturalidade", "local de nascimento da mae")),
    *_address("mother.residence", "Residência da mãe", "residência da mãe", DNV, ("da residencia", "de residencia da mae", "da mae"),
              _ADDRESS),
    _field("mother.obstetric.previous_pregnancies", "Gestações anteriores", "integer", "Histórico obstétrico",
           "Número de gestações anteriores da mãe", DNV, ("gestacoes anteriores", "n gestacoes anteriores", "numero de gestacoes")),
    _field("mother.obstetric.vaginal_births", "Partos vaginais", "integer", "Histórico obstétrico",
           "Número de partos vaginais anteriores", DNV, ("partos vaginais", "n partos vaginais"),
           ("mother.obstetric.previous_pregnancies", "mother.obstetric.fetal_losses")),
    _field("mother.obstetric.cesareans", "Cesáreas", "integer", "Histórico obstétrico", "Número de cesáreas anteriores", DNV,
           ("cesareas", "n cesareas", "partos cesareos"), ("mother.obstetric.previous_pregnancies", "mother.obstetric.fetal_losses")),
    _field("mother.obstetric.live_births", "Nascidos vivos", "integer", "Histórico obstétrico",
           "Número de filhos nascidos vivos em gestações anteriores", DNV, ("nascidos vivos", "n nascidos vivos", "filhos vivos"),
           ("mother.obstetric.vaginal_births", "mother.obstetric.cesareans")),
    _field("mother.obstetric.fetal_losses", "Perdas fetais/abortos", "integer", "Histórico obstétrico",
           "Número de perdas fetais ou abortos anteriores", DNV, ("perdas fetais", "abortos", "n perdas fetais / abortos"),
           ("mother.obstetric.previous_pregnancies",)),
    _field("father.name", "Nome do pai", "name", "Pai", "Nome completo do pai", BIRTH_DOCS, ("nome do pai", "pai")),
    _field("father.birth_date", "Data de nascimento do pai", "date", "Pai", "Data de nascimento do pai, coerente com a idade",
           DNV, ("data de nascimento do pai", "nascimento do pai"), ("birth.date",)),
    _field("father.age", "Idade do pai", "integer", "Pai", "Idade do pai em anos completos na data do nascimento", DNV,
           ("idade do pai",), ("father.birth_date", "birth.date")),
    _field("guardian.name", "Nome do responsável legal", "name", "Responsável legal",
           "Nome do responsável legal pela mãe; opcional, quase sempre presente quando a mãe tem menos de 18 anos",
           DNV, ("responsavel legal", "nome responsavel legal", "nome do responsavel legal"), ("mother.name", "mother.age"), True),
    _field("guardian.age", "Idade do responsável legal", "integer", "Responsável legal",
           "Idade do responsável legal, sempre maior de idade; vazia quando não há responsável", DNV,
           ("idade responsavel legal", "idade do responsavel legal"), ("guardian.name",), True),
    _field("parents.names", "Filiação", "text", "Filiação", "Nomes da mãe e do pai no formato “mãe e pai”", CERT,
           ("filiacao", "pais", "nome dos pais"), ("mother.name", "father.name")),
    _field("grandparents.names", "Avós", "text", "Filiação", "Avós maternos e paternos no formato “A e B; C e D”", CERT,
           ("avos", "nome dos avos", "avos maternos", "avos paternos"), ("mother.name", "father.name")),
    _field("birth.date", "Data do nascimento", "date", "Nascimento", "Data em que ocorreu o nascimento", BIRTH_DOCS,
           ("data do nascimento", "data de nascimento", "data e hora do nascimento", "nascimento",
            "data de nascimento por extenso", "dia", "mes", "ano")),
    _field("birth.time", "Hora do nascimento", "time", "Nascimento", "Hora em que ocorreu o nascimento", BIRTH_DOCS,
           ("hora do nascimento", "hora de nascimento", "hora")),
    _field("birth.twins", "Gêmeos", "boolean", "Nascimento", "Indica nascimento gemelar (cerca de 2% dos registros)", CERT,
           ("gemeos", "nascimento gemelar")),
    _field("twin.name_matricula", "Nome e matrícula do gêmeo", "text", "Nascimento",
           "Nome do irmão gêmeo, com os mesmos sobrenomes, e sua matrícula; vazio quando não há gêmeos", CERT,
           ("nome e matricula dos gemeos", "gemeos nome e matricula"), ("birth.twins", "newborn.name", "registration.matricula"), True),
    _field("birth.establishment.name", "Estabelecimento do nascimento", "text", "Local do nascimento",
           "Nome do estabelecimento de saúde onde ocorreu o nascimento", DNV, ("estabelecimento", "nome do estabelecimento",
                                                                              "hospital", "maternidade")),
    _field("birth.establishment.cnes", "CNES do estabelecimento", "cnes", "Local do nascimento",
           "Código CNES sintético do estabelecimento", DNV, ("cnes", "codigo cnes"), ("birth.establishment.name", "birth.place.ibge")),
    *_address("birth.place", "Local do nascimento", "local do nascimento", BIRTH_DOCS,
              ("da ocorrencia", "de ocorrencia", "do nascimento"), ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge")),
    _field("birth.place.summary", "Local, município e UF do nascimento", "text", "Local do nascimento",
           "Texto “estabelecimento, município - UF”", CERT, ("local, municipio de nascimento e uf", "local de nascimento",
                                                             "local do nascimento"),
           ("birth.establishment.name", "birth.place.municipality", "birth.place.uf")),
    _field("pregnancy.weeks", "Semanas de gestação", "integer", "Gestação", "Duração da gestação em semanas completas", DNV,
           ("semanas de gestacao", "n semanas de gestacao", "idade gestacional")),
    _field("pregnancy.last_menstrual_period", "Data da última menstruação", "date", "Gestação",
           "DUM coerente com a data do nascimento e as semanas de gestação", DNV,
           ("data da ultima menstruacao", "dum", "ultima menstruacao"), ("birth.date", "pregnancy.weeks")),
    _field("pregnancy.prenatal_visits", "Consultas de pré-natal", "integer", "Gestação", "Número de consultas de pré-natal",
           DNV, ("consultas de pre-natal", "n de consultas de pre-natal", "consultas pre-natal")),
    _field("pregnancy.prenatal_start_month", "Mês de início do pré-natal", "integer", "Gestação",
           "Mês de gestação em que o pré-natal começou", DNV, ("mes de gestacao que iniciou o pre-natal", "inicio do pre-natal",
                                                                "mes de inicio do pre-natal")),
    _field("filling.date", "Data do preenchimento", "date", "Preenchimento", "Data de preenchimento da declaração", DNV,
           ("data do preenchimento", "data de preenchimento"), ("birth.date",)),
    _field("filling.responsible.name", "Responsável pelo preenchimento", "name", "Preenchimento",
           "Nome do responsável pelo preenchimento", DNV, ("responsavel pelo preenchimento", "nome do responsavel pelo preenchimento")),
    _field("filling.responsible.document_number", "Documento do responsável", "document_number", "Preenchimento",
           "Número do documento do responsável pelo preenchimento", DNV, ("n do documento", "numero do documento", "documento")),
    _field("filling.responsible.issuer", "Órgão emissor", "issuer", "Preenchimento", "Órgão emissor do documento do responsável",
           DNV, ("orgao emissor", "emissor")),
    _field("registration.date", "Data do registro", "date", "Registro civil", "Data do registro civil do nascimento", BIRTH_DOCS,
           ("data do registro", "data de registro", "data do registro por extenso"), ("birth.date", "filling.date")),
    _field("registration.number", "Número do registro", "document_number", "Registro civil",
           "Número do termo de registro, igual ao termo da matrícula", BIRTH_DOCS, ("registro", "numero do registro", "termo"),
           ("registration.matricula",)),
    _field("registration.matricula", "Matrícula", "matricula", "Registro civil",
           "Matrícula sintética de 32 algarismos no leiaute do registro civil", CERT, ("matricula",), ("registration.date",)),
    _field("registration.dnv_number", "Número da DNV", "dnv_number", "Registro civil",
           "Número sintético da Declaração de Nascido Vivo", BIRTH_DOCS,
           ("numero da dnv", "declaracao de nascido vivo", "dnv", "numero do documento", "numero_documento")),
    _field("registration.office.name", "Cartório", "text", "Registro civil", "Nome do cartório de registro civil", BIRTH_DOCS,
           ("cartorio", "nome do cartorio", "serventia")),
    _field("registration.office.municipality", "Município do cartório", "municipality", "Registro civil",
           "Município do cartório de registro", BIRTH_DOCS, ("municipio do cartorio", "municipio de registro", "municipio")),
    _field("registration.office.uf", "UF do cartório", "uf", "Registro civil", "UF do cartório de registro", BIRTH_DOCS,
           ("uf do cartorio", "uf de registro", "uf")),
    _field("registration.place.summary", "Município e UF de registro", "text", "Registro civil", "Texto “município - UF”",
           CERT, ("municipio de registro e unidade da federacao", "municipio de registro", "local de registro"),
           ("registration.office.municipality", "registration.office.uf")),
    _field("deceased.name", "Nome do falecido", "name", "Falecido", "Nome completo do falecido", DO,
           ("nome do falecido", "falecido", "nome do obito")),
    _field("deceased.sex", "Sexo do falecido", "sex", "Falecido", "Sexo do falecido (M ou F)", DO, ("sexo", "sexo do falecido")),
    _field("deceased.birth_date", "Data de nascimento do falecido", "date", "Falecido",
           "Data de nascimento coerente com a idade na data do óbito", DO, ("data de nascimento do falecido", "data de nascimento"),
           ("death.date",)),
    _field("deceased.age", "Idade do falecido", "integer", "Falecido", "Idade em anos completos na data do óbito", DO,
           ("idade do falecido", "idade"), ("deceased.birth_date", "death.date")),
    _field("deceased.cpf", "CPF do falecido", "cpf", "Falecido", "CPF sintético do falecido", DO, ("cpf do falecido", "cpf")),
    _field("deceased.mother_name", "Nome da mãe do falecido", "name", "Falecido", "Nome da mãe do falecido", DO,
           ("nome da mae", "mae do falecido")),
    _field("deceased.father_name", "Nome do pai do falecido", "name", "Falecido", "Nome do pai do falecido", DO,
           ("nome do pai", "pai do falecido")),
    _field("deceased.occupation", "Ocupação do falecido", "occupation", "Falecido", "Ocupação habitual do falecido", DO,
           ("ocupacao habitual", "ocupacao do falecido", "ocupacao")),
    *_address("deceased.residence", "Residência do falecido", "residência do falecido", DO,
              ("da residencia", "de residencia do falecido"), ("street", "number", "neighborhood", "cep", "municipality", "uf", "ibge")),
    _field("death.date", "Data do óbito", "date", "Óbito", "Data em que ocorreu o óbito", DO, ("data do obito", "data de obito",
                                                                                           "obito")),
    _field("death.time", "Hora do óbito", "time", "Óbito", "Hora em que ocorreu o óbito", DO, ("hora do obito", "hora")),
    _field("death.place.establishment.name", "Estabelecimento do óbito", "text", "Óbito",
           "Estabelecimento de saúde onde ocorreu o óbito", DO, ("estabelecimento", "local do obito", "estabelecimento do obito")),
    *_address("death.place", "Óbito", "local do óbito", DO, ("da ocorrencia", "de ocorrencia", "do obito"),
              ("municipality", "uf", "ibge")),
    _field("death_registration.date", "Data do registro do óbito", "date", "Registro do óbito",
           "Data do registro civil do óbito, nunca anterior ao óbito", DO, ("data do registro", "data de registro"), ("death.date",)),
    _field("death_registration.number", "Número do registro do óbito", "document_number", "Registro do óbito",
           "Número do termo de registro do óbito", DO, ("registro", "numero do registro", "termo")),
    _field("death_registration.office.name", "Cartório do registro do óbito", "text", "Registro do óbito",
           "Nome do cartório que registrou o óbito", DO, ("cartorio", "nome do cartorio", "serventia")),
    _field("death_registration.office.municipality", "Município do cartório do óbito", "municipality", "Registro do óbito",
           "Município do cartório que registrou o óbito", DO, ("municipio do cartorio", "municipio de registro", "municipio"),
           ("death_registration.office.uf",)),
    _field("death_registration.office.uf", "UF do cartório do óbito", "uf", "Registro do óbito",
           "UF do cartório que registrou o óbito", DO, ("uf do cartorio", "uf de registro", "uf"), ("death_registration.office.municipality",)),
    _field("death_registration.place.summary", "Município e UF do registro do óbito", "text", "Registro do óbito",
           "Texto “município - UF” do cartório do óbito", DO, ("municipio de registro e unidade da federacao", "local de registro"),
           ("death_registration.office.municipality", "death_registration.office.uf")),
    _field("certifier.name", "Médico atestante", "name", "Médico atestante", "Nome do médico que atesta o óbito", DO,
           ("medico atestante", "nome do medico", "atestante")),
    _field("certifier.registration", "CRM do médico", "document_number", "Médico atestante",
           "Número de registro no conselho profissional (CRM), de 4 a 6 algarismos", DO, ("crm", "registro no conselho")),
]
FIELDS.extend([
    _field("birth.pregnancy_type", "Tipo de gravidez", "text", "Nascimento", "Única ou dupla, coerente com a gemelaridade", DNV,
           ("tipo de gravidez",), ("birth.twins",)),
    _field("birth.delivery_type", "Tipo de parto", "text", "Nascimento", "Via de parto sintética", DNV, ("tipo de parto",)),
    _field("birth.cesarean_before_labor", "Cesárea antes do trabalho de parto", "text", "Nascimento",
           "Não se aplica no parto vaginal; ignorado quando o início do trabalho de parto não é informado", DNV,
           ("cesarea ocorreu antes do trabalho de parto iniciar",), ("birth.delivery_type",)),
    _field("newborn.anomaly_status", "Anomalia congênita", "text", "Recém-nascido",
           "Ignorado: o gerador não produz diagnósticos médicos", DNV, ("detectada alguma anomalia congenita",)),
    _field("newborn.anomaly_description", "Descrição das anomalias", "text", "Recém-nascido",
           "Vazio na ausência de diagnóstico validado", DNV, ("descrever todas as anomalias congenitas observadas",),
           ("newborn.anomaly_status",), optional=True),
    _field("mother.occupation_cbo", "CBO da ocupação da mãe", "document_number", "Mãe",
           "Código CBO apenas quando a ocupação tem correspondência documentada; vazio nos demais casos", DNV,
           ("codigo cbo", "cbo 2002"), ("mother.occupation",), optional=True),
])
REGISTRY: dict[str, SemanticField] = {field.key: field for field in FIELDS}
ENTITY_KEYS: dict[str, tuple[str, ...]] = {
    entity: tuple(field.key for field in FIELDS if field.entity == entity) for entity in dict.fromkeys(field.entity for field in FIELDS)
}


def semantic_field(key: str) -> SemanticField:
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"Chave semântica desconhecida: {key}") from None


def semantic_options() -> list[dict]:
    return [{"key": field.key, "label": field.label, "group": field.group, "kind": field.kind, "description": field.description}
            for field in FIELDS]


CLINICAL_TERMS = (
    "causa", "causas", "cause", "cid", "icd", "doenca", "doencas", "disease", "diagnostico", "diagnosticos", "diagnosis",
    "patologia", "patologias", "patologico", "morbido", "morbidos", "necropsia", "morte", "anomalia", "anomalias", "apgar",
    "peso", "perimetro", "comprimento", "apresentacao", "induzido", "assistencia medica", "lesao", "lesoes", "acidente",
    "suicidio", "homicidio", "violencia", "cirurgia", "tratamento", "obstetrica", "puerperio",
)
OBSTETRIC_TERMS = (
    "gravidez", "parto", "partos", "gestacao", "gestacoes", "gestante", "fetal", "fetais", "aborto", "abortos", "cesarea",
    "cesareas", "menstruacao",
)
MEDICAL_TERMS = CLINICAL_TERMS + OBSTETRIC_TERMS
