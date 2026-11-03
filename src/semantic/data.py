from __future__ import annotations

FIRST_NAMES = (
    "Ana", "Beatriz", "Camila", "Clara", "Daniel", "Eduardo", "Felipe",
    "Gabriel", "Helena", "Isabela", "João", "Júlia", "Lucas", "Mariana",
    "Miguel", "Rafael", "Sofia", "Valentina",
)
LAST_NAMES = (
    "Almeida", "Barbosa", "Cardoso", "Costa", "Ferreira", "Gomes",
    "Lima", "Martins", "Oliveira", "Pereira", "Ribeiro", "Rodrigues",
    "Santos", "Silva", "Souza",
)
STREETS = ("Rua das Flores", "Avenida Brasil", "Rua São José", "Rua das Acácias", "Avenida Central")
CITIES = ("São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Recife", "Salvador")
OCCUPATIONS = ("Professora", "Enfermeiro", "Agricultora", "Comerciante", "Motorista", "Engenheira",
               "Pedreiro", "Advogada", "Estudante", "Técnico em informática", "Auxiliar administrativo", "Costureira")
# Inflection by the person's sex (item 4): canonical keys stay as drawn/coded (unchanged
# OCCUPATION_CODES/OCCUPATION_MIN_SCHOOLING lookups), and generators.inflect_occupation()
# swaps to the other grammatical gender when it mismatches the record's sex. Entries not
# listed here (Comerciante, Motorista, Estudante...) are already gender-neutral in Portuguese.
OCCUPATION_GENDER = {
    "Professora": "F", "Professor": "M", "Enfermeiro": "M", "Enfermeira": "F",
    "Agricultora": "F", "Agricultor": "M", "Engenheira": "F", "Engenheiro": "M",
    "Pedreiro": "M", "Pedreira": "F", "Advogada": "F", "Advogado": "M",
    "Costureira": "F", "Costureiro": "M",
    "Técnico de enfermagem": "M", "Técnica de enfermagem": "F",
    "Psicólogo clínico": "M", "Psicóloga clínica": "F",
    "Assistente administrativo": "M", "Assistente administrativa": "F",
    "Técnico em informática": "M", "Técnica em informática": "F",
    "Auxiliar administrativo": "M", "Auxiliar administrativa": "F",
}
OCCUPATION_OPPOSITE_FORM = {
    "Professora": "Professor", "Professor": "Professora", "Enfermeiro": "Enfermeira", "Enfermeira": "Enfermeiro",
    "Agricultora": "Agricultor", "Agricultor": "Agricultora", "Engenheira": "Engenheiro", "Engenheiro": "Engenheira",
    "Pedreiro": "Pedreira", "Pedreira": "Pedreiro", "Advogada": "Advogado", "Advogado": "Advogada",
    "Costureira": "Costureiro", "Costureiro": "Costureira",
    "Técnico de enfermagem": "Técnica de enfermagem", "Técnica de enfermagem": "Técnico de enfermagem",
    "Psicólogo clínico": "Psicóloga clínica", "Psicóloga clínica": "Psicólogo clínico",
    "Assistente administrativo": "Assistente administrativa", "Assistente administrativa": "Assistente administrativo",
    "Técnico em informática": "Técnica em informática", "Técnica em informática": "Técnico em informática",
    "Auxiliar administrativo": "Auxiliar administrativa", "Auxiliar administrativa": "Auxiliar administrativo",
}
UF_VALUES = ("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO")
# Brazilian macro-regions, used to make interstate geography plausible (item 1): a cross-state
# residence/cartório is far more likely to land in a NEIGHBOURING state or another state's
# capital (e.g. referred for hospital care) than in an arbitrary, far-away one.
REGION_BY_UF = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste", "PE": "Nordeste",
    "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}
# Each profile's "street"/"cep"/"neighborhood" stay the primary address (unchanged defaults
# for existing callers); "streets" adds more generic street names per municipality, each with
# its own CEP sharing the municipality's CEP prefix (item 5) AND its own real/plausible
# bairro (m-1), so "another address in the same city" is no longer always the same street —
# nor always the profile's single bairro — while the (street, CEP) pairing stays consistent.
LOCATION_PROFILES = (
    {"cep": "01001000", "uf": "SP", "city": "São Paulo", "neighborhood": "Sé", "street": "Praça da Sé", "ibge": "3550308",
     "streets": (("Praça da Sé", "01001000", "Sé"), ("Rua Direita", "01002020", "Sé"), ("Avenida Ipiranga", "01046010", "República"),
                 ("Rua Augusta", "01305000", "Consolação"), ("Avenida Paulista", "01310100", "Bela Vista"), ("Rua da Consolação", "01301000", "Consolação"))},
    {"cep": "20040002", "uf": "RJ", "city": "Rio de Janeiro", "neighborhood": "Centro", "street": "Rua da Assembleia", "ibge": "3304557",
     "streets": (("Rua da Assembleia", "20040002", "Centro"), ("Rua do Ouvidor", "20040030", "Centro"), ("Avenida Rio Branco", "20040004", "Centro"),
                 ("Rua Uruguaiana", "20050090", "Centro"), ("Rua da Carioca", "20050008", "Centro"), ("Avenida Presidente Vargas", "20071003", "Centro"))},
    {"cep": "30140071", "uf": "MG", "city": "Belo Horizonte", "neighborhood": "Funcionários", "street": "Avenida Afonso Pena", "ibge": "3106200",
     "streets": (("Avenida Afonso Pena", "30140071", "Centro"), ("Rua da Bahia", "30160011", "Centro"), ("Avenida Amazonas", "30180001", "Centro"),
                 ("Rua Pernambuco", "30130151", "Funcionários"), ("Avenida do Contorno", "30140002", "Funcionários"), ("Rua Tupinambás", "30190002", "Centro"))},
    {"cep": "40020000", "uf": "BA", "city": "Salvador", "neighborhood": "Centro", "street": "Avenida Sete de Setembro", "ibge": "2927408",
     "streets": (("Avenida Sete de Setembro", "40020000", "Centro"), ("Rua Chile", "40020010", "Centro"), ("Rua Carlos Gomes", "40060330", "Centro"),
                 ("Avenida Oceânica", "40170010", "Barra"), ("Rua Marquês de Caravelas", "40070110", "Barra"), ("Avenida Tancredo Neves", "41820021", "Caminho das Árvores"))},
    {"cep": "50030000", "uf": "PE", "city": "Recife", "neighborhood": "Santo Antônio", "street": "Rua do Imperador", "ibge": "2611606",
     "streets": (("Rua do Imperador", "50030000", "Santo Antônio"), ("Rua Nova", "50030170", "Santo Antônio"), ("Avenida Guararapes", "50030240", "Santo Antônio"),
                 ("Rua da Aurora", "50050000", "Boa Vista"), ("Avenida Conde da Boa Vista", "50060004", "Boa Vista"), ("Rua do Bom Jesus", "50030280", "Recife"))},
    {"cep": "60025061", "uf": "CE", "city": "Fortaleza", "neighborhood": "Centro", "street": "Rua Major Facundo", "ibge": "2304400",
     "streets": (("Rua Major Facundo", "60025061", "Centro"), ("Rua Senador Pompeu", "60025000", "Centro"), ("Avenida Duque de Caxias", "60015120", "Centro"),
                 ("Rua Guilherme Rocha", "60030000", "Centro"), ("Avenida Beira Mar", "60165121", "Meireles"), ("Rua Barão do Rio Branco", "60055090", "Centro"))},
    {"cep": "70040902", "uf": "DF", "city": "Brasília", "neighborhood": "Asa Sul", "street": "SBS Quadra 2", "ibge": "5300108",
     "streets": (("SBS Quadra 2", "70040902", "Asa Sul"), ("SQS 108", "70347050", "Asa Sul"), ("SCS Quadra 6", "70306000", "Asa Sul"),
                 ("SHIS QI 11", "71625000", "Lago Sul"), ("SQN 404", "70847040", "Asa Norte"), ("SGAS 905", "70390050", "Asa Sul"))},
    {"cep": "80010010", "uf": "PR", "city": "Curitiba", "neighborhood": "Centro", "street": "Rua XV de Novembro", "ibge": "4106902",
     "streets": (("Rua XV de Novembro", "80010010", "Centro"), ("Rua Marechal Deodoro", "80010020", "Centro"), ("Avenida João Gualberto", "80030000", "Alto da Glória"),
                 ("Rua Comendador Araújo", "80420000", "Batel"), ("Avenida Cândido de Abreu", "80230010", "Centro Cívico"), ("Rua Mateus Leme", "80510030", "São Francisco"))},
    {"cep": "90010000", "uf": "RS", "city": "Porto Alegre", "neighborhood": "Centro Histórico", "street": "Rua dos Andradas", "ibge": "4314902",
     "streets": (("Rua dos Andradas", "90010000", "Centro Histórico"), ("Avenida Borges de Medeiros", "90020021", "Centro Histórico"), ("Rua Sete de Setembro", "90010190", "Centro Histórico"),
                 ("Rua General Câmara", "90010230", "Centro Histórico"), ("Avenida Osvaldo Aranha", "90035190", "Bom Fim"), ("Rua Riachuelo", "90010273", "Centro Histórico"))},
    {"cep": "88010001", "uf": "SC", "city": "Florianópolis", "neighborhood": "Centro", "street": "Rua Felipe Schmidt", "ibge": "4205407",
     "streets": (("Rua Felipe Schmidt", "88010001", "Centro"), ("Avenida Hercílio Luz", "88010002", "Centro"), ("Rua Tenente Silveira", "88010300", "Centro"),
                 ("Rua Trajano", "88010010", "Centro"), ("Avenida Beira Mar Norte", "88015700", "Centro"), ("Rua Conselheiro Mafra", "88010100", "Centro"))},
    {"cep": "74003010", "uf": "GO", "city": "Goiânia", "neighborhood": "Setor Central", "street": "Avenida Goiás", "ibge": "5208707",
     "streets": (("Avenida Goiás", "74003010", "Setor Central"), ("Rua 3", "74013010", "Setor Central"), ("Avenida Tocantins", "74015010", "Setor Central"),
                 ("Rua 44", "74115110", "Setor Bueno"), ("Avenida 24 de Outubro", "74055110", "Setor Central"), ("Rua 4", "74023010", "Setor Central"))},
    {"cep": "69005040", "uf": "AM", "city": "Manaus", "neighborhood": "Centro", "street": "Avenida Eduardo Ribeiro", "ibge": "1302603",
     "streets": (("Avenida Eduardo Ribeiro", "69005040", "Centro"), ("Rua Saldanha Marinho", "69010040", "Centro"), ("Avenida Joaquim Nabuco", "69020030", "Centro"),
                 ("Rua Barroso", "69010180", "Centro"), ("Avenida Djalma Batista", "69050010", "Chapada"), ("Rua Dez de Julho", "69025030", "Centro"))},
)

FEMALE_ASSOCIATED_FIRST_NAMES = (
    "Ana", "Beatriz", "Camila", "Clara", "Helena", "Isabela", "Júlia", "Mariana", "Sofia", "Valentina",
    "Alice", "Aline", "Amanda", "Bruna", "Carolina", "Cecília", "Daniela", "Fernanda", "Gabriela", "Giovana",
    "Heloísa", "Jéssica", "Larissa", "Laura", "Letícia", "Lívia", "Luana", "Manuela", "Maria", "Patrícia",
    "Rafaela", "Renata", "Sandra", "Tatiane", "Vitória",
)
MALE_ASSOCIATED_FIRST_NAMES = (
    "Daniel", "Eduardo", "Felipe", "Gabriel", "João", "Lucas", "Miguel", "Rafael",
    "Antônio", "Arthur", "Bernardo", "Bruno", "Carlos", "Davi", "Diego", "Fábio", "Francisco", "Guilherme",
    "Gustavo", "Heitor", "Henrique", "José", "Leonardo", "Marcelo", "Matheus", "Paulo", "Pedro", "Ricardo",
    "Rodrigo", "Samuel", "Thiago", "Vinícius",
)
SURNAMES = LAST_NAMES + (
    "Araújo", "Batista", "Campos", "Carvalho", "Castro", "Correia", "Dias", "Fernandes", "Freitas", "Lopes",
    "Machado", "Mendes", "Moreira", "Nascimento", "Nunes", "Pinto", "Ramos", "Rocha", "Teixeira", "Vieira",
)
ESTABLISHMENT_NAMES = (
    "Hospital Municipal", "Hospital Regional", "Hospital Geral", "Hospital Universitário", "Maternidade Municipal",
    "Hospital e Maternidade São José", "Maternidade Nossa Senhora da Conceição", "Santa Casa de Misericórdia",
    "Hospital Santa Luzia", "Hospital Materno-Infantil", "Hospital São Lucas", "Maternidade Santa Mônica",
)
# Adult-admission establishments only (no maternity units) — used for the DO,
# whose deaths in phase 1 are all adults; ESTABLISHMENT_NAMES above (DNV-oriented,
# ~43% maternity) stays for birth documents.
ADULT_ESTABLISHMENT_NAMES = (
    "Hospital Municipal", "Hospital Regional", "Hospital Geral", "Hospital Universitário",
    "Santa Casa de Misericórdia", "Hospital Santa Luzia", "Hospital São Lucas", "Hospital das Clínicas",
    "Hospital de Pronto Socorro", "Hospital São Vicente de Paulo", "Hospital Nossa Senhora Auxiliadora",
    "Hospital Evangélico",
)
# Field-20 category 2 ("outro estabelecimento de saúde") (item 2): these are NOT hospitals —
# UPA, UBS, policlínica and pronto atendimento are the establishments that actually certify a
# death without admission. ADULT_ESTABLISHMENT_NAMES/ESTABLISHMENT_NAMES above (all "Hospital"/
# "Maternidade"/"Santa Casa") are category 1 only.
OTHER_HEALTH_ESTABLISHMENT_NAMES = (
    "UPA 24h Municipal", "UPA 24h Central", "UBS Central", "UBS Jardim das Flores",
    "Policlínica Municipal", "Policlínica Regional", "Pronto Atendimento Municipal",
    "Pronto Atendimento 24 Horas", "UBS Vila Nova",
)
REGISTRY_OFFICE_NAMES = (
    "Cartório de Registro Civil das Pessoas Naturais", "1º Ofício de Registro Civil das Pessoas Naturais",
    "2º Ofício de Registro Civil das Pessoas Naturais", "Cartório do 1º Subdistrito", "Cartório do 2º Subdistrito",
    "Registro Civil das Pessoas Naturais da Sede", "Serventia de Registro Civil do Distrito Sede",
)
ISSUERS = ("SSP", "SESP", "SDS", "PC", "IFP", "DETRAN", "SJS", "IGP")
COMPLEMENTS = ("Apto 101", "Apto 302", "Bloco B", "Casa 2", "Fundos", "Sala 4", "Casa", "Térreo")
PROFILES_BY_IBGE = {profile["ibge"]: profile for profile in LOCATION_PROFILES}

# CBO 2002 codes checked against the official titles (buscadorncm.com.br/cbo, 2026-11-03).
OCCUPATION_CODES = {
    "Professora": "231210",       # Professor de nível superior do ensino fundamental (primeira a quarta série) (CBO 2312-10)
    "Enfermeiro": "223505",       # Enfermeiro (CBO 2235-05)
    "Agricultora": "621005",      # Trabalhador agropecuário em geral (CBO 6210-05)
    "Comerciante": "141410",      # Comerciante varejista (CBO 1414-10)
    "Motorista": "782305",        # Motorista de carro de passeio (CBO 7823-05)
    "Engenheira": "214205",       # Engenheiro civil (CBO 2142-05)
    "Pedreiro": "715210",         # Pedreiro (CBO 7152-10)
    "Advogada": "241005",         # Advogado (CBO 2410-05)
    "Técnico em informática": "317210",     # Técnico de suporte ao usuário de tecnologia da informação (CBO 3172-10)
    "Auxiliar administrativo": "411005",    # Auxiliar de escritório/administrativo (CBO 4110-05)
    "Costureira": "763210",       # Costureiro na confecção em série (CBO 7632-10)
    "Técnico de enfermagem": "322205",      # Técnico de enfermagem (CBO 3222-05)
    "Psicólogo clínico": "251510",          # Psicólogo clínico (CBO 2515-10)
    "Assistente administrativo": "411010",  # Assistente administrativo (CBO 4110-10)
    # "Estudante", "Não se aplica" (and real-world "Do lar"/"Aposentado"/"Desempregado", not
    # modeled here) are not CBO-2002 occupations: no entry here means the CBO field stays
    # blank for them (occupation_cbo_for returns None), matching the DO/DNV instructions.
}


def occupation_cbo_for(name: str | None) -> str | None:
    """CBO code for an occupation title, whichever grammatical gender it was inflected to
    (item 4) — the code is the same regardless of the displayed form."""
    if not name:
        return None
    return OCCUPATION_CODES.get(name) or OCCUPATION_CODES.get(OCCUPATION_OPPOSITE_FORM.get(name, ""))


# Congenital anomalies recorded on the DNV (field 41) when field 6 ("Detectada alguma
# anomalia congênita?") is "Sim": (description, ICD-10 code, sex restriction or None, base
# weight). Weights are loosely calibrated to the relative frequency of these categories among
# SINASC-recorded anomalies; hypospadias is male-only (urethral meatus anomaly) and
# anencephaly is kept rare (largely incompatible with live birth / registration). Down
# syndrome's weight is boosted for older mothers (maternal-age-related trisomy 21 risk) by
# the caller.
ANOMALIES = (
    ("Pé torto congênito", "Q66.8", None, 25),
    ("Polidactilia", "Q69.9", None, 15),
    ("Espinha bífida", "Q05.9", None, 10),
    ("Anencefalia", "Q00.0", None, 2),
    ("Fissura labial e palatina", "Q37.9", None, 12),
    ("Síndrome de Down", "Q90.9", None, 8),
    ("Hidrocefalia congênita", "Q03.9", None, 8),
    ("Hipospádia", "Q54.9", "M", 12),
    ("Cardiopatia congênita", "Q24.9", None, 15),
)
ANOMALY_CODES = frozenset(code for _, code, _, _ in ANOMALIES)


OCCUPATION_CODE_SOURCES = (
    "https://bvsms.saude.gov.br/bvs/saudelegis/sas/2009/prt0416_24_11_2009.html",
    "https://www.gov.br/conab/pt-br/acesso-a-informacao/licitacoes-e-contratos/editais-de-chamamento/suregs/se/edital-processo-seletivo-programa-jovem-aprendiz-n-o-02-2023-conab-se",
)
DNV_CATEGORIES_SOURCE = "https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/vigilancia/declaracao-de-nascido-vivo-manual-de-instrucoes-para-preenchimento"
