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
UF_VALUES = ("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO")
LOCATION_PROFILES = (
    {"cep": "01001000", "uf": "SP", "city": "São Paulo", "neighborhood": "Sé", "street": "Praça da Sé", "ibge": "3550308"},
    {"cep": "20040002", "uf": "RJ", "city": "Rio de Janeiro", "neighborhood": "Centro", "street": "Rua da Assembleia", "ibge": "3304557"},
    {"cep": "30140071", "uf": "MG", "city": "Belo Horizonte", "neighborhood": "Funcionários", "street": "Avenida Afonso Pena", "ibge": "3106200"},
    {"cep": "40020000", "uf": "BA", "city": "Salvador", "neighborhood": "Centro", "street": "Avenida Sete de Setembro", "ibge": "2927408"},
    {"cep": "50030000", "uf": "PE", "city": "Recife", "neighborhood": "Santo Antônio", "street": "Rua do Imperador", "ibge": "2611606"},
    {"cep": "60025061", "uf": "CE", "city": "Fortaleza", "neighborhood": "Centro", "street": "Rua Major Facundo", "ibge": "2304400"},
    {"cep": "70040902", "uf": "DF", "city": "Brasília", "neighborhood": "Asa Sul", "street": "SBS Quadra 2", "ibge": "5300108"},
    {"cep": "80010010", "uf": "PR", "city": "Curitiba", "neighborhood": "Centro", "street": "Rua XV de Novembro", "ibge": "4106902"},
    {"cep": "90010000", "uf": "RS", "city": "Porto Alegre", "neighborhood": "Centro Histórico", "street": "Rua dos Andradas", "ibge": "4314902"},
    {"cep": "88010001", "uf": "SC", "city": "Florianópolis", "neighborhood": "Centro", "street": "Rua Felipe Schmidt", "ibge": "4205407"},
    {"cep": "74003010", "uf": "GO", "city": "Goiânia", "neighborhood": "Setor Central", "street": "Avenida Goiás", "ibge": "5208707"},
    {"cep": "69005040", "uf": "AM", "city": "Manaus", "neighborhood": "Centro", "street": "Avenida Eduardo Ribeiro", "ibge": "1302603"},
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
REGISTRY_OFFICE_NAMES = (
    "Cartório de Registro Civil das Pessoas Naturais", "1º Ofício de Registro Civil das Pessoas Naturais",
    "2º Ofício de Registro Civil das Pessoas Naturais", "Cartório do 1º Subdistrito", "Cartório do 2º Subdistrito",
    "Registro Civil das Pessoas Naturais da Sede", "Serventia de Registro Civil do Distrito Sede",
)
ISSUERS = ("SSP", "SESP", "SDS", "PC", "IFP", "DETRAN", "SJS", "IGP")
COMPLEMENTS = ("Apto 101", "Apto 302", "Bloco B", "Casa 2", "Fundos", "Sala 4", "Casa", "Térreo")
PROFILES_BY_IBGE = {profile["ibge"]: profile for profile in LOCATION_PROFILES}

OCCUPATION_CODES = {
    "Técnico de enfermagem": "322205",
    "Psicólogo clínico": "251510",
    "Assistente administrativo": "411010",
}
OCCUPATION_CODE_SOURCES = (
    "https://bvsms.saude.gov.br/bvs/saudelegis/sas/2009/prt0416_24_11_2009.html",
    "https://www.gov.br/conab/pt-br/acesso-a-informacao/licitacoes-e-contratos/editais-de-chamamento/suregs/se/edital-processo-seletivo-programa-jovem-aprendiz-n-o-02-2023-conab-se",
)
DNV_CATEGORIES_SOURCE = "https://www.gov.br/saude/pt-br/centrais-de-conteudo/publicacoes/svsa/vigilancia/declaracao-de-nascido-vivo-manual-de-instrucoes-para-preenchimento"
