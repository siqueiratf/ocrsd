import re

from src.field_rules import (COMPOSITE_TYPES, DATE_FORMAT_LABELS, DATE_PARTS, DNV_FORMATS, DNV_TEST_PREFIX, DOCUMENT_KINDS, FIELD_TYPES,
                              SEX_FORMATS, rule_errors)
from src.semantic import semantic_options
from src.web.services import ServiceError

UF_VALUES = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE",
             "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
TYPE_LABELS = {"text": "Texto", "integer": "Número", "decimal": "Decimal", "date": "Data", "time": "Hora",
               "checkbox": "Checkbox", "enum": "Lista de opções", "person_name": "Nome completo",
               "occupation": "Profissão", "document": "Documento", "signature": "Assinatura", "cep": "CEP",
               "uf": "UF", "municipality": "Município", "address": "Localização",
               "digit_sequence": "Sequência de dígitos", "text_sequence": "Sequência de letras", "cns": "Cartão SUS",
               "sex": "Sexo", "dnv_number": "Número da DNV"}
TYPE_ICONS = {"text": "type-text", "integer": "type-number", "decimal": "type-decimal", "date": "calendar",
              "time": "clock", "checkbox": "checkbox", "enum": "list", "person_name": "person", "occupation": "briefcase",
              "document": "id-card", "signature": "signature", "cep": "mail", "uf": "flag", "municipality": "building",
              "address": "pin", "digit_sequence": "digits", "text_sequence": "letters", "cns": "card", "sex": "person",
              "dnv_number": "id-card"}
CATEGORIES = {"basicos": "Campos básicos", "pessoais": "Dados pessoais", "geograficos": "Dados geográficos",
              "estruturados": "Campos estruturados"}
TYPE_CATEGORY = {"text": "basicos", "integer": "basicos", "decimal": "basicos", "date": "basicos", "time": "basicos",
                 "checkbox": "basicos", "enum": "basicos", "person_name": "pessoais", "occupation": "pessoais",
                 "document": "pessoais", "signature": "pessoais", "cep": "geograficos", "uf": "geograficos",
                 "municipality": "geograficos", "address": "geograficos", "digit_sequence": "estruturados",
                 "text_sequence": "estruturados", "cns": "estruturados", "sex": "pessoais", "dnv_number": "estruturados"}


def _item(key, title, subtitle, data_type, search, size, validation=None, **extra):
    config = {"data_type": data_type, **({"validation": validation} if validation else {}), **extra}
    return {"key": key, "title": title, "subtitle": subtitle, "icon": TYPE_ICONS[data_type], "search": search,
            "category": TYPE_CATEGORY[data_type], "config": config, "size": size}


PALETTE = [
    {"id": "basicos", "title": CATEGORIES["basicos"], "items": [
        _item("texto", "Texto", "Conteúdo textual livre", "text", "texto livre palavra frase observacao", [900, 70]),
        _item("numero", "Número", "Inteiros com limites", "integer", "numero inteiro quantidade idade", [300, 70],
              {"min_value": 0, "max_value": 999}),
        _item("decimal", "Decimal", "Valores com casas decimais", "decimal", "decimal peso medida valor", [300, 70],
              {"min_value": 0, "max_value": 100, "decimals": 2}),
        _item("data", "Data", "Formatos com barras, hífens, por extenso ou separados", "date",
              "data dia mes ano nascimento registro extenso separada", [420, 70], {"format": "DD/MM/AAAA"}),
        _item("hora", "Hora", "Horários no formato HH:MM", "time", "hora horario tempo", [260, 70]),
        _item("checkbox", "Checkbox", "Opções marcáveis", "checkbox", "checkbox marcar opcao caixa sexo sim nao", [520, 70],
              selection_mode="single"),
        _item("lista", "Lista de opções", "Um valor de lista fechada", "enum", "lista enum opcoes valores selecao", [520, 70],
              {"allowed_values": ["Opção 1", "Opção 2"]}),
    ]},
    {"id": "pessoais", "title": CATEGORIES["pessoais"], "items": [
        _item("nome", "Nome completo", "Nomes de pessoas", "person_name", "nome pessoa mae pai registrado declarante",
              [1100, 70], {"allow_digits": False, "portuguese_rules": True, "language": "pt-BR"}),
        _item("profissao", "Profissão", "Ocupação ou atividade", "occupation", "profissao ocupacao trabalho cargo", [700, 70]),
        _item("documento", "Documento", "CPF, RG ou CNS", "document", "documento cpf rg identidade registro", [560, 70],
              {"document_kind": "cpf"}),
        _item("assinatura", "Assinatura", "Região de assinatura", "signature", "assinatura rubrica firma", [900, 120]),
        _item("sexo", "Sexo", "Masculino/Feminino ou M/F", "sex", "sexo genero masculino feminino m f", [360, 70],
              {"format": "completo"}),
    ]},
    {"id": "geograficos", "title": CATEGORIES["geograficos"], "items": [
        _item("cep", "CEP", "Código postal brasileiro", "cep", "cep codigo postal endereco", [360, 70]),
        _item("uf", "UF", "Unidade federativa", "uf", "uf estado unidade federativa", [150, 70],
              {"allowed_values": UF_VALUES, "max_edit_distance": 1}),
        _item("municipio", "Município", "Cidade ou município", "municipality", "municipio cidade naturalidade localidade", [700, 70]),
        _item("localizacao", "Localização", "Endereço ou logradouro", "address", "localizacao endereco logradouro rua residencia local",
              [1100, 70]),
    ]},
    {"id": "estruturados", "title": CATEGORIES["estruturados"], "items": [
        {**_item("digitos", "Sequência de dígitos", "Um dígito por caixa", "digit_sequence",
                 "sequencia digitos caixas numero codigo", [640, 70]), "boxes": 8},
        {**_item("letras", "Sequência de letras", "Uma letra por caixa", "text_sequence", "sequencia letras caixas iniciais",
                 [640, 70], {"pattern": "^[A-Z]+$"}), "boxes": 8},
        _item("cns", "Cartão SUS", "Cartão Nacional de Saúde", "cns", "cns sus cartao saude", [800, 70]),
        _item("dnv", "Número da DNV", "Identificador sintético de teste", "dnv_number",
              "dnv declaracao nascido vivo numero sinasc identificador", [700, 70], {"format": "agrupado", "prefix": DNV_TEST_PREFIX}),
    ]},
]
LAYOUT_PALETTE = {"id": "layout", "title": "Layout do documento", "items": [
    {"key": "layout-texto", "title": "Texto fixo", "subtitle": "Títulos e rótulos impressos", "icon": "heading",
     "search": "texto fixo titulo rotulo cabecalho", "category": "layout", "size": [1000, 90],
     "layout": {"kind": "text", "text": "Texto fixo", "font_size": 12, "bold": False, "align": "left"}},
    {"key": "layout-linha", "title": "Linha", "subtitle": "Separador horizontal ou vertical", "icon": "line",
     "search": "linha separador divisoria", "category": "layout", "size": [1200, 24], "layout": {"kind": "line", "stroke": 1}},
    {"key": "layout-caixa", "title": "Caixa", "subtitle": "Moldura retangular", "icon": "frame",
     "search": "caixa moldura retangulo quadro borda", "category": "layout", "size": [900, 260],
     "layout": {"kind": "box", "stroke": 1, "fill": False}},
]}
CUSTOM_BASES = [item for item in FIELD_TYPES if item not in COMPOSITE_TYPES]
GENERATION_RULES = {"padrao": "Padrão do tipo base", "lista": "Lista de valores", "intervalo": "Intervalo numérico",
                    "mascara": "Máscara de caracteres"}
SLUG = re.compile(r"^[a-z0-9_]{2,40}$")


def _number(value: str, label: str):
    try:
        return float(value) if value.strip() else None
    except ValueError as exc:
        raise ServiceError(422, "validation_error", f"Confira o valor de “{label}”") from exc


def custom_field_config(form) -> dict:
    def text(key: str) -> str:
        value = form.get(key)
        return value.strip() if isinstance(value, str) else ""

    name, slug, base, rule = text("name"), text("slug").lower(), text("base_type"), text("rule") or "padrao"
    if not 1 <= len(name) <= 60:
        raise ServiceError(422, "validation_error", "Informe um nome com até 60 caracteres")
    if not SLUG.match(slug):
        raise ServiceError(422, "validation_error", "Use de 2 a 40 letras minúsculas, números ou “_” no identificador")
    if base not in CUSTOM_BASES:
        raise ServiceError(422, "validation_error", "Escolha um tipo base suportado")
    if rule not in GENERATION_RULES:
        raise ServiceError(422, "validation_error", "Escolha uma regra de geração suportada")
    validation = {}
    if rule == "lista":
        values = list(dict.fromkeys(line.strip() for line in text("allowed_values").splitlines() if line.strip()))
        if not values:
            raise ServiceError(422, "validation_error", "Informe ao menos um valor permitido, um por linha")
        validation["allowed_values"] = values
    elif rule == "intervalo":
        if base not in {"integer", "decimal"}:
            raise ServiceError(422, "validation_error", "O intervalo numérico exige o tipo base Número ou Decimal")
        low, high = _number(text("min_value"), "Valor mínimo"), _number(text("max_value"), "Valor máximo")
        if low is None or high is None:
            raise ServiceError(422, "validation_error", "Informe os valores mínimo e máximo")
        validation.update(min_value=int(low) if base == "integer" else low, max_value=int(high) if base == "integer" else high)
    elif rule == "mascara":
        validation["mask"] = text("mask")
    if base == "date" and text("format"):
        validation["format"] = text("format")
    if base == "document":
        validation["document_kind"] = text("document_kind") or "cpf"
    if base == "decimal" and text("decimals"):
        decimals = _number(text("decimals"), "Casas decimais")
        validation["decimals"] = int(decimals)
    blank = _number(text("blank_percent"), "Probabilidade de campo vazio")
    if blank:
        validation["blank_probability"] = round(blank / 100, 4)
    errors = rule_errors(base, validation)
    if errors:
        raise ServiceError(422, "validation_error", errors[0])
    config = {"data_type": base, "custom_type": slug, **({"validation": validation} if validation else {})}
    if form.get("printed") is not None:
        config["render_as_printed"] = True
    return {"name": name, "slug": slug, "description": text("description")[:200], "base_type": base, "config": config}


def custom_palette(field_types: list[dict]) -> dict:
    return {"id": "personalizados", "title": "Personalizados", "items": [
        {"key": f'tipo-{item["slug"]}', "title": item["name"], "subtitle": item["description"] or TYPE_LABELS[item["base_type"]],
         "icon": "sparkle", "search": f'{item["name"]} {item["slug"]} {TYPE_LABELS[item["base_type"]]} personalizado',
         "category": "personalizados", "config": item["config"], "size": [700, 70]}
        for item in field_types]}


ICON_NAMES = sorted({*TYPE_ICONS.values(), "heading", "line", "frame", "sparkle", "external"})
EDITOR_OPTIONS = {"type_labels": TYPE_LABELS, "type_category": TYPE_CATEGORY, "type_icons": TYPE_ICONS, "uf_values": UF_VALUES,
                  "date_formats": list(DATE_FORMAT_LABELS.items()), "date_parts": [[key, label] for key, (label, _) in DATE_PARTS.items()],
                  "sex_formats": list(SEX_FORMATS.items()), "dnv_formats": list(DNV_FORMATS.items()), "dnv_prefix": DNV_TEST_PREFIX,
                  "document_kinds": DOCUMENT_KINDS, "semantic_options": semantic_options()}
