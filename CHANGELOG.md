# Changelog

## [V1] — 2026-10-07

Primeira versão organizada do OCRSD, identificada no pacote Python como `1.0.0`.

### Funcionalidades

- CLI para processamento de PDFs, comparação OCR, pré-processamento e geração sintética.
- Interface desktop com mapeador de campos, visualizador, editor e ferramentas de datasets.
- Alinhamento por template, detecção por palavras ou campos, recortes e PDF com conteúdo ocultado.
- Catálogo com template e mapa oficial da DNV (69 campos) e certidão de nascimento (19 campos).
- Escrita sintética por fonte ou modelos locais sjvasquez, aaron1259 e wpvnteam.
- Filtros de digitalização e presets compartilhados de escrita e pré-processamento.

### Correções

- Carregamento do mapa oficial da DNV sem exigir rótulos ausentes; a descrição passa a identificar o campo.
- Ajuste da fonte ao espaço disponível para impedir textos omitidos em caixas pequenas.
- Remoção de um preset de imagem duplicado que era apresentado como mapa de campos.

### Organização

- Configuração central em `src/config.py` e documentação técnica em `docs/`.
- Dependências consolidadas em `pyproject.toml` e `uv.lock`.
- Remoção de comentários e docstrings explicativas dos arquivos de código e fontes do relatório.
- Remoção de `DNV_58027`, conteúdo de `examples/`, arquivos ZIP redundantes e implementações sem uso.
- Exclusão de `.idea` e dos resultados gerados do versionamento; regras correspondentes no `.gitignore`.
- Histórico de desenvolvimento consolidado em commits por finalidade.

### Validação

- Testes de regressão para mapas sem rótulos e textos em campos pequenos.
- `make check` e `make smoke` geram e processam uma DNV e uma certidão de nascimento.
- Verificação visual de todos os campos gerados, alinhamento, detecção, recortes e ocultação no PDF.
- GitHub Actions em pushes, pull requests e semanalmente, com artefatos da validação.
- Publicação automática das notas e arquivos do release a partir da tag `V1`.
