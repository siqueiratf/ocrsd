# Changelog

## [Não publicado]

### Interface web

- Interface migrada de React/Vite para páginas FastAPI com Jinja2 e HTMX,
  renderizadas no servidor, sem etapa de build nem dependências Node.
- Navegação horizontal, visual LabREC revisado, layout responsivo e componentes
  consistentes para modelos, editor, geração, resultados, presets e compartilhamento.
- Espaço privado mantido em cookie HttpOnly, com migração automática do token salvo
  pelas versões anteriores e dos links de compartilhamento antigos.
- Nginx do Compose passa a encaminhar toda a interface para a API; testes da interface
  web passam a usar `unittest`.

## [V1.1.0] — 2026-10-10

Evolução do OCRSD para uma plataforma web modular, preservando a CLI, a interface
desktop e o motor existente de geração sintética.

### Plataforma web

- Interface responsiva em React e TypeScript para escolher documento, mapa, campos,
  quantidade, semente, escrita e filtros, acompanhar progresso e visualizar PDFs.
- API FastAPI versionada com catálogo, templates, mapas, capacidades, presets,
  validação, jobs, cancelamento, resultados e artefatos protegidos por chave.
- Downloads de datasets em ZIP com PDFs, ground truth e manifesto, além de links
  temporários de compartilhamento que expiram junto com o job.
- Modelos locais por fonte, sjvasquez, aaron1259 e wpvnteam disponíveis quando seus
  pesos completos são instalados pelo administrador.

### Processamento e segurança

- Serviço reutilizável de geração que chama o motor existente sem depender de
  Tkinter ou do protocolo HTTP.
- Worker separado com fila SQLite, progresso persistente, cancelamento, timeout,
  recuperação após reinício, cotas, retenção e isolamento dos artefatos por job.
- Identificadores administrados no lugar de caminhos fornecidos pelo cliente,
  validação estrita, limites de corpo, memória, armazenamento e requisições.
- Documentos públicos marcados como sintéticos e sem validade legal no PDF, no
  ground truth e no manifesto.

### Operação e qualidade

- Docker Compose com frontend Nginx, API, worker, volume persistente, health checks,
  filesystems somente leitura, limites de recursos e políticas de reinício.
- Configurações para Nginx com HTTPS, unidades systemd e alternativa de implantação
  sem Docker, com procedimentos de atualização, backup e diagnóstico.
- Collection Postman v2.1 gerada a partir do OpenAPI, ambientes local e remoto e
  fluxo completo de geração, consulta, preview, download e cancelamento.
- Testes de serviço e contrato da API, concorrência, falhas, isolamento,
  determinismo e fluxos essenciais do frontend incorporados ao CI e ao release.

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
