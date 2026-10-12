# OCRSD Web — Briefing consolidado de produto, UX e arquitetura

**Versão do pacote:** 2026-10-10  
**Repositório de referência:** https://github.com/siqueiratf/ocrsd  
**Instituição e produto:** LabREC — Laboratório do Registro Civil | OCRSD  
**Natureza:** especificação proposta + protótipo de interface, **não** uma implementação de API ou frontend em produção.

## 1. Objetivo e escopo

Transformar OCRSD, hoje uma ferramenta Python/Tkinter com CLI, em plataforma web para criar templates e mapas de campos de documentos civis brasileiros, gerar datasets sintéticos e visualizar, salvar, reproduzir e compartilhar os resultados. Casos ilustrativos: Declaração de Nascido Vivo (DNV), certidão de nascimento; outras tipologias como certidão de óbito podem ser adicionadas quando houver templates e mapas consistentes. Finalidade educacional, pesquisa e treinamento de modelos de processamento de documentos no contexto do Registro Civil brasileiro. Não emitir, simular autenticidade oficial nem apresentar artefatos como documentos de registro válidos.

**Tese de produto:** o OCRSD não deve se apresentar como mera ferramenta de OCR. A UX deve enfatizar o fluxo **Modelar → Gerar → Explorar/Compartilhar**. Ferramentas de comparação OCR, debug, editor genérico e processamento experimental não precisam ocupar o menu primário.

## 2. Evidência e limites

**Base observada:** código do repositório acessível pelo conector GitHub e oito screenshots do desktop fornecidos pelo usuário. Módulos importantes incluem `src/gui.py`, `src/field_mapper_gui.py`, `src/synthetic_dataset_gui.py`, `src/synthetic_dataset.py`, `src/pipeline.py`, `src/pipeline_gui.py`, `src/generated_pdf_viewer_gui.py`, `src/handwriting_gallery_gui.py`, `src/document_filters_gui.py`, `src/document_types.py`, `src/writing_presets.py`, `src/cli.py`; configuração em `pyproject.toml`; templates em `documents/`; testes em `tests/`. Confirmar o estado atual do código antes de implementar porque o repositório pode mudar.

**Recursos observados no desktop:** mapeamento visual das bounding boxes e edição de tipos/validações/subelementos; seleção de campos; geração por quantidade e seed; opções de escrita (modelos e estilos, variantes de caneta e diferentes estilos para texto/números); geração de PDF + ground-truth JSON + manifesto; visualização de PDFs; galeria de escrita para escolher presets; realismo visual antes/depois (iluminação, cor, saturação, ruído, imperfeições de digitalização etc.); execução OCR e módulos auxiliares. O usuário considera os últimos predominantemente ferramentas de debug ou secundárias.

**Recursos já separados da GUI:** `generate_dataset(...)` em `src/synthetic_dataset.py` recebe template, mapa, campos, quantidade, seed, variações de escrita, filtros e `progress_callback`; `run_pipeline(...)` em `src/pipeline.py`; `discover_document_types(...)` em `src/document_types.py`; operações e presets em `src/writing_presets.py`; CLI funcional. Outros trechos, sobretudo `pipeline_gui.py` e componentes de editores, podem conter regras embutidas na interface que precisem ser extraídas para services.

**Não assumir:** API atual existente, autenticação pronta, infraestrutura de produção, autorização para exposição pública, marca LabREC oficial já incluída no protótipo, backend funcional dentro do HTML, suporte automático a qualquer documento futuro, ou compartilhamento efetivo. Todos esses pontos precisam ser implementados e/ou confirmados.

## 3. Navegação proposta

Núcleo principal: **Modelos**, **Geração**, **Resultados**. Navegação auxiliar: **Presets de escrita** e **Presets de realismo** como modais/laboratórios acessíveis dentro da geração (podem ganhar páginas próprias se a complexidade exigir). Opcional no futuro: painel de administração, configurações, ajuda e API. Evitar o padrão atual de oito abas concorrentes no topo.

### Tela A — Modelos: catálogo

- Cabeçalho claro "Modelos de documentos", explicação curta e botão primário "Novo modelo".
- Cards/tabela com preview de miniatura, nome/tipo (DNV, certidão de nascimento etc.), versão, número de campos, autor, atualização e estado (rascunho/publicado).
- Ações por modelo: abrir/editar, duplicar, criar versão, exportar template e mapeamento, compartilhar conforme permissões. Novo modelo: upload PDF + metadados + iniciar mapeamento.
- Separar o conceito **modelo** (PDF + mapa + validações + versão) do conceito **dataset** (resultado de uma execução imutável/reproduzível).

### Tela A2 — Editor visual de modelo

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Modelos / DNV / Versão 3              Desfazer · Refazer · Salvar · Publicar         │
├───────────────────────────────────────────────────────────────────────────────────────┤
│ Barra de ferramentas: selecionar · desenhar campo · duplicar · excluir · zoom        │
├──────────────────────┬─────────────────────────────────┬────────────────────────────┤
│ CAMPOS               │ DOCUMENTO PDF                   │ PROPRIEDADES                │
│ Busca e filtros      │                                 │ ID, nome/label, descrição  │
│ Lista por seção      │ Canvas com bounding boxes       │ tipo (texto/data/etc.)      │
│ nomes, IDs, tipos    │ Pan, zoom e edição direta       │ limites e validações        │
│ selecionar/expandir  │ Destaque do campo selecionado   │ checkbox/subelementos       │
│                      │                                 │ regras e controles          │
├──────────────────────┴─────────────────────────────────┴────────────────────────────┤
│ Informações e validação do mapeamento; exportação JSON                               │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Interação principal: clicar na lista destaca a caixa; clicar na caixa seleciona o item da lista e abre propriedades; arrastar/mover/redimensionar precisa respeitar coordenadas, escala de página e DPI; suporte a campos compostos, checkbox e índice de dígito; undo/redo; validação consistente; evitar perda de edições. Mostrar tipos como rótulos legíveis, sem comprometer os identificadores do backend. Lista pode recolher; documento deve continuar área central dominante.

### Tela B — Geração (principal)

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ Nova geração                                                Histórico / Ajuda        │
├──────────────────────────────────────────────┬────────────────────────────────────────┤
│ CONFIGURAÇÕES                                │ PREVIEW DO DOCUMENTO                   │
│ 1 Documento / versão de mapa                 │ Canvas escuro, PDF claro               │
│ 2 Campos selecionados + busca + seleção      │ Zoom / destaque de caixas             │
│ 3 Quantidade, seed, nome do job              │ Mini resumo: campos, quantidade, seed  │
│ 4 Escrita: preset + "Testar/Comparar"         │                                        │
│ 5 Realismo: leve/médio/forte/customizado     │                                        │
│    + "Ajustar e comparar antes/depois"       │                                        │
│ 6 Revisão (avançado recolhível)              │                                        │
│                                            ────────────────────────────────────────────│
│ Resumo e botão primário "Gerar dataset"      │                                        │
└──────────────────────────────────────────────┴────────────────────────────────────────┘
```

Não jogar todos os parâmetros em um formulário enorme. **Modo simples**: modelo, quantidade, preset de escrita, realismo e gerar. **Modo avançado**: campos individuais, seed, divisão de estilos números/textos, writer/model, bias, cor/espessura/pressão/falhas, seleção dos efeitos e outras configurações que realmente existem na API. Mostrar defaults seguros e feedback de validação. Preview à direita persistente em desktop e empilhado em mobile. O fluxo deve aceitar 50 documentos com poucos cliques sem esconder a potência do modo técnico.

**Galeria de escrita**: um laboratório integrado ao bloco "Escrita" que permite gerar amostras com os modelos atualmente suportados, comparar diferentes variações para textos, datas, CEP, números e nomes, ajustar regularidade e caneta (cor, pressão, espessura, falhas), salvar/duplicar/reusar/compartilhar presets e selecioná-los diretamente para uma geração. Amostras precisam vir do motor real, não de fontes tipográficas arbitrárias, quando a implementação estiver integrada.

**Realismo visual**: conjunto de transformações opcionais aplicado após a composição sintética (se a pipeline atual permitir), parametrizado como presets leve/médio/forte/customizado. Na tela de geração aparece configuração resumida. "Ajustar" abre comparador antes/depois e controles técnicos. Preservar também operação isolada de aplicar filtros a PDFs, como ferramenta secundária para experimentos. Exibir efeito da seed/reprodutibilidade e evitar alterar o original inadvertidamente.

**Estado de job**: ao enviar, retornar identificador e exibir fila / em execução / concluído / falhou / cancelado; barra de progresso adequada ao backend; jobs pesados em workers; permitir navegar para Resultados; não prometer execução instantânea.

### Tela C — Resultados e compartilhamento

- Lista/tabela de gerações com nome do job, tipo de documento, data, autor, número de documentos, status, seed, modelo/mapeamento (versões) e presets usados.
- Ao abrir: visualizador de PDFs com lista/miniaturas e navegação; abas "Documentos", "Metadados/ground truth", "Presets/configuração", "Compartilhamento".
- Ações: baixar ZIP completo, PDF individual, JSON associado, copiar configuração, gerar novamente, excluir (com confirmação/permissões), compartilhar com pessoas/equipe ou obter link de acesso restrito.
- **Proveniência**: dataset.json, seed, template/mapeamento versionados, parâmetros, versões da aplicação/modelos; distinguir o que pode ser reproduzido bit-a-bit do que depende do backend/aleatoriedade.
- Compartilhamento implica identidade, permissão, escopo, expiração, revogação e auditabilidade. Não criar links públicos irrestritos por padrão; se houver modo público, exigir avaliação institucional e garantia de dados totalmente sintéticos. Salvar datasets e presets compartilháveis requer persistência adequada (não apenas pasta local sobre o diretório do repositório).
- UI deve suportar estados vazios, falhas, processamento pendente, downloads indisponíveis, arquivos expirados e sem permissão.

## 4. Fluxos típicos

**Usuário básico:** Geração → selecionar DNV publicada → quantidade=50 → preset de escrita → preset de realismo → gerar → acompanhar job → visualizar/baixar/compartilhar.

**Usuário avançado:** Modelos → cadastrar template PDF → mapear caixas e tipos → salvar/publicar versão → laboratório de escrita (testar, ajustar, salvar preset) → ajustar realismo → gerar com campos selecionados, seed e opções específicas → inspecionar JSON/PDF → compartilhar com a equipe e reproduzir execução.

## 5. Identidade visual e regras de UX

**Direção:** moderna, institucional e técnica, sem visual de site governamental antigo nem dashboard genérico cheio de gráficos. Interface principal clara; lateral escura; canvas de documento escuro para destacar PDF branco. Sidebar com "LabREC — Laboratório do Registro Civil" e marca do produto "OCRSD" na barra superior. Não utilizar imagem inventada como logo institucional oficial.

**Marca:** A logomarca oficial do LabREC **não foi obtida/verificada**. O protótipo usa somente *wordmark* tipográfico provisório. Solicitar ao responsável o SVG/PNG oficial e seguir as regras de marca antes de produção. Informar associação ao Registro Civil sem afirmar endosso/identidade institucional que não estejam confirmados.

**Paleta de design proposta, inspirada nas cores associadas ao Registro Civil brasileiro — NÃO tokens oficiais de manual de marca:**

| Token CSS | Hex | Uso |
|---|---|---|
| `--green` | `#087F50` | botão primário, seleção, estado ativo |
| `--green-dark` | `#065D3F` | texto/acento verde escuro |
| `--green-soft` | `#E7F5ED` | seleção sutil, avisos positivos |
| `--blue` | `#16446F` | links, informações, ações secundárias |
| `--navy` | `#102C46` | sidebar, títulos, identidade |
| `--blue-soft` | `#EBF2F9` | cartões informativos |
| `--yellow` | `#F6C84E` | detalhes de marca e destaques muito pontuais |
| `--ink` | `#1D2B38` | texto principal |
| `--muted` | `#627180` | ajuda e metadados |
| `--line` | `#DCE5E9` | bordas |
| `--bg` | `#F4F7F7` | fundo de página |
| `--white` | `#FFFFFF` | superfícies |

**Tipografia:** Inter (fallback system UI), IBM Plex Sans aceitável como alternativa. Preferir inputs com labels legíveis, espaçamento consistente, painéis de contexto, headers hierárquicos, tooltips breves e interações de alta densidade apenas quando necessárias. Responsive: priorizar desktop para editor de bounding boxes; apresentação/geração/resultados adaptáveis a telas menores. Acessibilidade: contraste, teclado, foco visível, textos alternativos, affordances reais e sem depender apenas de cor.

**Componentes:** sidebar, breadcrumbs, cards, accordions, stepper opcional, searchable field list, split panels redimensionáveis, toolbars, inspector, PDF canvas, before/after slider, preset cards, modais/drawers, job table, upload area, progress & status, share dialog. Evitar badges decorativas e métricas fictícias na implementação final.

## 6. Stack/arquitetura proposta (sujeita à inspeção do código)

- **Frontend**: React + TypeScript + Vite; thin client, sem algoritmos de documentos no browser.
- **Backend**: FastAPI + Pydantic; endpoints REST versionados com OpenAPI; reaproveitar `src/synthetic_dataset.py`, `src/pipeline.py` e demais serviços.
- **Serviços**: separar domain/application services de adapters HTTP/Tkinter/CLI. Não refatorar por estética; preservar CLI, testes e comportamento.
- **Jobs**: worker Python independente para tarefas longas; avaliar fila + persistência conforme carga real. Redis/RQ ou Celery são opções, não dependências obrigatórias no MVP; não deixar trabalhos longos presos a requests HTTP.
- **Armazenamento**: persistência de modelos, versões, presets, jobs, permissões, artefatos e expiração. SQLite pode servir no MVP limitado; escolher banco/storage adequado a concorrência e infraestrutura institucional quando necessário. Não salvar artefatos em caminhos arbitrários enviados por clientes.
- **Deploy**: Linux institucional, reverse proxy Nginx + TLS HTTPS, frontend servido em `/` e API em `/api/` no mesmo domínio; FastAPI vinculado localmente; Docker Compose se aprovado, alternativa com systemd sem Docker, workers e logs. Acesso público depende de autorização, rede, DNS, firewall e política institucional.

**Fluxo conceitual:** Navegador → HTTPS/Nginx → frontend estático + FastAPI → serviço/worker OCRSD → storage de jobs e datasets → endpoints de resultado. Nunca expor diretamente serviço interno do Registro Civil à internet.

**Exemplos de recursos de API para o Codex projetar/validar, não endpoints já implementados:** `GET /api/v1/document-types`, `GET /api/v1/document-types/{id}`, `GET /api/v1/models/{id}/fields`, `GET /api/v1/writing-presets`, `POST /api/v1/dataset-jobs`, `GET /api/v1/dataset-jobs/{id}`, `GET /api/v1/dataset-jobs/{id}/files`, `GET /api/v1/dataset-jobs/{id}/download`. CRUD de modelos, presets, permissões e compartilhamento conforme escopo das fases seguintes.

## 7. Segurança e responsabilidade institucional

- Renderizar e compartilhar **exclusivamente dados sintéticos** no fluxo público; sinalização inequívoca de "SINTÉTICO / SEM VALIDADE LEGAL" nas saídas, sem apresentar documentos como expedidos oficialmente.
- Processamento de documentos reais, se existir, deve ser segregado e ficar fora da superfície pública por padrão, sujeito a LGPD e controles institucionais.
- Limitar quantidade/tempo/memória/concor­rência, rate limiting, autenticação/autorização quando necessário, controles de upload, saneamento de nomes, paths restritos, isolation per job e download autorizado.
- Revisar o uso de templates/logomarcas/símbolos oficiais e a política de publicação de presets/modelos/documentos antes da abertura ao público.
- Seed sozinha não garante reprodutibilidade para modelos neurais e ambientes distintos: registrar versões e condições de execução.

## 8. Postman e implantação — entregas que o Codex deve produzir

- **Postman Collection v2.1 JSON importável** cobrindo SOMENTE endpoints realmente implementados, com pastas e exemplos; variables `baseUrl`, headers, auth quando aplicável, payloads válidos, testes representativos e fluxos geração→status→download. Exemplo de ambiente local e de produção sem credenciais, sincronizar com OpenAPI.
- Arquivos reais de deploy (por exemplo, Compose, Nginx, `.env.example`, configurações de serviço, health check), não apenas tutorial abstrato.
- Comandos exatos para instalação limpa, execução local, build frontend, startup API/worker, configuração TLS e DNS, atualização/reinício, logs, verificação, troubleshooting e permissões administrativas necessárias. Alternativa sem Docker se instituição proibir contêineres.
- Ambiente de servidor e permissões **não foram fornecidos**: separar passos executáveis dos pendentes de autorização. Nada de domínio, IP ou credenciais inventados.

## 9. Plano incremental recomendado

**Etapa 1**: auditar interfaces desktop, CLI e serviços existentes; extrair regras do Tkinter só quando necessário, preservando os testes.  
**Etapa 2**: vertical slice de leitura do catálogo → envio de geração → fila/progresso → armazenamento → resultado ZIP via API.  
**Etapa 3**: frontend web completo do fluxo de geração; estados reais e preview; coleção Postman e testes.  
**Etapa 4**: editor de modelos e campo visual; versionamento, validação e API para mapas.  
**Etapa 5**: laboratório de escrita, presets compartilháveis e realismo visual antes/depois.  
**Etapa 6**: biblioteca de resultados, colaboração, controles de acesso, auditabilidade e deploy institucional.  

Prioridade do MVP: uma geração real completa com PDF, JSON, manifesto e download, não simulação de frontend.

## 10. Restrições de implementação expressas pelo usuário

- **Não adicionar comentários ao código-fonte.** Preferir funções, nomes e tipos autoexplicativos; preservar comentários preexistentes quando fizer sentido. Evitar docstrings de preenchimento.
- **Não criar muitos `.md`.** No repositório de destino, atualizar README existente e criar no máximo um documento consolidado adicional se indispensável; não gerar um Markdown para cada fase, resumo, decisão, API e deploy. Entregar resumos no chat do Codex.
- Evitar muitos arquivos e abstrações sem benefício; manter monorepo atual se conveniente.
- Não perder CLI, interface desktop, formatos de mapas, geração e testes.
- Configurações e documentação em português do Brasil; prompt de implementação em inglês é aceitável.
- Entregar resultado funcional e instruções objetivas, Postman validado e arquivos reais de deploy quando a implementação evoluir.

## 11. Artefatos do pacote

- `prototipo/ocrsd_wireframe_labrec.html`: protótipo **offline e autossuficiente**, imagens inline, dados ilustrativos e interações simuladas; abrir no navegador. **Não está conectado à API**.
- `prototipo/build_mockup.py`: código opcional para reconstruir o HTML a partir dos screenshots incluídos (requer Pillow), não é código da aplicação final.
- `referencias/telas_originais/`: screenshots do desktop enviados pelo usuário, contexto fiel das funções existentes.
- `referencias/previews_web/`: imagens estáticas das páginas propostas (modelos, editor, geração, resultados).
- `PROMPT_CODEX.txt`: prompt para passar ao Codex junto deste briefing e das referências.

**Critério final:** não converter o Tkinter literalmente para páginas web. Construir uma plataforma coerente com 3 áreas principais, núcleo reutilizado, frontend simples, backend bem delimitado, segurança e fluxo de geração/compartilhamento reais.
