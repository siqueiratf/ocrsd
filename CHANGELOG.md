# Changelog

## [Não publicado]

### Interface web

- Interface migrada de React/Vite para páginas FastAPI com Jinja2 e HTMX,
  renderizadas no servidor, sem etapa de build nem dependências Node.
- Cabeçalho com o logotipo oficial do LabREC em SVG, extraído de `labrec.pdf`, com
  link para a Galeria de Modelos, e o nome OCRSD em destaque com o significado
  “Only Civil Registry Synthetic Dataset”.
- Rodapé na cor da faixa inferior de `labrec.pdf`, com o logotipo do Registro Civil
  em SVG extraído do mesmo arquivo.
- Navegação horizontal, visual LabREC revisado, layout responsivo e componentes
  consistentes para modelos, editor, geração, resultados, presets e compartilhamento.
- Espaço privado mantido em cookie HttpOnly, com migração automática do token salvo
  pelas versões anteriores e dos links de compartilhamento antigos.
- Nginx do Compose passa a encaminhar toda a interface para a API; testes da interface
  web passam a usar `unittest`.
- Editor de modelos com paleta de elementos pesquisável, aba Contextos, canvas com zoom,
  ajuste, pan, tela cheia, alças de redimensionamento, subelementos, desfazer/refazer,
  painéis redimensionáveis e inspetor por seções.
- Novo modelo por PDF importado ou documento em branco com textos, linhas e caixas que
  geram o template PDF; pré-visualização por geração real, publicação explícita e
  proteção contra salvamentos concorrentes.
- Tipos Município, Localização, Profissão, Documento (CPF, RG, CNS) e Assinatura, máscaras,
  formatos de data, intervalos decimais e campos opcionais no motor de geração.
- Tipos de campo personalizados reutilizáveis por espaço de trabalho.
- Presets de Escrita com galeria própria, filtros, busca, paginação e prévias geradas
  pelo motor real com cache; editor em página dedicada com seleção de modelo, escritor,
  aparência, comportamento e playground com até três variações.
- Presets padrão por fonte manuscrita, duplicação, versão, “Salvar como novo”,
  compartilhamento por link revogável e uso direto na geração com retorno ao rascunho.
- Cada job registra a referência e a versão do preset, além da configuração efetiva.
- Resultados com colunas centralizadas, indicador de status em posição fixa, coluna
  Tempo (processamento sem a fila, com ordenação numérica), renomeação inline estável
  e tooltip de status que não gera rolagem.
- Visualizador de resultados com controles compartilhados com o editor de mapeamento
  (zoom, ajuste à página e à largura, centralizar, pan e tela cheia), ajuste estável
  sem redução contínua de zoom e lista de documentos atualizada em tempo real durante
  a geração, sem reiniciar o PDF aberto.
- Notificações unificadas com contador de 15 segundos, botão de fechar, sincronização
  entre abas e aviso de conclusão independente do aviso de progresso; erros permanecem
  até serem fechados.
- Etapa Campos destaca e localiza no PDF o campo marcado, inclusive fora da área
  visível; etapa Escrita desativa estilos incompatíveis com os campos escolhidos.

### Motor de geração

- Data com formatos DD/MM/AAAA, DD-MM-AAAA, por extenso e em áreas separadas de dia,
  mês e ano, sempre a partir de uma única data válida; tipo Sexo (por extenso ou M/F);
  tipo Número da DNV sintético no formato XX-XXXXXXXX-X do manual do SINASC, com
  prefixo de teste 00 e marcação no ground truth de que o último algarismo não é um
  dígito verificador oficial, porque o manual não publica o algoritmo.
- Campos do mesmo tipo com o mesmo vínculo semântico recebem o mesmo valor no documento.
- Cada geração recebe uma semente única automática (tempo em nanossegundos, contador,
  bytes aleatórios e verificação no banco), registrada no job, no manifesto e no ground
  truth; cada job guarda uma cópia do mapa e do template para reprodução exata.
- Cada modelo tem um único mapa ativo (`documents/<tipo>/modelo.json` nos modelos padrão),
  carregado automaticamente na geração; o assistente não oferece mais a escolha de mapas.
- Modelos padrão ficam protegidos contra edição direta; “Duplicar e editar” cria uma
  cópia independente.

- Sequências de dígitos ligadas a localidades (por exemplo, código IBGE do município)
  recebiam o nome da cidade; agora recebem apenas dígitos, o que corrige a falha dos
  modelos neurais com todos os campos da DNV.
- Validação centralizada recusa antes do envio as combinações de escrita neural com
  caracteres que o modelo não escreve (por exemplo, política estrita com acentos ou
  `/`), com base em amostras de todos os valores possíveis de cada campo.
- O filtro de sombra do realismo visual usava um desfoque de raio muito grande na
  resolução total e esgotava o limite de CPU do worker após cerca de 11 documentos;
  o desfoque agora usa resolução reduzida com resultado equivalente.

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
