# OCRSD

Ferramentas para gerar documentos civis sintéticos, localizar conteúdo preenchido,
comparar motores OCR e ocultar campos em PDFs. O alinhamento e a detecção usam
visão computacional; o reconhecimento OCR fornece texto e confiança no manifesto.

## Plataforma web

A aplicação web organiza o trabalho em Modelos, Geração e Resultados. Ela permite
cadastrar templates PDF privados, editar mapas no canvas, manter presets de escrita
e configurações, gerar documentos no worker Python, acompanhar o progresso, revisar
PDF e ground truth, baixar o ZIP e criar links revogáveis. O motor Python, a CLI e
o desktop continuam disponíveis. Os documentos gerados pela web recebem marca
visível **SINTÉTICO — SEM VALIDADE LEGAL**.

Para desenvolver com Docker Engine e Compose instalados:

```bash
cp deploy/.env.development.example deploy/.env.development
docker compose --env-file deploy/.env.development up -d --build
docker compose --env-file deploy/.env.development ps
curl --fail http://127.0.0.1:8080/api/v1/health
```

Abra **http://127.0.0.1:8080**. O Compose de desenvolvimento usa o projeto
`ocrsd-dev` e o volume `ocrsd-dev_jobs`, separados da produção no servidor 2.
A publicação fica restrita ao computador local.
O Compose monta `assets/handwriting` somente para leitura e habilita os modelos
neurais quando os pesos completos estão presentes. Ajuste
`OCRSD_HANDWRITING_ASSETS` se os pesos forem mantidos fora do checkout.
`OCRSD_BIND_ADDRESS` controla a interface publicada; mantenha `127.0.0.1` quando
houver proxy no host e use um endereço privado específico para acesso por VPN.
Modelos, presets, configurações e histórico pertencem ao espaço privado criado no
primeiro acesso e persistem em SQLite. O navegador guarda a chave desse espaço em um
cookie HttpOnly; o token salvo por versões anteriores é migrado automaticamente.
Preserve esse cookie: esta versão não integra login institucional nem recuperação de acesso. Links de resultado
são opcionais, podem impedir downloads, expiram com o job e podem ser revogados. A
chave do compartilhamento fica no fragmento da URL e é removida após a leitura.
O OpenAPI está em `/api/openapi.json` e a documentação interativa em `/api/docs`.
`docker compose --env-file deploy/.env.development logs -f api worker web` mostra os logs; `docker compose --env-file deploy/.env.development stop`
interrompe os serviços preservando os dados. Não use `down -v` se precisar
preservar jobs e resultados.

Para desenvolver sem Docker, use Linux, Python 3.11 e uv. A interface usa FastAPI,
Jinja2 e HTMX, sem etapa de build. O comando abaixo prepara as dependências quando
necessário e inicia a API, que também entrega a interface, e o worker; pressione
Ctrl+C para encerrar os dois processos:

```bash
make web
```

No modo local, esse alvo habilita automaticamente todos os modelos neurais cujos
pesos estejam presentes em `assets/handwriting`. Abra **http://127.0.0.1:8000**.
Se a porta 8000 estiver ocupada, use `make web PORT=8001`. `make web` também
habilita `OCRSD_STANDARD_EDITS`, que adiciona ao editor as ações “Salvar como
modelo padrão” e “Salvar nova versão do padrão”; ambas gravam em
`documents/<chave>/` do repositório, então confira o diff e rode
`git add documents/<chave>` para versionar. Mantenha a variável desativada
(`OCRSD_STANDARD_EDITS=false`, o padrão em Compose e systemd) fora do
desenvolvimento local.
Para iniciar cada processo separadamente:

```bash
uv sync --locked --extra web --extra test
```

Execute em dois terminais, ambos na raiz do repositório:

```bash
uv run --no-sync uvicorn src.web.api:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
uv run --no-sync python -m src.web.worker
```

A interface fica em `http://127.0.0.1:8000` e a API em `/api/v1`. Os jobs ficam
em `var/web/`; feche o worker com Ctrl+C, aguardando o job atual terminar.
API e worker devem usar as mesmas variáveis `OCRSD_*`; a execução direta não lê
`deploy/.env.development` automaticamente. Para carregá-las no shell, use
`set -a; . deploy/.env.development; set +a` antes de iniciar cada processo.

Importe [a coleção Postman](postman/ocrsd.postman_collection.json) e
[o ambiente local](postman/local.postman_environment.json). Ajuste `baseUrl`
para `http://127.0.0.1:8080` ao usar Compose. Execute a coleção no Collection
Runner: ela seleciona recursos, gera um dataset, aguarda a conclusão, baixa
artefatos e testa o cancelamento de outro job. Requisições individuais podem ser
executadas na mesma ordem; repita “Aguardar conclusão” até `succeeded`. Para salvar
o ZIP em disco no Postman, use **Send and Download**. A chave retornada na criação
é privada, não aparece novamente e não deve ser incluída em exports compartilhados.

```bash
uv run --no-sync python -m unittest discover -s tests -q
uv run --no-sync python -m scripts.smoke_generate --output output/smoke --seed 123
uv run --no-sync python -m scripts.export_postman --check
```

Após alterar contratos, regenere a coleção com
`uv run --no-sync python -m scripts.export_postman` e execute os testes.
[Desenvolvimento e operação](docs/development.md) concentra a análise da arquitetura,
os limites, os comandos para instalar em Linux, HTTPS, systemd, atualizações e
recuperação. O deploy institucional exige revisão de rede, domínio, recursos e
permissões pela equipe responsável; os arquivos não publicam serviços internos.
A produção usa `deploy/compose.ssh2.yaml`, `deploy/.env.production`, o volume
`ocrsd_jobs` e a URL `https://ocrsd.onrcpn.org.br/`.

O branch `develop` concentra as próximas mudanças e roda a validação no GitHub.
O branch `main` registra versões aprovadas para produção; o fluxo `deploy.yml`
atualiza o servidor 2 de forma automática após cada commit aprovado em `main`.
Cada publicação também recebe uma tag `V*` para a release, mas a tag não é mais
um passo manual de deploy. Os arquivos `.env.development` e `.env.production`
são locais a cada ambiente e não entram no Git.

### Publicação em produção

O push para `main` executa o fluxo `validation.yml` no GitHub Actions: testes,
smoke e conferência do Postman. Quando esse fluxo termina com sucesso em `main`,
o fluxo `deploy.yml` inicia e atualiza o servidor 2 de forma automática, sem
passo manual. O push de uma tag `V*` continua a repetir as verificações e a
criar a release com artefatos, mas a release em si não aciona o deploy; o
deploy segue cada commit aprovado em `main`.

O fluxo `deploy.yml` roda em um executor próprio (self-hosted runner) instalado
no servidor 2, porque o servidor fica em uma rede privada. Esse executor:

1. Faz checkout do commit que passou nos testes.
2. Roda `docker compose -f deploy/compose.ssh2.yaml --env-file deploy/.env.production up -d --build --remove-orphans`.
3. Verifica `https://ocrsd.onrcpn.org.br/api/v1/health`.

O executor roda sob a conta de sistema `ocrsd`, membro do grupo `docker`. Essa
conta não usa `sudo`; o grupo `docker` já dá acesso ao comando `docker compose`.
Para preparar o executor no servidor 2, siga as instruções de "Runners de
repositório" (ou de organização) em Settings → Actions → Runners no GitHub, e
dê ao executor o rótulo `ocrsd-server2`.

Para disparar um deploy manual sem um novo push, use `workflow_dispatch` no
fluxo `validation.yml`, ou publique um commit vazio em `main`.

Mantenha `deploy/.env.production` e os pesos locais no servidor. O volume
`ocrsd_jobs` preserva banco, histórico e resultados entre atualizações.

## Instalação

Requer Python 3.11. As dependências e suas versões estão em `pyproject.toml` e `uv.lock`.

```bash
git clone git@github.com:siqueiratf/ocrsd.git
cd ocrsd
uv sync --locked
uv sync --locked --extra benchmark
uv sync --locked --extra paddle
uv sync --locked --extra handwriting
uv run ocrsd doctor
uv run ocrsd --help
```

PyTorch usa o índice de CPU configurado no projeto. Os extras podem ser instalados
com `uv sync --locked --extra benchmark`, `--extra paddle` ou `--extra handwriting`.
`benchmark` inclui TrOCR e a integração com Tesseract; o executável Tesseract e os
idiomas `por` e `eng` precisam ser instalados pelo gerenciador do sistema.
`handwriting` inclui TensorFlow para converter o checkpoint do modelo sjvasquez.
Os modelos OCR podem baixar pesos no primeiro uso.

## Geração e validação

Os templates e os mapas usados pela aplicação ficam em `documents/`. O mapa
oficial da DNV tem 69 campos e o da certidão de nascimento tem 19.

```bash
make dataset COUNT=1 SEED=42
make check
```

`make dataset` cria PDFs, JSONs com os valores esperados e `dataset.json` em
`documents/<tipo>/dataset/aaaammdd_hhmm-<tipo>-<quantidade>/`. Esses resultados
são locais e ignorados pelo Git. Use `TEMPLATE=... MAPPING=...` para outro
modelo ou `FIELDS="ID1 ID2"` para selecionar campos.

`make check` executa os testes e gera uma DNV e uma certidão novas. A validação
confere a renderização de cada campo, o alinhamento, a detecção, os recortes e o
PDF com os campos ocultados. Seus resultados ficam em `output/smoke/`.

```bash
uv run python -m scripts.smoke_generate --output output/validacao --seed 123
```

O GitHub Actions executa essa validação em pushes, pull requests e semanalmente,
às segundas-feiras às 06:00 UTC. Os documentos sintéticos e o relatório da
validação ficam disponíveis como artefatos da execução.

## Processamento de documentos

```bash
uv run ocrsd run \
  --document output/smoke/dnv/dataset/synthetic_0001.pdf \
  --template documents/dnv/template.pdf \
  --field-map documents/dnv/mappings/mapeamento_seguindo_norma.json \
  --output output/processamento \
  --skip-ocr
```

Remova `--skip-ocr` para reconhecer texto. `--ocr-engine` seleciona o backend;
`uv run ocrsd models` lista as opções. `--dpi` deve coincidir com o mapa.
`--stop-after raster`, `align` ou `diff` permite inspecionar uma etapa.

A saída contém o PDF reconstruído com os campos ocultados, a imagem em
`masked/`, o manifesto JSON, os recortes originais em `crops/` e imagens de
diagnóstico em `debug/`. O PDF reconstruído contém somente a imagem processada.
Os recortes e o documento original conservam o conteúdo preenchido.

## Interface desktop

```bash
make gui
```

A interface Tkinter reúne editor de imagem, mapeador de campos, execução de
documentos, comparação OCR, geração de datasets, visualização de PDFs,
galeria de escrita e filtros de digitalização. Tkinter requer uma sessão gráfica
e as bibliotecas Tcl/Tk do sistema.

O mapeador permite criar e editar caixas, campos compostos, tipos de dados e
validações. A interface descobre templates e mapas no catálogo `documents/`.
Os presets de pré-processamento ficam em `documents/<tipo>/presets/`, e a
biblioteca de estilos de escrita fica em `documents/writing_presets.json`.

## Escrita e filtros

O renderizador padrão `font` funciona sem baixar modelos neurais. Também estão
integrados `sjvasquez`, `aaron1259` e `wpvnteam`.

```bash
make download-sjvasquez
make dataset-sjvasquez COUNT=1 WRITER=9
make download-aaron1259
make dataset-aaron1259 COUNT=1
```

Os ajustes disponíveis incluem `BIAS`, `PEN_COLOR`, `PEN_WIDTH`, `PEN_PRESSURE`,
`PEN_DROPOUT`, `TEXT_POLICY`, `INTENSITY` e `THREADS`. Por padrão, o Makefile usa
`TEXT_POLICY=normalize`; os alfabetos neurais têm restrições para acentos e
símbolos. As origens e limitações estão em [assets/handwriting/README.md](assets/handwriting/README.md).

Na CLI, `ocrsd generate --printed` usa texto impresso. `--realistic-filters`
ativa efeitos de iluminação, sombra, ruído e compressão; `--filter-preset`
aceita `light`, `medium` e `strong`.

## Comparação e pré-processamento OCR

```bash
uv run --extra benchmark ocrsd benchmark \
  --input output/smoke/dnv/pipeline/crops \
  --models easyocr,tesseract \
  --output output/comparacao.json
uv run ocrsd preprocess \
  --input output/smoke/dnv/pipeline/crops \
  --output output/recortes-processados \
  --preset documents/dnv/presets/preset_001.json
```

Os backends implementados são EasyOCR, TrOCR manuscrito, TrOCR impresso,
TrOCR manuscrito grande, Tesseract e PaddleOCR. O pré-processamento aceita
`traditional` e `vlm`. A pasta de saída deve ficar fora da entrada; use
`--overwrite` para substituir arquivos já existentes.

## Organização

| Pasta ou arquivo | Conteúdo |
| --- | --- |
| `src/` | Aplicação, CLI, interfaces e configuração em `src/config.py` |
| `tests/` | Testes automatizados |
| `scripts/` | Validação sintética e publicação de releases |
| `documents/` | Templates, mapas e presets |
| `assets/` | Fonte e informações dos modelos de escrita |
| `docs/` | Documentação de desenvolvimento e operação |
| `.github/workflows/` | Validação contínua, publicação e deploy |
| `CHANGELOG.md` | Alterações por versão |

As configurações da IDE, os datasets gerados, as saídas e os pesos locais dos
modelos são ignorados. A pasta local `examples/` permanece vazia.

## Limitações

O processamento usa somente a primeira página e exige um template do mesmo
layout. A detecção por campos é preferível para documentos digitalizados;
a detecção por palavras é mais suscetível a ruído. A qualidade do OCR de texto
manuscrito depende do modelo. A validação sintética usa o renderizador padrão
e dispensa o download de pesos OCR ou neurais.

## Release

O release [V1](https://github.com/siqueiratf/ocrsd/releases/tag/V1) corresponde à
versão Python `1.0.0`. Seu changelog está em [CHANGELOG.md](CHANGELOG.md).
O workflow de publicação valida os documentos e publica as notas, o arquivo do
código-fonte e documentos sintéticos de validação quando uma tag `V*` é enviada.
