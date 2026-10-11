# Desenvolvimento e operação

## Fronteiras da aplicação

A entrega web usa FastAPI/Pydantic, páginas Jinja2 renderizadas no servidor com HTMX,
SQLite e um worker Linux independente. Templates sintéticos podem ser enviados para espaços privados; não há
upload nem processamento público de documentos pessoais. A CLI e o desktop mantêm
suas entradas e algoritmos; os extras `web` e `test` são opcionais.

| Implementação examinada | Comportamento e fronteira de migração |
| --- | --- |
| `synthetic_dataset.py` | Motor independente de Tkinter: valores por tipo, campos compostos, canetas, filtros, PDFs, ground truth, manifesto e callback por documento. Reutilizado integralmente. |
| `synthetic_dataset_gui.py` | Seleção de arquivos/campos, conversão de entradas, validação, presets mistos, preferências JSON, criação da pasta e thread/eventos. A web traduz opções em seu serviço, sem importar a GUI ou gravar suas preferências. |
| `pipeline.py`, `config.py` | Pipeline e configuração independentes: rasterização, alinhamento, diferenças, segmentação, OCR, ocultação e manifestos. Caminhos são aceitos no uso local; uma API futura deverá resolver IDs antes de chamá-los. |
| `pipeline_gui.py` | Ainda contém orquestração adicional: escolha de modelos/tipos, pré-processamento por campo, OCR de sequências e agregação de previsões. Extrair esse caso de uso antes de migrar a tela; chamar só `run_pipeline` não reproduziria tudo. |
| `gui.py` | Edição de imagem, histórico/crop, presets, processamento de pastas e subprocessos de benchmark com cancelamento/timeout. Os controles são Tkinter; operações do motor e seus parâmetros podem ser compartilhados. |
| `field_mapper_gui.py` | O editor web implementa canvas, coordenadas originais, zoom, undo/redo e caixas; conserva `dpi`, `page_size_px`, `bbox`, `elements` e `validation`. Operações avançadas exclusivas do desktop continuam disponíveis nele. |
| `fields.py`, `document_types.py` | Leitura/associação de mapas e descoberta de templates independentes. A web reutiliza o catálogo, incluindo `mappings/*.json` e o formato antigo `*_fields.json`. PDFs conservam o layout original. |
| `writing_presets.py` | Biblioteca JSON independente; CRUD e descrições podem servir desktop/serviços. Na web há leitura e conversão de presets simples. Presets mistos são listados como indisponíveis. |
| `ocr.py`, `ocr_bench.py`, `preprocess.py` | Reconhecimento, backends e filtros independentes. EasyOCR, TrOCR, Tesseract e Paddle têm pesos/dependências próprios; não são carregados pela API de geração. |
| `neural_handwriting.py`, `sj_handwriting.py` | Canetas, alfabetos, inferência, verificação de pesos e conversão são reutilizáveis. A web não baixa pesos; modelos ficam desabilitados por padrão. Sjvasquez exige também checksum e arquivos dos 13 estilos. |
| `document_filters.py` e GUI | Filtros e PDFs têm funções próprias e semente; a GUI mantém seleção, prévia e exportação. A web oferece os três presets existentes. |
| Galeria e visualizador PDF | As prévias web chamam diretamente os renderizadores Python sem importar Tkinter. O visualizador usa somente artefatos autorizados e o leitor PDF do navegador. |

`src/web/models.py` define o contrato; `services.py` resolve recursos, valida opções
e chama o motor, sem depender de HTTP. `jobs.py` persiste estados e controla acesso,
cotas e retenção. `worker.py` executa os processos; `api.py` adapta HTTP para esses
serviços. `ui.py` renderiza as páginas e os fragmentos HTMX com os mesmos serviços, sem
chamar a API por HTTP; `templates/` e `static/` contêm o HTML, o CSS, o HTMX local e o
JavaScript mínimo do editor de campos. `settings.py` centraliza limites e armazenamento. Essas fronteiras são
suficientes para o primeiro caso de uso, sem replicar o motor ou criar repositórios
abstratos. O serviço de geração também pode ser chamado diretamente em Python.

## Módulos web

A interface está organizada em três áreas, mantendo o processamento no backend Python:

| Área | Escopo |
| --- | --- |
| Modelos | Cadastro de templates PDF, cópia editável do catálogo, metadados e canvas com caixas, zoom, movimentação, redimensionamento, busca, undo/redo, propriedades, validações e exportação do mapa no schema existente. |
| Geração | Seleção de modelo, mapa e campos, quantidade, semente, escrita impressa/manuscrita, estilos distintos, presets privados, configurações reutilizáveis e realismo com prévia executada pelo backend. |
| Resultados | Histórico privado persistente, busca/filtro, progresso, cancelamento, PDFs, ground truth, manifesto, downloads, repetição e compartilhamentos revogáveis com permissão de download. |

Presets criados pela web são privados e armazenados por proprietário, sem um arquivo
JSON global mutável. A configuração do dataset copia seus parâmetros para o job. O
catálogo administrado continua somente leitura e só expõe pesos e presets aprovados.

O navegador mantém um token opaco do espaço em `localStorage`; seu hash fica no banco.
Esse token delimita modelos, presets, configurações, jobs e compartilhamentos, mas não
substitui autenticação institucional. Um link compartilhado leva ID e chave no
fragmento, que não chega ao proxy e é removido da barra após a leitura. Quem possuir o
link acessa o resultado até revogação ou expiração. O catálogo é administrado por
arquivos locais, nunca por caminhos recebidos da API.

## Jobs e limites

A API grava opções e hash da chave em SQLite/WAL e retorna `202`; não executa geração.
Um único worker por armazenamento, protegido por `flock`, consome a fila com transações
atômicas. Cada job roda em um subprocesso `spawn`, carrega modelos uma vez para o lote
e libera a memória ao terminar. Há progresso por documento, seguido de marcação e ZIP.
O cancelamento encerra o subprocesso; saídas incompletas são removidas.

Estados: `queued`, `running`, `succeeded`, `failed`, `cancelled`, `expired`.
Ao reiniciar, jobs em execução são marcados como falhos, sem repetição silenciosa;
jobs na fila são preservados. A mesma semente permite reenviar. SIGTERM/Ctrl+C no
worker termina o job atual antes de sair. O filho morre se o supervisor morrer.
A saúde retorna HTTP 200 com `status=degraded` quando não há heartbeat recente;
`worker_ready=true` indica que o supervisor está ativo, não garante sucesso do modelo.

SQLite e volume **local** atendem uma máquina e concorrência baixa. Não usar NFS nem
replicar o worker. Redis/Celery não justificam o custo operacional nessa escala.
Múltiplas máquinas exigirão fila com leases, banco compartilhado e armazenamento
compatível. Esta versão não tem retry automático, HA, agendamento por usuário nem
cache de modelos entre jobs. A implantação fornecida utiliza CPU.

| Variável | Padrão | Efeito |
| --- | --- | --- |
| `OCRSD_STORAGE` | `var/web` | Banco e saídas; `/data` no Compose, `/var/lib/ocrsd` no systemd |
| `OCRSD_DOCUMENTS` | catálogo do projeto | Raiz administrada de templates, mapas e presets |
| `OCRSD_MAX_COUNT` | 10 | Documentos por job; limite absoluto do contrato: 100 |
| `OCRSD_MAX_PENDING` | 20 | Jobs simultaneamente na fila/em execução |
| `OCRSD_JOBS_PER_HOUR` | 10 | Submissões por IP/hora, transação persistente |
| `OCRSD_RETENTION_HOURS` | 24 | Retenção após término; limpeza pelo worker |
| `OCRSD_JOB_TIMEOUT` | 900 | Tempo máximo do subprocesso em segundos |
| `OCRSD_MAX_JOB_BYTES` | 268435456 | 256 MiB por job, incluindo ZIP e temporários |
| `OCRSD_MAX_UPLOAD_BYTES` | 12582912 | Limite do template PDF enviado; 12 MiB |
| `OCRSD_MAX_STORAGE_BYTES` | 4294967296 | Reserva de saídas: 4 GiB; até 16 jobs não expirados no padrão |
| `OCRSD_MIN_FREE_BYTES` | 536870912 | Margem livre mínima; 512 MiB |
| `OCRSD_MEMORY_MB` | 4096 | Limite de espaço virtual do subprocesso, além do cgroup |
| `OCRSD_NEURAL_ENABLED` | false no código; true no exemplo de deploy | Requer habilitação e pesos completos instalados pelo administrador |
| `OCRSD_ACCEPTING_JOBS` | true | false suspende submissões; consulta/download continuam |
| `OCRSD_BIND_ADDRESS` | `127.0.0.1` no exemplo | Interface do host usada pela publicação web |
| `OCRSD_HTTP_PORT` | 8080 | Porta HTTP local ou privada da interface web |

O worker verifica tempo/disco a cada 0,5 s e limita CPU, memória virtual e tamanho
individual de arquivo. A medição de volume pode ultrapassar brevemente o teto entre
checagens. O banco/logs não fazem parte da cota de artefatos: monitore disco, use
rotação de logs e quotas do sistema quando necessárias. Metadados expirados são
removidos após mais 24 horas. API e worker devem compartilhar configuração e volume.
IP é uma cota básica: NAT institucional compartilha limite e clientes distribuídos
podem contorná-lo. Nginx também limita requisições e conexões; grandes exposições
públicas podem exigir proteção adicional aprovada na borda.

Os arquivos públicos são marcados visualmente e contêm `synthetic=true` e aviso nos
JSONs. A sobreposição visual pode afetar reconhecimento em suas regiões: os valores
do ground truth representam o texto gerado antes da marcação. A geometria original
é preservada; não use os PDFs públicos como baseline sem considerar a marca.
Determinismo significa mesmos valores e pixels para versões, assets e opções iguais;
PDF/ZIP podem ter metadados temporais diferentes. Atualizações de bibliotecas ou
pesos podem mudar resultados. Os testes reais usam fonte local, sem inferência neural.

## Contrato e verificação

O catálogo possui tipos/mapas, template PDF, campos na resposta do mapa, presets e
capacidades. `POST /api/v1/generation/validate` valida a configuração;
`POST /api/v1/jobs` retorna a chave; consultas, cancelamento, resultados e artefatos
exigem `Authorization: Bearer <chave>`. `download=false` entrega PDF inline.
Resultados incompletos retornam 409; expirados, 410; acessos com chave incorreta, 404.
Erros da aplicação seguem `{"error":{"code":"...","message":"..."}}`.
POST exige JSON e Content-Length. Corpos comuns têm limite de 32 KiB, mapas de 128 KiB
e cadastro de modelo permite o PDF base64 dentro do limite configurado. O servidor
confere também o corpo efetivamente recebido. Não há CORS aberto nem credenciais em
query strings; links compartilhados transportam o segredo no fragmento do navegador.

As rotas reais estão no OpenAPI e em `postman/ocrsd.postman_collection.json`.
O gerador percorre o OpenAPI e usa respostas reais do catálogo como exemplos.
Testes verificam cobertura de todas as operações, exemplos contra schemas e a
igualdade entre coleção versionada e gerada. O schema oficial Postman v2.1 pode
ser validado adicionalmente:

```bash
curl -fsSL https://schema.getpostman.com/json/collection/v2.1.0/collection.json -o /tmp/ocrsd-postman-schema.json
uv run --no-sync python -c 'import json,jsonschema; jsonschema.validate(json.load(open("postman/ocrsd.postman_collection.json")),json.load(open("/tmp/ocrsd-postman-schema.json")))'
npx --yes newman@6.2.1 run postman/ocrsd.postman_collection.json --env-var baseUrl=http://127.0.0.1:8080
```

Execute Newman com worker ativo e cota disponível: cada execução cria dois jobs.
Não exporte o ambiente após testes com chaves preenchidas. A documentação Swagger
carrega scripts do CDN jsDelivr; redes isoladas podem usar o OpenAPI baixado no
Postman, sem depender dessa interface. A interface web não depende de CDN.

A suíte `unittest` cobre motor/CLI/auxiliares desktop e serviços/API, incluindo
processos reais, concorrência, entradas inválidas, isolamento, falhas, cancelamento,
expiração e repetição determinística. `tests/test_web_ui.py` cobre páginas, fragmentos
HTMX, sessão por cookie, submissão, progresso, prévia, download autenticado,
compartilhamento, editor, presets, erros e recuperação de acesso. O smoke testa os 69 campos
da DNV e os 19 da certidão, alinhamento, detecção, recortes e ocultação, sem OCR.
A interação manual desktop ainda precisa de validação própria. A suíte automatizada
usa fonte local; os três modelos neurais foram exercitados separadamente no servidor
2. Os comandos de execução e testes estão no README.

## Instalação institucional com Compose

Use uma VM/servidor Linux aprovado para esta aplicação, sem ligação direta da API
pública a bases ou serviços internos. A equipe institucional precisa aprovar domínio,
DNS, rota externa ou VPN, portas, TLS, logs, armazenamento, instalação de Docker e
política de atualização. As instruções abaixo são para **Debian 12 amd64**, com sudo;
outras distribuições exigem ajuste dos pacotes. Nenhum domínio/IP real é presumido.

Os limites fornecidos somam aproximadamente 6 GiB para containers; reserve também
RAM para o sistema e espaço para imagens/dependências, volume de 4 GiB e backups.
São parâmetros iniciais, não especificações conhecidas do servidor. Confirme a
capacidade disponível e meça lotes representativos antes de liberar acesso público.

Instalação pelo administrador, se Docker for permitido:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl git nginx certbot
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
printf '%s\n' 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable' | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker nginx
sudo git clone https://github.com/siqueiratf/ocrsd.git /opt/ocrsd
cd /opt/ocrsd
sudo cp deploy/.env.example deploy/.env
sudo chmod 600 deploy/.env
sudo docker compose --env-file deploy/.env config --quiet
sudo docker compose --env-file deploy/.env up -d --build
sudo docker compose --env-file deploy/.env ps
curl --fail http://127.0.0.1:8080/api/v1/health
```

Selecione a revisão aprovada com `sudo git checkout REVISAO_APROVADA` antes do build.
O build instala dependências do lockfile e inclui o catálogo e a interface web.
Downloads exigem saída HTTPS para registries Docker, PyPI e índice PyTorch;
o funcionamento padrão não precisa de saída da rede privada dos containers.
A API e o worker usam UID 10001, filesystem somente leitura, volume persistente e
limites de CPU/memória/processos. O Nginx público do container atende apenas no
loopback do host, porta 8080. Para outra porta: `sudo env OCRSD_HTTP_PORT=18080 docker compose --env-file deploy/.env up -d`.

As sub-redes `172.30.85.0/24` (API/worker) e `172.30.86.0/24` (entrada web)
são **exemplos operacionais reservados à pilha**, não IPs institucionais. Antes do deploy, verifique conflitos com Docker/VPN/rede local.
Se precisar alterá-la, atualize `compose.yaml`, o IP confiável em `deploy/Dockerfile`
e o gateway confiável em `deploy/nginx.conf`, então reconstrua. Só `172.30.85.2` é
confiável para a API; só o gateway `172.30.86.1` é confiável para o Nginx do container.
O proxy de borda sobrescreve `X-Forwarded-For`; nunca confie em cabeçalhos enviados
diretamente pela internet. Não publique a porta 8000 nem o volume de jobs.

### Instância atual do servidor 2

A primeira implantação está em `/opt/ocrsd` no Ubuntu 22.04 do servidor
`192.168.3.102`. O Compose v2 foi instalado pelo pacote da distribuição e o projeto
mantém API, worker e Nginx de entrada em containers próprios, com o volume `ocrsd_jobs`.
O Nginx de entrada escuta somente em `192.168.3.102:8080`; a API não publica porta no host.
O container preexistente `chandra-ocr-vllm`, na porta 8010, não faz parte desta pilha.

A VPN atual encaminha SSH, mas a política de rede bloqueia TCP 8080 antes de chegar
ao host. O firewall local está inativo, portanto a validação externa deve usar um
túnel até que a infraestrutura libere uma rota aprovada ou publique HTTPS. Na
estação com a VPN conectada, mantenha este comando aberto:

```bash
ssh -N -L 18080:192.168.3.102:8080 ia@192.168.3.102
```

Também é possível manter uma sessão interativa com
`ssh2 -L 18080:192.168.3.102:8080`. Abra `http://127.0.0.1:18080`, importe
`postman/servidor.postman_environment.json` e execute a collection. Para operar os
serviços no servidor:

```bash
cd /opt/ocrsd
sudo docker compose --env-file deploy/.env ps
sudo docker compose --env-file deploy/.env logs --tail=100 api worker web
sudo docker compose --env-file deploy/.env restart api web
sudo docker compose --env-file deploy/.env stop
sudo docker compose --env-file deploy/.env start
```

Ainda não há FQDN institucional, registro DNS ou certificado TLS configurado. Para
publicação sem túnel, a infraestrutura deve fornecer o domínio aprovado e decidir
se a terminação HTTPS ficará neste host ou em uma borda institucional. Se o Nginx
ficar no mesmo host, altere `OCRSD_BIND_ADDRESS` para `127.0.0.1`, recrie o serviço `web`
com Compose e siga a configuração TLS abaixo. Para acesso direto apenas pela VPN,
a equipe de rede deve liberar a porta escolhida somente para a origem autorizada.

### DNS, HTTPS e rede

Defina `OCRSD_DOMAIN` como o nome efetivamente aprovado; solicite seu registro A/AAAA
apontando para a borda autorizada. A validação HTTP-01 abaixo exige porta 80 acessível
pela autoridade certificadora; HTTPS exige 443. Se o serviço for intranet/VPN,
use certificado da instituição ou desafio DNS aprovado, em vez de presumir acesso
público. Não habilite regras de firewall sem aprovação. Quando UFW já for a política
adotada, regras típicas aprovadas são `sudo ufw allow 80/tcp` e
`sudo ufw allow 443/tcp`; preserve a regra de administração SSH antes de qualquer
ativação. Não abra 8000/8080 externamente.

Bootstrap HTTP para obter o primeiro certificado:

```bash
export OCRSD_DOMAIN=DOMINIO_APROVADO
sudo mkdir -p /var/www/letsencrypt
printf 'server { listen 80; server_name %s; location /.well-known/acme-challenge/ { root /var/www/letsencrypt; } location / { return 404; } }\n' "$OCRSD_DOMAIN" | sudo tee /etc/nginx/sites-available/ocrsd
sudo ln -sfn /etc/nginx/sites-available/ocrsd /etc/nginx/sites-enabled/ocrsd
sudo nginx -t
sudo systemctl reload nginx
sudo certbot certonly --webroot -w /var/www/letsencrypt -d "$OCRSD_DOMAIN"
sed "s/DOMINIO_APROVADO/$OCRSD_DOMAIN/g" deploy/institutional.nginx.conf | sudo tee /etc/nginx/sites-available/ocrsd
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl enable --now certbot.timer
sudo install -d /etc/letsencrypt/renewal-hooks/deploy
printf '#!/bin/sh\nsystemctl reload nginx\n' | sudo tee /etc/letsencrypt/renewal-hooks/deploy/ocrsd-reload >/dev/null
sudo chmod 755 /etc/letsencrypt/renewal-hooks/deploy/ocrsd-reload
sudo certbot renew --dry-run
curl --fail "https://$OCRSD_DOMAIN/api/v1/health"
```

Adapte caminhos de certificado se a instituição usar outra autoridade. Não sobreponha
virtual hosts existentes: revise o resultado de `nginx -T` com a infraestrutura.
O mesmo domínio serve `/` e `/api/`. De outra máquina na rede autorizada:

```bash
curl --fail "https://$OCRSD_DOMAIN/api/v1/health"
curl --fail "https://$OCRSD_DOMAIN/api/v1/document-types"
```

Importe `postman/servidor.postman_environment.json`, substitua o domínio e execute
a coleção. Verifique `worker_ready`, criação/conclusão, aviso no PDF, download ZIP e
cancelamento. Certificados válidos devem funcionar sem desabilitar verificação TLS.

## Alternativa sem Docker

Use Debian 12 com Python 3.11, mantendo código em `/opt/ocrsd`, dados em
`/var/lib/ocrsd` e processos sem privilégios. Instalação pelo administrador:

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates python3.11 python3.11-venv nginx certbot
sudo useradd --system --home-dir /var/lib/ocrsd --shell /usr/sbin/nologin ocrsd
sudo install -d -o ocrsd -g ocrsd -m 700 /var/lib/ocrsd
sudo git clone https://github.com/siqueiratf/ocrsd.git /opt/ocrsd
sudo python3.11 -m venv /opt/ocrsd-tools
sudo /opt/ocrsd-tools/bin/pip install uv==0.12.10
cd /opt/ocrsd
sudo /opt/ocrsd-tools/bin/uv sync --locked --extra web --python /usr/bin/python3.11
sudo cp deploy/.env.example /etc/ocrsd.env
sudo chown root:ocrsd /etc/ocrsd.env
sudo chmod 640 /etc/ocrsd.env
sudo cp deploy/ocrsd-api.service deploy/ocrsd-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ocrsd-api ocrsd-worker
sudo systemctl status ocrsd-api ocrsd-worker --no-pager
```

As unidades definem `OCRSD_STORAGE` e variáveis de threads, limite de memória,
restart e parada graciosa. Para publicar a interface em 8080, mantenha o mesmo proxy
TLS descrito acima e configure o servidor interno Nginx:

```bash
sed -e 's/listen 8080;/listen 127.0.0.1:8080;/' -e 's|http://api:8000|http://127.0.0.1:8000|g' deploy/nginx.conf | sudo tee /etc/nginx/conf.d/ocrsd-local.conf
sudo nginx -t
sudo systemctl reload nginx
curl --fail http://127.0.0.1:8080/api/v1/health
```

Nessa alternativa, a API só confia no proxy em 127.0.0.1. Complete DNS, TLS e testes
externos com os mesmos passos da seção anterior. Não sirva a pasta do repositório
nem `/var/lib/ocrsd` diretamente pelo Nginx. A API entrega as páginas e os arquivos estáticos.

## Operação, atualização e recuperação

Com Compose, na raiz do checkout, use `sudo docker compose --env-file deploy/.env` onde necessário:

```bash
docker compose --env-file deploy/.env start
docker compose --env-file deploy/.env stop
docker compose --env-file deploy/.env restart api web
docker compose --env-file deploy/.env logs --tail=100 api worker web
docker compose --env-file deploy/.env logs -f worker
docker compose --env-file deploy/.env ps
docker compose --env-file deploy/.env exec worker python -m src.web.worker --health
docker compose --env-file deploy/.env exec api python -c 'from src.web.jobs import JobStore; from src.web.settings import Settings; s=JobStore(Settings()); print(s.root); print(s.ready())'
docker compose --env-file deploy/.env exec api df -h /data
```

`restart` não relê `env_file`; use `up -d --force-recreate api` após mudar configuração.
Worker possui 16 minutos de tolerância no shutdown. Se aumentar timeout, aumente
também `stop_grace_period` e `TimeoutStopSec`. O serviço reinicia após reboot pelas
políticas Compose/Docker ou pelas unidades systemd habilitadas.

Para atualizar sem interromper jobs ativos: prepare/teste a revisão em checkout
separado, mantenha o volume, suspenda novas submissões, drene a fila e só então troque
o worker. Exemplo no checkout de implantação, com alterações locais revisadas:

```bash
sed -i 's/OCRSD_ACCEPTING_JOBS=true/OCRSD_ACCEPTING_JOBS=false/' deploy/.env
docker compose --env-file deploy/.env up -d --no-deps --force-recreate api
docker compose --env-file deploy/.env exec api python -c 'import sqlite3; from src.web.settings import Settings; print(sqlite3.connect(Settings().storage / "jobs.sqlite3").execute("SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall())'
```

Aguarde `queued` e `running` zerarem. Consulte status também pelo Postman. Depois:

```bash
docker compose --env-file deploy/.env stop worker
git fetch --tags
git checkout REVISAO_APROVADA
docker compose --env-file deploy/.env build
docker compose --env-file deploy/.env up -d
curl --fail http://127.0.0.1:8080/api/v1/health
sed -i 's/OCRSD_ACCEPTING_JOBS=false/OCRSD_ACCEPTING_JOBS=true/' deploy/.env
docker compose --env-file deploy/.env up -d --no-deps --force-recreate api
```

Não faça `git pull` que modifique código carregado por um worker nativo ativo. No
systemd: altere `/etc/ocrsd.env`, reinicie só a API, aguarde a fila esvaziar, pare
`ocrsd-worker`, faça checkout/sync/build/cópia dos estáticos e reinicie ambas unidades.
Use `sudo systemctl stop|start|restart ocrsd-api ocrsd-worker` conforme a operação e
`sudo journalctl -u ocrsd-api -u ocrsd-worker -n 100 -f` para logs. Uma janela curta de
manutenção é preferível a alterar catálogo/pesos durante jobs. Migrações futuras do
schema deverão incluir backup e política de compatibilidade; esta versão apenas
cria as tabelas iniciais. Para rollback, restaure a revisão/imagem anterior e um
backup compatível se houver migração.

Faça backup consistente com API e worker parados, copiando o volume completo para
armazenamento autorizado, inclusive SQLite/WAL e permissões. No modo nativo, com os
serviços parados: `sudo tar -czf /CAMINHO_APROVADO/ocrsd-backup.tgz -C /var/lib ocrsd`.
No Compose, localize o volume com `docker volume ls` e inspecione seu ponto de montagem
com `docker volume inspect NOME_EXATO`; a cópia é tarefa do administrador. Backups
contêm saídas, opções e hashes das chaves e precisam de retenção própria.

A limpeza normal é automática pelo worker. Para executá-la manualmente no Compose:

```bash
docker compose --env-file deploy/.env exec api python -c 'from src.web.jobs import JobStore; from src.web.settings import Settings; JobStore(Settings()).cleanup()'
```

Esse comando remove apenas resultados já expirados. Reduzir retenção muda jobs que
terminarem depois da alteração, não as datas já persistidas. Não remova diretórios
manualmente durante jobs ativos. Logs do worker incluem IDs e detalhes técnicos;
respostas públicas não incluem tracebacks. O Compose limita logs locais a três arquivos de 10 MiB por serviço. Ajuste a
rotação Docker/journald/Nginx segundo a política institucional, sem registrar tokens Authorization.

Se a saúde estiver degradada, confira logs, heartbeat e existência de outro worker
no mesmo armazenamento. Jobs falhos podem indicar limite de memória/tempo, disco,
mapa incompatível ou pesos ausentes; procure pelo ID. Erros 429 podem ser cota por
IP, fila cheia ou reserva de armazenamento. Em 502, confira API/porta/rede do proxy;
em falha TLS, confira DNS, validade/caminhos dos certificados e `nginx -t`.

Modelos neurais só devem ser habilitados após instalação/verificação dos pesos e
medição do consumo no servidor. No modo nativo, use os comandos `make download-*`
do README durante a preparação, fora do worker. O Compose monta o caminho definido
em `OCRSD_HANDWRITING_ASSETS` somente para leitura em `/app/assets/handwriting`, na
API e no worker; o build continua excluindo pesos das imagens. Não baixe modelos
durante requisições. Fontes, modelos e templates precisam de revisão de licença institucional.

## Limites atuais e referências

Ainda não implementados na web: autenticação institucional e recuperação de conta,
papéis/grupos, compartilhamento de presets ou configurações, comparação OCR,
processamento de documentos reais, ajuste livre de cada parâmetro dos filtros e
pós-processamento avulso de um artefato concluído. A galeria neural depende dos pesos
aprovados no servidor. O canvas preserva subelementos já existentes no JSON, mas não
oferece uma ferramenta gráfica dedicada para desenhar cada subelemento. Antes de
expor documentos reais, defina autenticação, autorização, retenção segregada e uma
implantação privada aprovada; não reutilize a superfície por token como login final.

As decisões de implantação usam a documentação de
[processos FastAPI](https://fastapi.tiangolo.com/deployment/server-workers/),
[variáveis do Compose](https://docs.docker.com/compose/how-tos/environment-variables/set-environment-variables/)
e [endereços confiáveis no Nginx](https://nginx.org/en/docs/http/ngx_http_realip_module.html).
O servidor 2 foi implantado e validado pela VPN. DNS, certificado TLS e a política
da borda institucional não foram alterados.

Para releases, mantenha versão/changelog e os pontos de entrada `ocrsd`,
`python -m src.pipeline`, `python -m src.ocr_bench` e `python -m src.gui`.
O workflow de release valida documentos antes de publicar tags `V*`; não publique
pesos, dados reais, preferências locais ou saídas temporárias no repositório.
