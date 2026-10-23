.DEFAULT_GOAL := help

UV ?= uv
UV_CACHE_DIR ?= /tmp/ocrsd-uv-cache-$(shell id -u)
export UV_CACHE_DIR
RUN ?= $(UV) run --offline --no-sync ocrsd
TEMPLATE ?= documents/dnv/template.pdf
MAPPING ?= documents/dnv/mappings/mapeamento_seguindo_norma.json
MODEL ?= font
COUNT ?= 10
SEED ?= 42
FIELDS ?=
INTENSITY ?= 1.0
WRITER ?=
BIAS ?= $(if $(filter aaron1259 wpvnteam,$(MODEL)),2.1,0.75)
PEN_COLOR ?= \#193787
PEN_WIDTH ?= 2
PEN_PRESSURE ?= 0.3
PEN_DROPOUT ?= 0.1
TEXT_POLICY ?= normalize
THREADS ?= 1
PORT ?=
API_PORT ?= $(if $(strip $(PORT)),$(PORT),8000)

.PHONY: help dataset dataset-font dataset-sjvasquez dataset-aaron1259 dataset-wpvnteam
.PHONY: download-sjvasquez download-aaron1259 download-wpvnteam
.PHONY: test smoke check gui web web-clean

WEB_PY_STAMP := .venv/.ocrsd-web-dependencies

help:
	@printf '%s\n' \
	  'make dataset                  Gera 10 DNVs com font e os campos suportados.' \
	  'make dataset-sjvasquez         Usa o modelo neural sjvasquez.' \
	  'make dataset-aaron1259         Usa o modelo neural aaron1259.' \
	  'make dataset-wpvnteam          Usa o modelo neural wpvnteam.' \
	  'make download-sjvasquez        Prepara os pesos do sjvasquez.' \
	  'make download-aaron1259        Prepara os pesos do aaron1259.' \
	  'make download-wpvnteam         Prepara os pesos do wpvnteam.' \
	  'make dataset COUNT=5 SEED=123' \
	  'make dataset-sjvasquez WRITER=9 PEN_WIDTH=2' \
	  'make dataset MAPPING=caminho/mapa.json TEMPLATE=caminho/template.pdf' \
	  'make dataset FIELDS="campo_1 campo_2"' \
	  'make check                    Executa os testes e gera/processa uma DNV e uma certidao.' \
	  'make smoke                    Valida geracao, alinhamento, recortes e ocultacao.' \
	  'make web                      Prepara e inicia a interface web, a API e o worker.' \
	  'make web PORT=8001            Usa outra porta local para a interface e a API.' \
	  'make web-clean                Inicia web com armazenamento vazio; mantém modelos padrão do catálogo.' \
	  'make gui                      Abre a interface desktop.' \
	  '' \
	  'Cada execucao cria documents/<tipo>/dataset/yyyymmdd_hhmm-<tipo>-<total>.' \
	  'WRITER vazio sorteia por documento.' \
	  'Requer ambiente uv preparado e pesos locais para os modelos neurais.'

dataset:
	env OMP_NUM_THREADS='$(THREADS)' MKL_NUM_THREADS='$(THREADS)' $(RUN) generate \
	  --template '$(TEMPLATE)' --field-map '$(MAPPING)' \
	  --count '$(COUNT)' --seed '$(SEED)' --handwriting-model '$(MODEL)' \
	  --intensity '$(INTENSITY)' --bias '$(BIAS)' \
	  --pen-color '$(PEN_COLOR)' --pen-width '$(PEN_WIDTH)' \
	  --pen-pressure '$(PEN_PRESSURE)' --pen-dropout '$(PEN_DROPOUT)' \
	  --text-policy '$(TEXT_POLICY)' \
	  $(if $(strip $(WRITER)),--writer '$(WRITER)') \
	  $(if $(strip $(FIELDS)),--fields $(foreach field,$(FIELDS),'$(field)'))

dataset-font:
	@$(MAKE) dataset MODEL=font

dataset-sjvasquez:
	@$(MAKE) dataset MODEL=sjvasquez

dataset-aaron1259:
	@$(MAKE) dataset MODEL=aaron1259

dataset-wpvnteam:
	@$(MAKE) dataset MODEL=wpvnteam

download-sjvasquez:
	$(RUN) handwriting --model sjvasquez --download

download-aaron1259:
	$(RUN) handwriting --model aaron1259 --download

download-wpvnteam:
	$(RUN) handwriting --model wpvnteam --download

test:
	$(UV) run --offline --no-sync python -m unittest discover -s tests -v

smoke:
	env OMP_NUM_THREADS='$(THREADS)' MKL_NUM_THREADS='$(THREADS)' $(UV) run --offline --no-sync python -m scripts.smoke_generate --seed '$(SEED)'

check: test smoke

gui:
	$(UV) run --offline --no-sync python -m src.gui

$(WEB_PY_STAMP): pyproject.toml uv.lock
	$(UV) sync --locked --extra web --extra test
	@touch '$(WEB_PY_STAMP)'

web: $(WEB_PY_STAMP)
	env OCRSD_NEURAL_ENABLED=true OCRSD_API_PORT='$(API_PORT)' $(UV) run --offline --no-sync python -m scripts.run_web

web-clean: $(WEB_PY_STAMP)
	@demo_storage=$$(mktemp -d /tmp/ocrsd-web-clean.XXXXXX); \
	  trap 'rm -rf "$$demo_storage"' EXIT; \
	  env OCRSD_STORAGE="$$demo_storage" OCRSD_NEURAL_ENABLED=true OCRSD_API_PORT='$(API_PORT)' $(UV) run --offline --no-sync python -m scripts.run_web
