# Desenvolvimento

Prepare o ambiente com `uv sync --locked`. O pacote `src` contém a aplicação e
preserva os pontos de entrada `ocrsd`, `python -m src.pipeline`,
`python -m src.ocr_bench` e `python -m src.gui`.

## Verificação

```bash
make check
uv run python -m compileall -q src tests scripts
```

O script `scripts.smoke_generate` usa exclusivamente templates e mapas
versionados. Gera um documento de cada tipo e confere que todos os campos
selecionados aparecem no PDF, são detectados e têm recortes válidos. Também
confere a ausência de camada de texto no PDF ocultado e a cor dos campos
ocultados. O OCR é dispensado nesse teste para permitir execução sem pesos.

Os testes cobrem separadamente CLI, mapas, PDF, pré-processamento, filtros,
geração sintética, presets, inferência neural e auxiliares das interfaces.
A inferência com pesos reais e a interação manual com a interface dependem do
ambiente e dos modelos locais.

## Catálogo

Cada tipo fica em `documents/<tipo>/`. O catálogo exige um PDF de template e
um mapa em `mappings/*.json` ou `<tipo>_fields.json`. Os mapas podem incluir
caixas compostas, descrições, tipos de dados e validações. Os presets de
imagem ficam em `presets/`; resultados de geração ficam em `dataset/`.
Somente `.gitkeep` é versionado nas pastas de datasets.

## Publicação

Atualize a versão em `pyproject.toml`, execute `uv lock` e registre as alterações
em `CHANGELOG.md` usando um título `## [TAG]`. Depois da validação:

```bash
git tag -a V1 -m 'OCRSD V1 — 1.0.0'
git push origin main V1
```

O workflow `release.yml` executa os testes e a validação sintética antes de
publicar o release. A publicação usa o token temporário do GitHub Actions e
não exige armazenar credenciais no código. O arquivo `ocrsd-V1.tar.gz` contém
os arquivos versionados; `synthetic-validation-V1.zip` contém documentos
sintéticos e seus manifestos. Uma nota Git em `refs/notes/releases` registra o
identificador, a URL e os arquivos publicados após a conclusão da publicação.
