# Certidão Simplificada (DREI) — regras de preenchimento

## Contexto

A Certidão Simplificada é expedida pelas Juntas Comerciais estaduais (sistema
SINREM) e consiste em um extrato atualizado dos dados constantes dos atos
arquivados de um empresário, sociedade ou outro ente do Registro Público de
Empresas. Este documento usa o modelo "Certidão Simplificada para Empresário
e suas Filiais" do Anexo VIII da Instrução Normativa DREI nº 81, de 10 de
junho de 2020 (consolidada, com a revogação do modelo específico para EIRELI
pela IN DREI nº 112/2022). Fonte:
https://www.gov.br/empresas-e-negocios/pt-br/drei/legislacao/arquivos/legislacoes-federais/IN812020AnexoVIIIModelosdeCertidesRevogadoModelodeCertidoSimplificadaparaEIRELIpelaIN112de2022V2.pdf

## Regras de preenchimento

- CNPJ no formato 00.000.000/0000-00 (14 dígitos).
- Natureza jurídica sempre "EMPRESÁRIO" (individual) — modelo específico
  desse ente; não usar outras naturezas.
- "Nome Empresarial": no empresário individual, normalmente é o nome civil
  do titular (eventualmente + elemento fantasia) — `(convenção)`.
- Datas em DD/MM/AAAA: arquivamento do ato de inscrição, início de
  atividade, último arquivamento.
- Capital: valor em reais, vírgula decimal (ex.: 15000,00).
- CPF do titular em 000.000.000-00; RG/Identidade em formato livre com
  pontuação (ex.: 12.345.678-9).
- Estado civil: SOLTEIRO(A), CASADO(A), VIÚVO(A) ou DIVORCIADO(A).
  "Regime de bens" só se aplica a CASADO(A) `(convenção)`; frequentemente
  em branco.
- "Ato/eventos" do último arquivamento: ato típico de empresário individual
  (INSCRIÇÃO, ALTERAÇÃO, TRANSFORMAÇÃO, EXTINÇÃO).
- Número do protocolo: sequência numérica, sem máscara fixa no modelo
  oficial `(convenção: 9–10 dígitos)`.
- UF da Junta Comercial: sigla válida (ex.: SP, RJ, MG).
- Data de emissão: por extenso ("DD de mês de AAAA").

## Coerência entre campos

- "Data de início de atividade" ≥ "Data de arquivamento do ato de
  inscrição" `(convenção)`.
- "Data do último arquivamento" ≥ "Data de arquivamento do ato de
  inscrição" `(convenção)`.
- "Data de emissão" ≥ todas as demais datas do documento `(convenção)`.
- CPF do campo "Identidade" refere-se à mesma pessoa do "Nome do
  Empresário" (titular certificado).
- "Regime de bens" só preenchido quando "Estado civil" = CASADO(A)
  `(convenção)`.
