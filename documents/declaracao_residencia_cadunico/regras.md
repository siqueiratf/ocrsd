# Declaração de Residência — Cadastro Único (CadÚnico)

## Contexto

Documento: "Modelo de Declaração de Residência", publicado pelo Ministério do
Desenvolvimento e Assistência Social (MDS) para uso no Cadastro Único para
Programas Sociais (CadÚnico). É um modelo de declaração, com força de
autodeclaração sob as penas da lei, usado quando a família não possui
comprovante de endereço em seu nome. O modelo tem duas variantes na mesma
publicação ("Responsável Familiar" e "Representante Legal"); este documento
reproduz a variante "Responsável Familiar" (página 1), a mais comum.
Fonte oficial: https://mds.gov.br/webarquivos/MDS/2_Acoes_e_Programas/Cadastro_Unico/Arquivos/Modelo_Declaracao_de_Residencia.pdf

## Regras de preenchimento

- O declarante se identifica pelo nome completo e por **um dos dois**
  documentos impressos no modelo: CPF (`000.000.000-00`) **ou** Título de
  Eleitor (sequência numérica, sem pontuação). O texto usa "ou" entre os
  dois — `(convenção)`: título de eleitor tem `blank_probability` alta, por
  ser a alternativa menos comum ao CPF.
- Endereço: o mesmo "indicado no Cadastro Único", residencial plausível
  (rua/avenida, número, complemento opcional, bairro, município) —
  `(convenção)`: o modelo só traz duas linhas em branco, sem formato exigido.
- Texto fixo/impresso (não alterar nem usar como fonte de valores): declaração
  "sob as penas da lei" (art. 299 do Código Penal) citando art. 3º, I, do
  Decreto nº 11.016/2022 e art. 18 da MP nº 1.164/2023.
- "Local e data" é uma única linha impressa acima da assinatura —
  `(convenção)`: o gerador preenche só a data (sem município), como já feito
  em outros mapeamentos do projeto (ex.: "Local e data de emissão" da
  certidão de óbito).
- Assinatura: do(a) "Responsável pela Unidade Familiar", mesma pessoa do
  início da declaração.

## Coerência entre campos

- Nome do declarante e nome da assinatura devem ser a mesma pessoa —
  `(convenção, sem vínculo automático no gerador)`.
- Preencher CPF **ou** Título de Eleitor, raramente os dois — `(convenção)`.
- Data em "Local e data": plausível, não futura.
