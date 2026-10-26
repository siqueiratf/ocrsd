# CNPJ — Quadro de Sócios e Administradores (QSA)

## Contexto

Reconstrução do quadro "Consultar QSA", exibido ao final do "Comprovante de
Inscrição e de Situação Cadastral" na consulta pública de CNPJ da Receita
Federal do Brasil. Não existe um PDF oficial em branco publicado
separadamente para esse trecho; o layout foi recomposto a partir da
descrição textual oficial dos campos da consulta e das colunas documentadas
no leiaute público dos Dados Abertos do CNPJ (arquivo de sócios), que usa a
mesma estrutura de campos. O documento traz o cabeçalho (CNPJ, nome
empresarial, capital social) e uma tabela com 4 linhas de sócios/
administradores — `(convenção)`: a Receita exibe quantas linhas existirem no
cadastro; 4 é um número ilustrativo fixado pelo projeto.

## Regras de preenchimento

- CNPJ: `00.000.000/0000-00`. **Limitação**: a máscara é preenchida com
  dígitos aleatórios; o gerador não calcula os dígitos verificadores reais
  do CNPJ.
- "Nome Empresarial" usa o mesmo gerador de nome de pessoa física do projeto
  (por conter "nome" no rótulo) — `(convenção, limitação do gerador)`, igual
  à prática já adotada em `drei_certidao_simplificada` para o mesmo campo.
- Cada linha de sócio representa, por simplificação, um sócio pessoa física:
  o campo "Nome" usa o gerador de nome de pessoa — `(convenção)`. O caso de
  sócio pessoa jurídica (nome empresarial na linha) não é gerado nesta
  versão.
- "Qualificação": lista fechada com papéis usuais (Sócio-Administrador,
  Sócio, Administrador, Diretor, Presidente, Sócio-Gerente), sem os códigos
  numéricos da Tabela de Qualificação de Sócios e Administradores da Receita
  Federal — `(convenção)`.
- "Representante Legal" só é preenchido quando o sócio é pessoa jurídica,
  incapaz ou estrangeiro sem procurador no Brasil — por isso tem alta
  `blank_probability` (0,85).
- "Nome no Exterior" e "País de Origem" só se aplicam a sócio estrangeiro
  sem CPF — ambos com `blank_probability` de 0,9.
- Capital social: valor decimal plausível entre R$ 1.000 e R$ 5.000.000.

## Coerência entre campos

- Cada linha de sócio preenchida deve ter, no mínimo, nome e qualificação —
  `(convenção, sem vínculo automático no gerador)`.
- "Representante Legal", "Nome no Exterior" e "País de Origem" são
  exceções raras e tendem a ficar em branco na maioria das linhas, o que é
  esperado e não indica erro de preenchimento.
- Todos os campos têm `"semantic": false`: nenhum vínculo automático do
  catálogo semântico do projeto se aplica a um quadro societário de CNPJ.
