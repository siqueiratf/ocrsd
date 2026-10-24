# Regras de preenchimento — Procuração Pública (movimentação de pensão/benefício do INSS)

## Contexto
Procuração pública lavrada por Consulado brasileiro no exterior pela qual o(a)
outorgante nomeia um(a) procurador(a) para tratar assuntos de aposentadoria/benefício
junto ao INSS (receber pagamentos, fazer prova de vida, movimentar conta bancária
onde o benefício é creditado etc.). Modelo oficial de texto: "INSS - recebimento de
pensão do", MRE/Consulado-Geral do Brasil em Bruxelas
(https://www.gov.br/mre/pt-br/consulado-bruxelas/procuracoes/procuracao-publica). O
site do Consulado também informa que este é o único tipo de procuração pública com
taxa reduzida (5 euros, contra 20 euros para as demais finalidades).

## Regras de preenchimento
- CPF no formato 000.000.000-00 (pode ser declarado).
- Datas no formato DD/MM/AAAA; a data da lavratura ("Aos ...") é por extenso.
- Nacionalidade e estado civil em maiúsculas (convenção).
- A cláusula de poderes é o texto oficial do modelo "INSS - recebimento de pensão
  do" e deve permanecer inalterada; a única variável citada no próprio texto
  oficial é a data de validade final.
- O bloco "Dados do benefício" (NB e espécie) é um acréscimo (convenção) feito para
  identificar qual benefício está sendo tratado, já que o texto oficial do Itamaraty
  não traz nenhum campo específico para isso. O número do benefício (NB) segue o
  padrão de 10 algarismos do INSS (sem os dígitos verificadores/formatação de
  pontuação, que variam conforme o sistema).

## Coerência entre campos
- Outorgante e outorgado devem ser pessoas diferentes; não garantido automaticamente
  entre dois campos `person_name` independentes (convenção/limitação conhecida).
- `data_fechamento` deveria repetir a mesma data de `data_lavratura`
  (convenção/limitação conhecida).
- `data_validade` deveria ser posterior a `data_lavratura` (convenção/limitação
  conhecida).
- `outorgante_cpf`/`outorgado_cpf` e `outorgante_numero_documento`/
  `outorgado_numero_documento` podem coincidir por usarem a mesma lista pequena de
  valores de exemplo (convenção/limitação conhecida).
- `especie_beneficio` "PENSÃO POR MORTE" pressupõe que o/a outorgante é
  dependente/pensionista de alguém falecido, e não o próprio titular aposentado; essa
  coerência entre espécie e demais dados da pessoa não é verificada pelo gerador
  (convenção/limitação conhecida).
