# Regras de preenchimento — Procuração Pública (transferência de veículo)

## Contexto
Procuração pública lavrada por Consulado brasileiro no exterior pela qual o(a)
outorgante nomeia um(a) procurador(a) para tratar da transferência de documentos de
um veículo de sua propriedade junto ao DETRAN/CIRETRAN no Brasil. Modelo oficial de
texto: "Transferência de veículo", MRE/Consulado-Geral do Brasil em Bruxelas
(https://www.gov.br/mre/pt-br/consulado-bruxelas/procuracoes/procuracao-publica).

## Regras de preenchimento
- CPF no formato 000.000.000-00 (pode ser declarado).
- Data da lavratura ("Aos ...") por extenso; data de fechamento em DD/MM/AAAA.
- Nacionalidade e estado civil em maiúsculas (convenção).
- Chassi (VIN) com 17 caracteres alfanuméricos maiúsculos, sem as letras I, O, Q
  (convenção de formato do padrão VIN, não detalhada no texto do Itamaraty).
- Código RENAVAM com 11 algarismos.
- Placa no padrão antigo brasileiro (AAA-0000) ou no padrão Mercosul (AAA0A00).
- O texto oficial cita, para o veículo: descrição, cor predominante, chassi,
  RENAVAM, placa, cidade e estado de emplacamento; e, para o DETRAN competente: o
  nome do estado por extenso, a sigla da UF e a cidade-UF da unidade.
- Diferentemente dos demais modelos de procuração pública consultados neste
  Consulado, o modelo "Transferência de veículo" não contém, no texto oficial, a
  frase final "Esta procuração é válida até ___/___/___"; por isso este documento
  não tem campo `data_validade`.

## Coerência entre campos
- Outorgante e outorgado devem ser pessoas diferentes; não garantido entre
  dois campos `person_name` independentes (convenção/limitação).
- `data_fechamento` deveria repetir `data_lavratura` (convenção/limitação).
- `municipio_veiculo`/`uf_veiculo` deveriam coincidir com
  `estado_detran`/`sigla_uf_detran`/`municipio_sigla_detran` (o DETRAN
  competente costuma ser o do estado de emplacamento); campos independentes,
  sem garantia de coerência geográfica (convenção/limitação).
- CPFs e números de documento de outorgante/outorgado podem coincidir por
  usarem a mesma lista pequena de exemplos (convenção/limitação).
- NOTA SOBRE DEFEITO DO GERADOR: um campo de texto livre (sem
  `allowed_values`) cujo id/label/descrição contenha a sequência "idade" —
  mesmo dentro de outra palavra ("cidade", "unidade", "nacionalidade") —
  pode ser convertido indevidamente em inteiro pela heurística antiga de
  `src/synthetic_dataset.py` (defeito distinto do já conhecido para a
  palavra "registro" em campos inteiros). Por isso o campo de cidade/UF do
  DETRAN foi nomeado `municipio_sigla_detran` e recebeu `allowed_values`.
