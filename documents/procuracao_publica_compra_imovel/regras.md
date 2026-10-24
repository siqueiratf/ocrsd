# Regras de preenchimento — Procuração Pública (compra de imóvel)

## Contexto
Procuração pública lavrada por Consulado brasileiro no exterior pela qual o(a)
outorgante nomeia um(a) procurador(a) com poderes específicos para comprar um imóvel
em seu nome no Brasil (assinar escritura, registrar, pagar sinal/parcelas etc.).
Modelo oficial de texto: "Imóvel - compra de", MRE/Consulado-Geral do Brasil em
Bruxelas (https://www.gov.br/mre/pt-br/consulado-bruxelas/procuracoes/procuracao-publica).

## Regras de preenchimento
- CPF no formato 000.000.000-00 (pode ser declarado).
- Datas no formato DD/MM/AAAA; a data da lavratura ("Aos ...") é por extenso.
- Nacionalidade e estado civil em maiúsculas (convenção, como nas certidões
  brasileiras).
- O endereço do imóvel (`endereco_imovel`) e o município (`municipio_imovel`) são os
  dois dados variáveis citados explicitamente no texto oficial ("imóvel/eis situado
  em ___, na cidade de ___, Brasil").
- A cláusula de poderes é o texto oficial do modelo "Imóvel - compra de" e deve
  permanecer inalterada; apenas endereço do imóvel, cidade e data de validade são
  variáveis.
- O texto oficial menciona "cartórios de registros de imóveis" e "registro de
  matrícula" como órgãos/atos envolvidos na aquisição, mas não exige um campo de
  matrícula específico no próprio instrumento de procuração (a matrícula só existirá
  depois da compra).

## Coerência entre campos
- Outorgante e outorgado devem ser pessoas diferentes; o gerador não garante isso
  automaticamente entre dois campos `person_name` independentes (convenção/limitação
  conhecida).
- `data_fechamento` deveria repetir a mesma data de `data_lavratura` (mesmo ato); são
  campos independentes no gerador (convenção/limitação conhecida).
- `data_validade` deveria ser posterior a `data_lavratura` (convenção/limitação
  conhecida).
- `outorgante_cpf`/`outorgado_cpf` e `outorgante_numero_documento`/
  `outorgado_numero_documento` podem coincidir por usarem a mesma lista pequena de
  valores de exemplo (convenção/limitação conhecida).
- O `municipio_imovel` não precisa coincidir com o endereço/município de residência
  do outorgante ou do outorgado (o imóvel comprado pode estar em cidade diferente de
  onde moram); não há, portanto, regra de coerência exigindo igualdade aqui.
