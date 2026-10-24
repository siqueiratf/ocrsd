# Regras de preenchimento — Procuração Pública (venda de imóvel)

## Contexto
Procuração pública lavrada por Consulado brasileiro no exterior pela qual o(a)
outorgante nomeia um(a) procurador(a) com poderes específicos para vender um imóvel
de sua propriedade no Brasil (assinar escritura de venda, receber e dar quitação
etc.). Modelo oficial de texto: "Imóvel - venda de", MRE/Consulado-Geral do Brasil em
Bruxelas (https://www.gov.br/mre/pt-br/consulado-bruxelas/procuracoes/procuracao-publica).

## Regras de preenchimento
- CPF no formato 000.000.000-00 (pode ser declarado).
- Datas no formato DD/MM/AAAA; a data da lavratura ("Aos ...") é por extenso.
- Nacionalidade e estado civil em maiúsculas (convenção).
- O texto oficial cita três dados variáveis do imóvel: endereço (`endereco_imovel`),
  cidade (`municipio_imovel`) e estado (`uf_imovel`) — "imóvel situado em ___, na
  cidade de ___, estado ___".
- A cláusula de poderes é o texto oficial do modelo "Imóvel - venda de" e deve
  permanecer inalterada; apenas endereço/cidade/estado do imóvel e data de validade
  são variáveis.

## Coerência entre campos
- Outorgante e outorgado devem ser pessoas diferentes; não garantido automaticamente
  entre dois campos `person_name` independentes (convenção/limitação conhecida).
- `municipio_imovel` e `uf_imovel` deveriam ser geograficamente compatíveis (a cidade
  precisa pertencer ao estado informado); como são dois campos independentes
  (`municipality` e `uf`), o gerador sintético não garante essa coerência
  (convenção/limitação conhecida — valores gerados podem combinar cidade e UF
  incompatíveis).
- `data_fechamento` deveria repetir a mesma data de `data_lavratura`
  (convenção/limitação conhecida).
- `data_validade` deveria ser posterior a `data_lavratura` (convenção/limitação
  conhecida).
- `outorgante_cpf`/`outorgado_cpf` e `outorgante_numero_documento`/
  `outorgado_numero_documento` podem coincidir entre as duas pessoas por usarem a
  mesma lista pequena de valores de exemplo (`allowed_values`); isso é uma limitação
  do gerador, não uma regra do modelo (convenção/limitação conhecida).
