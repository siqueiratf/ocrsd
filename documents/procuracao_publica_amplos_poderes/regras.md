# Regras de preenchimento — Procuração Pública (amplos/plenos poderes)

## Contexto
Procuração pública lavrada por Consulado brasileiro no exterior (ex.: Consulado-Geral
do Brasil em Bruxelas), pela qual o(a) outorgante nomeia um(a) procurador(a) com
poderes gerais ("plenos poderes") para representá-lo em juízo, repartições públicas,
bancos, INSS, DETRAN, cartórios etc. Base legal: a procuração pública é formalizada
perante autoridade consular com fé pública (art. 18 da Lei de Introdução às Normas do
Direito Brasileiro, e regulamento consular do Itamaraty). Modelo oficial de texto:
"Plenos poderes - simplificada", MRE/Consulado-Geral do Brasil em Bruxelas
(https://www.gov.br/mre/pt-br/consulado-bruxelas/procuracoes/procuracao-publica).

## Regras de preenchimento
- CPF no formato 000.000.000-00 (pode ser declarado).
- Datas de validade/lavratura/fechamento em DD/MM/AAAA; a data da lavratura
  ("Aos ...") é por extenso.
- Nacionalidade e estado civil em maiúsculas (ex.: "BRASILEIRO(A)",
  "CASADO(A)"), convenção de atos notariais brasileiros `(convenção)`.
- Documento de identificação: RG, passaporte, CNH ou, para estrangeiros,
  RNM/RNE.
- A cláusula de poderes ("concedendo-lhe poderes para tratar e defender...")
  é o texto oficial do modelo "Plenos poderes - simplificada" e deve
  permanecer inalterada; só a data final de validade é variável.
- Livro, folha e ato referem-se ao livro de procurações do Consulado onde o
  instrumento é arquivado.
- Pagamento/valor do serviço não é dado do instrumento; não é campo deste
  modelo.

## Coerência entre campos
- Outorgante e outorgado devem ser pessoas diferentes; dois campos
  `person_name` independentes, sem garantia automática de nomes distintos
  (convenção/limitação conhecida).
- `data_fechamento` deveria repetir `data_lavratura` (mesmo ato), mas os
  campos são independentes e podem divergir (convenção/limitação).
- `data_validade` deveria ser posterior a `data_lavratura`, não garantido
  automaticamente (convenção/limitação).
- `outorgante_cpf`/`outorgado_cpf` e os números de documento usam a mesma
  lista pequena de `allowed_values`, podendo coincidir entre as duas pessoas
  (convenção/limitação).
- `cargo_autoridade_consular` deve ser compatível com autoridade com fé
  pública no Consulado (Cônsul-Geral, Cônsul, Vice-Cônsul ou chefe do setor
  consular).
