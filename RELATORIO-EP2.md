# EP2 - Clareza da resposta em sistemas e-SIC

ACH2118 - Introdução ao Processamento de Língua Natural

## Integrantes

- Cleben Junior Cosendey Garcia (14691220)
- João Pedro Oliveira de Paula Marcondes (14582570)
- Luis Henrique Moraes (14615590)
- Renan Bezerra Guimarães (14605640)

## Breve descrição

A tarefa é a mesma do EP1: classificar a clareza de uma resposta do e-SIC em {c1, c234, c5}, usando apenas o texto da resposta. O modelo do EP1 era um único BERTimbau com fine-tuning.

O modelo final do EP2 é um **ensemble por soft voting de quatro modelos**: três instâncias do BERTimbau (`neuralmind/bert-base-portuguese-cased`) com fine-tuning, diferindo apenas na semente, mais um **ComplementNB sobre TF-IDF de palavras** (uni e bigramas). Cada modelo produz uma distribuição sobre as três classes; as quatro distribuições são somadas, divididas por quatro, e a classe final é a de maior probabilidade. São portanto duas representações combinadas: embeddings contextuais (BERT) e contagens ponderadas de termos (TF-IDF).

A escolha dos membros não foi guiada pela acurácia individual de cada um, e sim pelo desempenho do conjunto. Isso se mostrou essencial: o ComplementNB é o pior dos quatro modelos isoladamente (0,4711 no teste1) e ainda assim é o que mais agrega, porque concorda com o BERT em apenas 63% dos casos. Modelos mais fortes mas redundantes foram testados e descartados (seção 9).

## Pré-processamento

O pré-processamento é mínimo e isso foi verificado experimentalmente, não assumido. Os textos recebem apenas preenchimento de valores ausentes com string vazia e conversão para texto; os rótulos têm espaços removidos e são validados contra {c1, c234, c5}. Para o BERT usa-se o tokenizador WordPiece nativo do modelo, com truncamento em 512 tokens; a versão *cased* preserva maiúsculas, acentuação e pontuação. Para o TF-IDF, a vetorização fica dentro do Pipeline do scikit-learn, recalculada em cada partição.

Quatro alternativas de pré-processamento foram avaliadas com validação cruzada de 5 folds no conjunto de desenvolvimento, e **todas apresentaram desempenho igual ou inferior ao texto cru**:

| Variante | Acurácia (logreg) | Acurácia (CNB) |
|---|---|---|
| Texto cru | **0,4565** | **0,4535** |
| Remoção de saudação e assinatura | 0,4518 | 0,4515 |
| Remoção de acentos e minúsculas | 0,4560 | 0,4530 |
| Truncagem nas 60 primeiras palavras | 0,4485 | 0,4527 |
| Remoção de boilerplate e truncagem | 0,4445 | 0,4477 |

A remoção de saudação e assinatura reduziu o texto em 16% (de 1001 para 839 caracteres em média) e piorou o resultado, o que indica que o boilerplate carrega sinal: o template usado identifica o órgão respondente, e órgãos diferem sistematicamente na qualidade das respostas (ver seção 9). A truncagem também piora, o que mostra que o sinal não está concentrado no início do texto.

## Parâmetros

Foram realizadas duas buscas em grade independentes, uma para cada família de modelo.

### TF-IDF + ComplementNB (270 combinações, 5 folds, 1350 ajustes)

| Parâmetro | Valores |
|---|---|
| ngram_range | {(1,1), (1,2), (1,3)} |
| min_df | {1, 2, 5} |
| max_df | {0.9, 0.95, 1.0} |
| sublinear_tf | {True, False} |
| alpha | {0.05, 0.1, 0.3, 0.5, 1.0} |

### BERTimbau (4 configurações, avaliadas no holdout)

| Parâmetro | Valores |
|---|---|
| learning_rate | {1e-5, 2e-5, 3e-5} |
| num_train_epochs | {2, 3, 4, 5} |
| per_device_train_batch_size | {8, 16, 32} |
| max_length | {384, 512} |
| warmup_steps | {500} |
| weight_decay | {0.01} |

Modelos adicionais avaliados como candidatos a membro do ensemble, com os respectivos parâmetros: ComplementNB sobre n-gramas de caracteres (3 a 5), MultinomialNB (alpha 0,3), LinearSVC calibrado (C 0,5), k-NN (k=45, distância de cosseno), centroide mais próximo, e gradient boosting sobre 150 componentes de SVD. Também foram avaliados o BERTimbau-large e o mDeBERTa-v3-base como alternativas de encoder.

## Valores Ótimos

Configuração adotada em cada membro do modelo final.

### BERTimbau (3 membros)

| Parâmetro | Valor |
|---|---|
| learning_rate | 2e-5 |
| num_train_epochs | 4 (com early stopping) |
| per_device_train_batch_size | 8 |
| max_length | 512 |
| warmup_steps | 500 |
| weight_decay | 0.01 |
| sementes | {42, 1, 2} |

### ComplementNB (1 membro)

| Parâmetro | Valor |
|---|---|
| ngram_range | (1,2) |
| min_df | 2 |
| max_df | 0.95 |
| sublinear_tf | True |
| alpha | 0.3 |

### Combinação

| Parâmetro | Valor |
|---|---|
| estratégia | soft voting (média das probabilidades) |
| pesos | iguais |

Dois pontos merecem registro, porque contrariam a expectativa usual de uma busca em grade.

O ótimo da busca 1 (alpha 0,5, min_df 1, max_df 0,9, sublinear_tf False) foi superior isoladamente (CV 0,4580 contra 0,4535 da configuração adotada), mas **reduziu a acurácia do ensemble** de 0,4760 para 0,4715. A razão é mensurável: a configuração ótima aumentou a concordância com o BERT de 0,631 para 0,650, reduzindo a diversidade que torna o membro útil. Adotou-se portanto a configuração com alpha 0,3, que é subótima individualmente e superior em conjunto.

Na busca 2, a configuração de melhor desempenho isolado foi learning_rate 3e-5 com 3 épocas (0,4583 no holdout, contra 0,4399 da configuração do EP1), mas o ensemble das três sementes com learning_rate 2e-5 e 4 épocas obteve 0,4760 em conjunto com o ComplementNB, contra 0,4710 da configuração isolada superior. Manteve-se a primeira.

Pesos não uniformes no ensemble também foram testados e descartados: o peso ótimo variou de 0,55 para 0,80 entre duas metades independentes do holdout, indicando sobreajuste do critério de seleção, e a média simples superou a ponderada em ambas as direções.

## Procedimento

O conjunto `train.xlsx` (20.092 exemplos) foi dividido de forma estratificada com `random_state=42` em 80% para desenvolvimento e 20% (4.019 exemplos) reservados como holdout, intocado até a avaliação final. A divisão é idêntica à do EP1, de modo que os resultados dos dois trabalhos são comparáveis sobre os mesmos exemplos.

Durante o desenvolvimento, cada membro candidato foi treinado no conjunto de desenvolvimento e avaliado no holdout. Para o BERT, a validação usada no early stopping foi retirada do próprio conjunto de desenvolvimento, nunca do holdout.

A seleção da combinação de membros exigiu um cuidado adicional. Como o número de combinações possíveis é grande, escolher a melhor olhando o holdout inteiro produziria uma estimativa otimista. Adotou-se portanto um protocolo de **seleção em metades**: o holdout foi dividido aleatoriamente em duas partes; a combinação foi escolhida em uma e o resultado reportado na outra, nas duas direções. As duas metades selecionaram independentemente a mesma combinação, e os valores reportados (0,4701 e 0,4818) confirmaram o ganho sem contaminação.

Para a entrega, os quatro membros foram re-treinados em todo o `train.xlsx` (com 10% reservado apenas para early stopping no caso do BERT) e aplicados ao `test2.xlsx`. O mesmo modelo foi aplicado ao `test1.xlsx` uma única vez, para o teste de McNemar da seção 8. Nenhum ajuste foi feito após observar o resultado do teste1, para não invalidar a comparação.

## Resultado

Acurácia no conjunto teste1 (900 exemplos, rótulos fornecidos com o EP2):

| Modelo | Acurácia | F1-macro |
|---|---|---|
| Classe majoritária | 0,3333 | - |
| Baseline oficial (TF-IDF + RegLog) | 0,4667 | - |
| ComplementNB isolado | 0,4711 | 0,4621 |
| BERTimbau, 3 sementes | 0,5011 | 0,4987 |
| **Ensemble final (EP2)** | **0,5044** | 0,4975 |

O modelo final supera o baseline oficial em 3,8 pontos percentuais e a classe majoritária em 17 pontos.

Os rótulos entregues para o `test2.xlsx` (897 exemplos) distribuem-se em 33,6% c1, 22,2% c234 e 44,3% c5.

## Evolução em relação ao EP1

| Estratégia | Teste | Acurácia |
|---|---|---|
| Modelo EP1 (BERTimbau único) | teste1 (900) | 0,4867 |
| Modelo EP2 (ensemble de 4) | teste1 (900) | **0,5044** |

O ganho é de **+1,78 ponto percentual**. A tabela de contingência do teste de McNemar sobre as 900 instâncias é:

| | EP2 correto | EP2 incorreto |
|---|---|---|
| **EP1 correto** | 351 | 87 |
| **EP1 incorreto** | 103 | 359 |

São 190 casos discordantes: 103 em que apenas o EP2 acerta e 87 em que apenas o EP1 acerta.

**Estatística de McNemar com correção de continuidade: chi2 = 1,1842, com 1 grau de liberdade, p = 0,2765.** O teste binomial exato bilateral fornece o mesmo valor (p = 0,2765).

Reportando o resultado de forma direta: **a vantagem do modelo do EP2 não é estatisticamente significativa a p < 0,05**. O ganho de 1,78 ponto é consistente em direção (o EP2 acerta 16 casos líquidos a mais), mas o número de discordâncias é insuficiente para descartar o acaso neste tamanho de amostra. Para alcançar p < 0,05 com 190 discordâncias seria necessária uma diferença líquida de aproximadamente 27 casos, o que corresponderia a cerca de 3 pontos de acurácia sobre o EP1.

Optou-se por reportar este resultado em vez de buscar significância por tentativas repetidas no teste1. Ajustar o modelo após observar o resultado do teste e medir novamente converteria o McNemar em uma busca por significância, invalidando a própria comparação que o teste se propõe a fazer.

## Análise

### (i) Interpretação global

As características mais importantes foram extraídas dos coeficientes de uma regressão logística sobre TF-IDF de palavras (uni e bigramas, min_df 5), treinada em todo o conjunto de treino. Os termos de maior peso por classe:

| Classe | Termos de maior peso positivo |
|---|---|
| c1 | luis (+1,29), demanda (+1,28), solicitação (+1,27), essa informação (+1,12), sr luis (+1,06), nup (+0,96), processo (+0,95) |
| c234 | sistemas (+1,40), prezado em (+1,21), territorial (+1,01), projetos (+0,96), extensão (+0,96), reitoria (+0,94), concurso (+0,90) |
| c5 | dnpm (+2,07), questionário (+1,89), sic dnpm (+1,61), solicitação encaminhamos (+1,55), ainda não (+1,39), nenhum (+1,37), olá (+1,23), odila (+1,19) |

Este é o resultado mais informativo do trabalho, e ele é negativo quanto à natureza do que o modelo aprendeu. **Os termos de maior peso não são marcadores de clareza, e sim identificadores de órgão e nomes de pessoas.** O termo de maior peso absoluto em todo o modelo é `dnpm` (+2,07 para c5), sigla do Departamento Nacional de Produção Mineral; `sic dnpm` aparece em seguida. Em c1 destacam-se `luis` e `sr luis`, e em c5 `odila`, que são nomes próprios de solicitantes ou de servidores. Termos como `reitoria`, `concurso` e `territorial` em c234 identificam domínio temático, não clareza.

A interpretação é que o modelo aprendeu majoritariamente **quem respondeu**, não **se a resposta foi clara**. Isso é um proxy legítimo e com poder preditivo real, uma vez que órgãos diferem sistematicamente na qualidade de suas respostas, mas não é a tarefa enunciada. Também explica por que a remoção de boilerplate piorou o desempenho (seção 3): o boilerplate é precisamente onde a identidade do órgão está codificada.

### (ii) Interpretação local

Dois exemplos classificados corretamente, com a contribuição individual de cada termo para a classe escolhida:

**Linha 3** (gold c1, predito c1, probabilidades c1 0,518 / c234 0,375 / c5 0,106). Texto: *"Prezada Senhora, De acordo com o art.13 do Decreto n. 7724, de 16 de maio de 2012, que regulamenta a Lei 12.527/2011, por meio deste canal de comunicação..."*. Termos de maior contribuição: `de` (-0,132), `uma` (-0,063), `emec` (+0,057), `consulta` (+0,044), `mec gov` (+0,034).

**Linha 2** (gold c5, predito c5, probabilidades c1 0,161 / c234 0,359 / c5 0,481). Texto: *"Prezada Srª. Raylane, Em resposta a sua solicitação informamos que a Fiocruz encaminhou o ofício nº 047/2017 ao Ministério da Saúde..."*. Termos de maior contribuição: `para que` (-0,073), `01` (+0,065), `que` (-0,054), `ao ministério` (-0,051).

O padrão relevante é a **magnitude**: nenhuma contribuição individual passa de 0,13 em valor absoluto, e as listas são dominadas por palavras funcionais (`de`, `que`, `uma`) e fragmentos sem relação semântica com clareza. A decisão é difusa, distribuída por centenas de termos de peso baixo, em vez de ancorada em evidência localizada. Classificadores com evidência localizada produzem contribuições de magnitude muito superior.

### (iii) Análise de erros

O modelo erra 446 das 900 instâncias do teste1. A distribuição dos erros:

| Gold | Predito c1 | Predito c234 | Predito c5 |
|---|---|---|---|
| c1 | - | 67 | 86 |
| c234 | 82 | - | 109 |
| c5 | 46 | 56 | - |

Duas estatísticas agregadas sobre os erros:

- **Comprimento do texto não discrimina**: a média é 952 caracteres nos acertos e 950 nos erros.
- **O modelo tem alguma calibração**: a confiança média (probabilidade máxima) é 0,534 nos acertos e 0,486 nos erros. Ele é mensuravelmente menos confiante quando erra, embora a diferença seja pequena.

Quatro casos incorretos, examinados manualmente:

**Linha 6** (gold c5, predito c234). Texto: *"A informação solicitada encontra-se na página da UFF, no link http://www.uff.br/sites/default/files/quadro_de_referencia_dos_cargos..."*. A resposta entrega um link direto para exatamente o que foi pedido, e o cidadão a avaliou com a nota máxima de clareza. O modelo hesitou (c234 0,421 contra c5 0,368), com contribuições dominadas por fragmentos de URL (`uff br`, `default`, `pdf`). O erro é de representação: o modelo não tem como saber que aquele link resolvia o pedido.

**Linha 0** (gold c5, predito c234). Texto sobre pesquisas domiciliares do IBGE, explicando que apenas o Censo Demográfico produz resultados municipais. Os termos de maior peso foram `demográfico`, `censo demográfico`, `censo`, ou seja, **assunto, não clareza**. A resposta é objetivamente explicativa e foi avaliada como muito clara, mas o modelo só captou o tópico.

**Linha 1** (gold c1, predito c234). Resposta do Ministério do Planejamento, iniciada por agradecimento protocolar. As contribuições foram `do`, `de`, `ministério do`, `sistemas`, isto é, inteiramente boilerplate e identificação de órgão. Não há no texto nenhum sinal que explique por que o cidadão a considerou pouco clara.

**Linha 4** (gold c234, predito c1). Resposta do INEP com protocolo e menção a microdados e portal. O modelo apoiou-se em `inep`, `portal`, `microdados`, novamente identificação de órgão e tema. A diferença entre nota 1 e notas intermediárias é indistinguível a partir do texto.

Os quatro casos têm a mesma estrutura: **a predição se apoia em identidade de órgão e assunto, e o rótulo depende de informação ausente dos dados**, a saber, qual foi a pergunta original e se a resposta atendeu à expectativa de quem perguntou.

### Considerações sobre o resultado

Ao longo do EP2 foram avaliadas cinco categorias de intervenção, com resultados no holdout:

| Categoria | Tentativas | Resultado |
|---|---|---|
| Regra de decisão | limiares sobre modelo binário dos extremos (0,4292), cascata neutro/extremo (0,4496), CORAL ordinal (0,4459), rebalanceamento para prior uniforme (0,4694) | nenhuma supera o baseline do ensemble |
| Limpeza de dados | deduplicação, remoção de textos conflitantes, colapso para rótulo majoritário | 0,4412 a 0,4429, sem ganho |
| Pré-processamento | boilerplate, normalização, truncagem | todas iguais ou piores |
| Capacidade do modelo | BERTimbau-large (0,4486), mDeBERTa-v3-base | não superam o base, nem agregam ao ensemble |
| Diversidade de membros | ComplementNB, char-CNB, MultinomialNB, k-NN, centroide, SVD+GBM, SVM calibrado | apenas o ComplementNB agrega (+1,9 ponto) |

Três medições explicam esse padrão.

**A decisão difícil é única e não é aprendível.** Uma cascata com oráculo no primeiro estágio (ou seja, recebendo de graça a resposta a "este caso é neutro?") alcançaria 0,7843 de acurácia, contra 0,4496 da cascata real. Toda a dificuldade está concentrada nessa decisão. Mas a AUC de P(c234) para separar neutro de extremo é de apenas 0,567, e a acurácia do primeiro estágio com limiar ótimo (0,6569) é inferior à do classificador trivial que sempre responde "extremo" (0,6589). O oráculo quantifica o valor da informação; a AUC mostra que ela não está nos dados.

Daí a razão pela qual nenhuma regra de decisão funciona: **limiares redistribuem decisões sobre a informação existente, não criam informação nova.** Três métodos independentes, livres para escolher como tratar a classe intermediária, convergiram ao mesmo comportamento: os limiares ajustados reduziram a faixa central a 6,6% das predições, e no CORAL os vieses aprendidos colapsaram (diferença de 0,068 após três épocas), levando o modelo a nunca prever c234.

**O rótulo contém ruído irredutível.** Em 10,6% das linhas de treino o texto da resposta é exatamente duplicado, e nessas linhas um oráculo que atribuísse a cada texto o seu rótulo mais frequente alcançaria apenas 63% de acerto. São 291 textos distintos com rótulos conflitantes. Entradas idênticas recebem rótulos humanos diferentes, o que estabelece um teto independente de qualquer modelo. Convém registrar a ressalva de que essa estimativa vem das linhas duplicadas, majoritariamente respostas padronizadas e possivelmente mais ambíguas que a média.

**O que o modelo aprende é um proxy.** A interpretação global mostra que as características de maior peso são identificadores de órgão e nomes próprios. O modelo prevê qualidade por procedência, e esse proxy tem poder preditivo real mas limitado.

A conclusão é que o problema não é difícil de modelar, e sim **mal especificado**: pede-se prever uma reação subjetiva (a clareza percebida por um cidadão específico, numa escala de 1 a 5 cujas notas intermediárias foram agregadas) a partir de uma entrada que não contém os determinantes dessa reação. Falta a pergunta original, falta o contexto da expectativa, e falta saber se o pedido foi atendido. Uma negativa redigida com clareza perfeita pode ser avaliada como pouco clara, e o texto não distingue os dois casos.

O ganho obtido no EP2 veio da única via que acrescenta informação em vez de reprocessá-la: combinar modelos cujos erros são descorrelacionados. Ganhos substancialmente maiores exigiriam alterar os dados, incorporando a pergunta do cidadão junto da resposta, e não trocar ou ajustar o classificador.

## Link para repositório do código

https://github.com/clebenjr/ep1-ipln

## Instruções

Dependências: Python 3.12 com `pandas`, `openpyxl`, `scikit-learn`, `scipy` para os modelos clássicos e as análises; `torch`, `transformers` e `accelerate` para o fine-tuning do BERTimbau; `eli5` opcionalmente para a interpretação.

```bash
pip install pandas openpyxl scikit-learn scipy torch transformers accelerate
```

Passo a passo para reproduzir o resultado final:

1. **Membros BERT.** Execute `experimentos/colab-membros-finais.ipynb` em ambiente com GPU (Google Colab atende; cerca de 10 minutos por semente). Ele treina as três sementes em todo o `train.xlsx` e gera `probs_final_test1_seed<N>.npz` e `probs_final_test2_seed<N>.npz`. Localmente o mesmo é feito por `python experimentos/final_member.py --seed <N>`, porém com duração de várias horas por semente em CPU ou MPS.
2. **Ensemble, McNemar e entrega.** Com os seis arquivos de probabilidades na raiz do projeto, execute `python experimentos/build_final.py`. Ele treina o ComplementNB em todo o `train.xlsx`, combina os quatro membros por soft voting, avalia no teste1, reporta a estatística de McNemar e grava `test2_ep2.xlsx`.
3. **Análises da seção 9.** `python experimentos/analise.py` reproduz a interpretação global, a interpretação local e a análise de erros.

Scripts auxiliares usados no desenvolvimento, não necessários para reproduzir o resultado final: `main.py` e `baseline.py` (modelos clássicos e baseline oficial do EP1), `experimentos/ensemble_bert.py` (ensemble das sementes no holdout), `experimentos/train_encoder.py` (fine-tuning de encoders alternativos), `experimentos/train_coral.py` (abordagem ordinal CORAL), `experimentos/colab-grid-bert.ipynb` (busca em grade do BERT) e `experimentos/colab-bertimbau-large.ipynb` (BERTimbau-large).
