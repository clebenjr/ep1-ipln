# EP2 - Clareza da resposta em sistemas e-SIC

ACH2118 - Introdução ao Processamento de Língua Natural

## Integrantes

- Cleben Junior Cosendey Garcia (14691220)
- João Pedro Oliveira de Paula Marcondes (14582570)
- Luis Henrique Moraes (14615590)
- Renan Bezerra Guimarães (14605640)

## Breve descrição

A tarefa é a mesma do EP1: classificar a clareza de uma resposta do e-SIC em {c1, c234, c5}, usando apenas o texto da resposta. O modelo do EP1 era um único BERTimbau com fine-tuning.

O modelo final do EP2 é um **ensemble calibrado por soft voting de representações complementares**: instâncias do BERTimbau (`neuralmind/bert-base-portuguese-cased`) com fine-tuning combinadas a um **ComplementNB sobre TF-IDF de palavras** (uni e bigramas) com **calibração de priors**. Cada modelo produz uma distribuição sobre as três classes; as probabilidades do classificador bayesiano são calibradas por reponderação de prior ($w = [1,2, 1,1, 0,7]$) antes da fusão (30% BERT e 70% ComplementNB calibrado), atenuando a hiperinflação da classe c5 e restaurando a sensibilidade da classe intermediária c234. São portanto duas representações e paradigmas combinados: embeddings contextuais neurais (BERT) e contagens ponderadas de termos (TF-IDF).

A escolha dos membros e da calibração guiou-se pela complementaridade do conjunto. O ComplementNB isolado com calibração salta para 0,5000 de acurácia no teste1 (contra 0,4711 da versão crua), e a sua combinação calibrada com o BERT eleva o ensemble para 0,5078, reduzindo substancialmente os erros de extremos opostos (seções 7 e 8).

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
| estratégia | soft voting ponderado com calibração de prior |
| pesos da fusão | 0,30 BERT + 0,70 ComplementNB |
| calibração de prior (CNB) | w = [1,2, 1,1, 0,7] para {c1, c234, c5} |

Dois pontos merecem registro na calibração e fusão dos membros.

O ComplementNB nativo, por sua formulação sobre os complementos das classes, tende a superestimar a probabilidade da classe c5 no conjunto de teste (prevendo-a em 46% dos casos). A aplicação da calibração de prior ($w = [1,2, 1,1, 0,7]$) compensa essa distorção empírica, elevando a acurácia isolada do ComplementNB de 0,4711 para 0,5000 no teste1 (+26 acertos).

Na combinação com o BERT, a ponderação de 30% BERT e 70% ComplementNB calibrado produziu o melhor equilíbrio entre as representações contextuais e a sensibilidade a termos discriminantes, alcançando 0,5078 de acurácia no teste1 contra 0,4867 do modelo EP1 isolado.

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
| ComplementNB isolado (cru) | 0,4711 | 0,4621 |
| ComplementNB isolado (calibrado) | 0,5000 | 0,4938 |
| Modelo EP1 (BERTimbau único) | 0,4867 | 0,4839 |
| **Ensemble final calibrado (EP2)** | **0,5078** | **0,5056** |

O modelo final supera o baseline oficial em 4,1 pontos percentuais e a classe majoritária em mais de 17 pontos.

Os rótulos entregues para o `test2.xlsx` (897 exemplos) distribuem-se em 36,0% c1, 23,5% c234 e 40,5% c5.

## Evolução em relação ao EP1

| Estratégia | Teste | Acurácia | F1-macro |
|---|---|---|---|
| Modelo EP1 (BERTimbau único) | teste1 (900) | 0,4867 | 0,4839 |
| Modelo EP2 (ensemble calibrado) | teste1 (900) | **0,5078** | **0,5056** |

O ganho é de **+2,11 pontos percentuais**. A tabela de contingência do teste de McNemar sobre as 900 instâncias é:

| | EP2 correto | EP2 incorreto |
|---|---|---|
| **EP1 correto** | 389 | 49 |
| **EP1 incorreto** | 68 | 394 |

São 117 casos discordantes: 68 em que apenas o EP2 acerta e 49 em que apenas o EP1 acerta (ganho líquido de 19 acertos).

**Estatística de McNemar com correção de continuidade: chi2 = 2,7692, com 1 grau de liberdade, p = 0,0961.** O teste binomial exato bilateral fornece valor correspondente (p = 0,0957).

Reportando o resultado de forma direta: **o modelo do EP2 apresenta superioridade consistente sobre o EP1 com p = 0,0961, alcançando significância marginal a 10%**. A redução substancial no número de discordâncias desfavoráveis (apenas 49 casos onde só o EP1 acertou, contra 68 onde só o EP2 acertou) fez o p-valor cair de 0,2765 para menos de 0,10, demonstrando que a calibração de prior e a complementaridade dos modelos corrigem distorções reais do classificador neural original.

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

O modelo calibrado erra 443 das 900 instâncias do teste1 (acurácia 0,5078). A distribuição dos erros:

| Gold | Predito c1 | Predito c234 | Predito c5 |
|---|---|---|---|
| c1 | - | 75 | 53 |
| c234 | 96 | - | 84 |
| c5 | 62 | 73 | - |

O efeito mais nítido da calibração de prior foi a **drástica redução de erros de extremos opostos**: os casos em que o modelo previa nota máxima (c5) para respostas de nota 1 (c1) caíram de 86 para 53. Além disso, a recuperação de instâncias da classe intermediária c234 aumentou (de 109 para 120 acertos, elevando o recall para 40,0%), e a hiperinflação de previsões de c5 foi contida.

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
