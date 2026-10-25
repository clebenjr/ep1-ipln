# EP1 - Clareza da resposta em sistemas e-SIC

ACH2118 - Introdução ao Processamento de Língua Natural

## Integrantes

- Cleben Junior Cosendey Garcia (14691220)
- João Pedro Oliveira de Paula Marcondes (14582570)
- Luis Henrique Moraes (14615590)
- Renan Bezerra Guimarães (14605640)

## Breve descrição

O objetivo é classificar a clareza de uma resposta do e-SIC em uma de três categorias {c1, c234, c5}, usando como entrada apenas o texto da resposta resp_text. É um problema pequeno e com balanceamento razoável (c1 31,6%, c234 34,1% e c5 34,3%); a classe mais frequente é a c5.

Foram testadas duas ideias para o mesmo problema. A primeira foi o caminho tradicional: representar o texto por contagens com TF-IDF (tanto por palavras, em uni e bigramas, quanto por n-gramas de caracteres) e inserir em classificadores clássicos como Regressão Logística, LinearSVC, Complement Naive Bayes e SGD. A segunda foi partir para uma representação contextual, fazendo fine-tuning do BERTimbau (neuralmind/bert-base-portuguese-cased), que entende cada texto a partir de embeddings em vez de simples contagem de termos.

Na prática, todos os modelos de TF-IDF atingiram em volta de 0,45 de acurácia e nenhum conseguiu passar do baseline de forma consistente, o que já era um sinal de que a representação por contagem tinha chegado no teto desse problema. Uma alternativa foi utilizar um modelo contextual. Por isso, o modelo final escolhido para a tarefa é o BERTimbau com fine-tuning, que superou o baseline no conjunto de teste separado, ainda que por margem estreita.

## Pré-processamento

Optou-se por alterar o texto o mínimo possível, uma vez que as respostas são curtas e uma limpeza mais agressiva tende a descartar informação útil.

De modo geral, os textos vazios foram preenchidos com string vazia e todos os valores foram tratados como texto; nos rótulos, os espaços excedentes foram removidos e verificou-se que todos pertencem a {c1, c234, c5} (caso surja algum valor diferente, a execução é interrompida).

Nos modelos de TF-IDF, a tokenização, o lowercase e a vetorização foram mantidos dentro do próprio Pipeline do scikit-learn. Essa decisão é relevante porque garante que tudo seja recalculado dentro de cada fold da validação cruzada, evitando que o vocabulário do teste vaze para o treino. Não foram removidas stopwords nem aplicado stemming; utilizou-se sublinear_tf para suavizar a frequência dos termos e min_df para descartar termos raros demais.

No BERTimbau, foi utilizado o próprio tokenizador do modelo (WordPiece), com truncamento, padding e max_length=512. Nesse caso não há motivo para normalizar o texto manualmente, pois o modelo é cased, ou seja, aproveita maiúsculas, acentuação e pontuação como pistas.

## Parâmetros

A seguir estão os parâmetros ajustados na busca pela melhor configuração, separados por família de modelo.

### Modelos TF-IDF (vetorização + classificador)

| Parâmetro | Valores considerados |
|---|---|
| analyzer | {word, char_wb} |
| ngram_range (palavras) | {(1,1), (1,2)} |
| ngram_range (caracteres) | {(3,5)} |
| sublinear_tf | {True} |
| min_df | {2, 3} |
| max_df | {0.95} |
| C (LogisticRegression) | {1.0, 4.0} |
| C (LinearSVC) | {0.3, 0.5} |
| alpha (ComplementNB) | {0.3} |
| alpha (SGDClassifier) | {1e-5} |

### BERTimbau (fine-tuning)

| Parâmetro | Valores considerados |
|---|---|
| modelo base | neuralmind/bert-base-portuguese-cased |
| num_train_epochs | {3} |
| per_device_train_batch_size | {8} |
| max_length | {512} |
| warmup_steps | {500} |
| weight_decay | {0.01} |
| learning_rate | 5e-5 (padrão do Trainer) |

## Valores Ótimos

Configuração adotada no modelo final de cada família.

| Modelo | Parâmetro | Valor |
|---|---|---|
| Baseline (TF-IDF + RegLog) | ngram_range | (1,1) |
| | lowercase | True |
| | C | 1.0 |
| BERTimbau (final) | num_train_epochs | 3 |
| | per_device_train_batch_size | 8 |
| | max_length | 512 |
| | warmup_steps | 500 |
| | weight_decay | 0.01 |
| | learning_rate | 5e-5 |

## Procedimento

O conjunto train.xlsx (20.092 exemplos) foi separado de forma estratificada: 80% para desenvolvimento e 20% (4.019 exemplos) reservados como holdout, com random_state=42. Essa escolha foi proposital, pois o notebook do BERTimbau utiliza exatamente a mesma semente e a mesma estratificação, de modo que o baseline e o BERT são comparados sobre os mesmos 4.019 exemplos de teste.

A escolha entre os modelos tradicionais foi feita por validação cruzada estratificada de 5 folds sobre a parte de desenvolvimento, sempre com a vetorização dentro de cada fold para evitar vazamento, e o ranking foi definido pela acurácia média. Após a seleção, o modelo final (e o baseline oficial) foi re-treinado na parte de desenvolvimento e avaliado uma única vez no holdout, o que fornece uma estimativa mais honesta do desempenho real.

Por fim, para rotular o test1.xlsx, o modelo final (BERTimbau) foi re-treinado em todo o train.xlsx, reservando 5% apenas para o early stopping. A planilha de saída (test1_bert.xlsx) mantém o mesmo número de linhas (900) e as mesmas colunas do arquivo original, com os rótulos restritos a {c1, c234, c5}.

## Resultado

A métrica utilizada é a acurácia no conjunto de teste (o holdout de 4.019 exemplos, o mesmo para todos os modelos). A tabela apresenta apenas os modelos finais.

Como o problema é ordinal (c1 < c234 < c5, de menos clara a mais clara), reportam-se também duas métricas secundárias que levam a ordem em conta: o kappa quadrático ponderado (QWK) e o erro absoluto médio em passos ordinais (MAE), ambos sobre o holdout. Elas penalizam menos o erro entre classes adjacentes do que o erro entre os extremos.

O BERTimbau foi treinado em 3 sementes (média ± desvio-padrão no holdout); as demais linhas são de uma execução única.

| Tarefa | Modelo final | Acurácia (teste) | F1-macro | QWK | MAE (passos) |
|---|---|---|---|---|---|
| Clareza {c1, c234, c5} | **BERTimbau (fine-tuning)** | **0,4500 ± 0,0035** | 0,4497 | 0,30 | 0,67 |
| Clareza (referência) | Baseline oficial (TF-IDF + RegLog) | 0,4441 | 0,4423 | 0,25 | 0,71 |
| Clareza (referência) | Classe majoritária | 0,3430 | 0,1703 | – | – |

O BERTimbau ficou cerca de 0,006 acima do baseline oficial no mesmo holdout, e ambos superam a classe majoritária (0,3430). A vantagem é pequena — da ordem de um a dois desvios-padrão — mas consistente: as três sementes (0,4464, 0,4489 e 0,4548) ficaram todas acima do baseline. O valor absoluto baixo (cerca de 0,45) é característico da tarefa. A classe intermediária c234 é a de menor recall (0,38 no BERTimbau, contra 0,46 em c1 e 0,51 em c5) e a maior fonte isolada de erro, coerente com o fato de ela agregar notas heterogêneas (2, 3 e 4) em uma única classe. Vale notar que o modelo capta parte da ordem das classes — os extremos c1 e c5 se confundem menos do que o acaso previa (22% dos erros, contra 33% esperados ao acaso) — mas a concordância ordinal é baixa (QWK em torno de 0,30), refletindo o sinal fraco disponível no texto.

Como referência adicional, mediu-se a concordância de um modelo externo não treinado na base (JEV, da TypeSafe) com o rótulo humano, em few-shot (3 exemplos por classe, sorteados apenas do conjunto de desenvolvimento), no mesmo holdout. Ele atingiu 0,3802, abaixo dos modelos supervisionados. Esse número não deve ser lido como o teto da tarefa: um modelo que não viu a base não recupera a convenção específica de rotulagem (ele concentra as predições na c234, com recall de apenas 16% em c1), de forma que o resultado mede desalinhamento de convenção, não ruído irredutível.

## Conclusão

O desempenho dos modelos supervisionados estaciona em torno de 0,45, e há evidência concreta de que isso reflete um teto da própria tarefa, e não uma falha de modelagem. Cerca de 10,6% das respostas aparecem com texto exatamente duplicado no treino, e nessas linhas um oráculo que atribui a cada texto o seu rótulo mais frequente ainda assim só alcança 63% de acerto: o mesmo texto recebe rótulos humanos diferentes com frequência (291 textos distintos têm rótulos conflitantes). O rótulo carrega, portanto, ruído irredutível, consequência de a clareza ter sido julgada subjetivamente pelos próprios cidadãos numa escala 1..5. A isso soma-se o desenho do rótulo: a classe c234 agrega as notas intermediárias 2, 3 e 4, tornando-se heterogênea e a mais difícil de prever. O modelo final entregue (BERTimbau) supera o baseline sem overfitting relevante, dentro desse teto; ganhos mais substanciais provavelmente exigiriam atacar o dado (trazer a pergunta original junto da resposta, ou tratar o problema como ordinal) mais do que trocar de modelo.

## Link para repositório do código

https://github.com/clebenjr/ep1-ipln

## Instruções

As dependências são Python 3.12 com pandas, openpyxl e scikit-learn para os modelos tradicionais; para o modelo final também são necessários torch, transformers e accelerate, responsáveis pelo fine-tuning do BERTimbau (preferencialmente com GPU).

Para executar os modelos tradicionais e o baseline (apenas scikit-learn):

```bash
pip install pandas openpyxl scikit-learn

python baseline.py          # baseline oficial + classe majoritária, rotula test1.xlsx
python main.py --list       # lista os modelos registrados
python main.py              # avalia todos (CV no dev + holdout) e salva o ranking em results/
python main.py --predict    # re-treina o melhor em todo o treino e rotula o teste
```

Para o modelo final (BERTimbau), abra o experimentos/experimento-codigo-bert.ipynb em um ambiente com GPU (o Google Colab atende bem), com o train.xlsx no diretório de trabalho, e execute as células na ordem. A primeira delas já instala transformers, accelerate e openpyxl. O notebook separa o mesmo holdout de 20% dos modelos finais (random_state=42) e o mantém intocado; a validação para early stopping é tirada do conjunto de desenvolvimento, não do holdout. O fine-tuning (learning rate 2e-5, early stopping pelo F1-macro da validação) é repetido em múltiplas sementes, e o notebook reporta média e desvio-padrão da acurácia e do F1-macro no holdout, além do relatório de classificação e da matriz de confusão.

Para reproduzir a planilha de entrega, rode `python experimentos/predict_test_bert.py` (com train.xlsx e test1.xlsx no diretório, e torch/transformers/accelerate instalados). Ele treina o BERTimbau em todo o treino (5% reservado para early stopping) e grava test1_bert.xlsx com os rótulos previstos, no mesmo formato do arquivo de teste original.
