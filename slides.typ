#import "@preview/touying:0.6.1": *
#import themes.metropolis: *

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [EP2: Classificação de clareza no e-SIC],
    subtitle: [Ensemble calibrado e limites empíricos da tarefa],
    author: [Cleben Garcia, João Pedro Marcondes, Luis Moraes, Renan Guimarães],
    date: [Novembro 2026],
    institution: [ACH2118 | EACH-USP],
  ),
  config-colors(
    primary: rgb("#b85437"),          // Terracota suave
    secondary: rgb("#2d2621"),        // Café / grafite quente
    tertiary: rgb("#c07d32"),         // Ocre queimado
    neutral-lightest: rgb("#fdfcf9"),   // Papel marfim / linho
    neutral-dark: rgb("#2d2621"),
    neutral-darkest: rgb("#221c18"),
  ),
)

#set text(font: "Liberation Sans", size: 13.5pt)

// Componentes visuais com paleta quente e editorial
#let card(title: none, body, fill: rgb("#f7f4ee"), stroke: 0.7pt + rgb("#ded8cb"), title-color: rgb("#2d2621")) = {
  block(
    fill: fill,
    stroke: stroke,
    inset: (x: 9pt, y: 7pt),
    radius: 5pt,
    width: 100%,
    [
      #if title != none [
        #text(weight: "bold", fill: title-color, size: 0.95em)[#title]
        #v(2pt)
      ]
      #body
    ]
  )
}

#let metric-badge(val, label, fill: rgb("#f4f0e8"), stroke: 0.7pt + rgb("#ded6c6"), text-color: rgb("#38302a")) = {
  align(center)[
    #block(
      fill: fill,
      stroke: stroke,
      inset: (x: 8pt, y: 5pt),
      radius: 5pt,
      width: 100%,
      [
        #text(weight: "bold", size: 1.35em, fill: text-color)[#val] \
        #text(size: 0.72em, fill: rgb("#61574d"))[#label]
      ]
    )
  ]
}

#title-slide()

== Do EP1 ao EP2: lições e hipótese de trabalho

#grid(
  columns: (1.2fr, 0.8fr),
  gutter: 12pt,
  [
    #card(title: [Diagnóstico do modelo do EP1])[
      - Modelo anterior: fine-tuning de um único BERTimbau-base
      - Acurácia no teste 1: 0,4867 (F1-macro: 0,4839)
      - Limitações observadas:
        - Concentração excessiva de predições na classe majoritária (c5)
        - Recall baixo na classe intermediária c234 (38%)
        - 86 casos em que c1 foi confundido com c5 no teste 1
        - Uma única arquitetura atinge saturação precoce nesse vocabulário
    ]
    #v(4pt)
    #card(title: [Hipótese para o EP2], fill: rgb("#fbf5f2"), stroke: 0.7pt + rgb("#ebd9ce"), title-color: rgb("#a3432a"))[
      Combinar representações com padrões de erro descorrelacionados (BERTimbau para contexto semântico e ComplementNB para frequências de termos), corrigindo a distribuição a priori de probabilidades antes da decisão final.
    ]
  ],
  [
    #metric-badge("0,4867", [BERTimbau (EP1)])
    #v(4pt)
    #metric-badge("0,5078", [Ensemble calibrado (EP2)], fill: rgb("#f2f5f1"), stroke: 0.7pt + rgb("#d4ded1"), text-color: rgb("#2c5737"))
    #v(4pt)
    #align(center)[
      #block(
        fill: rgb("#f5f1e9"),
        inset: 7pt,
        radius: 5pt,
        width: 100%,
        [
          #text(size: 0.85em, weight: "bold", fill: rgb("#453c35"))[Evolução no teste 1:] \
          #text(size: 1.15em, weight: "bold", fill: rgb("#2c5737"))[+2,11 p.p.] \
          #text(size: 0.72em, fill: rgb("#695f54"))[Saldo de 19 acertos líquidos a mais]
        ]
      )
    ]
  ]
)

== Tentativas descartadas e diagnóstico

#text(size: 0.85em)[Resultados das abordagens avaliadas no holdout antes da escolha final:]

#v(2pt)

#table(
  columns: (1.2fr, 2.2fr, 1.6fr),
  stroke: (x, y) => if y == 0 { (bottom: 1.5pt + rgb("#2d2621")) } else { 0.5pt + rgb("#e8e2d5") },
  fill: (x, y) => if y == 0 { rgb("#f2eee6") } else if calc.even(y) { rgb("#f9f7f2") } else { white },
  inset: 5.5pt,
  table.header([*Família*], [*Técnica investigada*], [*Resultado no holdout*]),
  [Pré-processamento], [Remoção de saudações e assinaturas, minúsculas, truncagem], [Acurácia caiu de 0,4565 para 0,4445. O cabeçalho identifica o órgão emissor.],
  [Modelos ordinais], [CORAL ordinal (projeções binárias com limiares ordenados)], [A rede convergiu para nunca prever a classe intermediária c234.],
  [Regras de decisão], [Cascata de 2 estágios (separar neutro versus extremos)], [AUC para neutro foi de 0,567. Apenas redistribui erros.],
  [Limpeza de dados], [Deduplicação e descarte de textos com rótulos conflitantes], [Acurácia entre 0,4412 e 0,4429, sem ganho prático.],
  [Modelos maiores], [BERTimbau-large (0,4486) e mDeBERTa-v3-base], [Não superaram o modelo base e aumentaram o custo de treino.],
)

#v(3pt)
#card(fill: rgb("#fbf6ee"), stroke: 0.7pt + rgb("#ebdcc3"), title-color: rgb("#8c601b"))[
  Diagnóstico: regras de decisão e ajustes de limiar apenas redistribuem a incerteza existente. O ganho prático veio da combinação de representações distintas.
]

== Estrutura do ensemble híbrido

#align(center)[
  #block(
    fill: rgb("#f8f5ee"),
    stroke: 1.2pt + rgb("#3d332a"),
    inset: 8pt,
    radius: 6pt,
    width: 98%,
    [
      #grid(
        columns: (1fr, auto, 1fr),
        gutter: 10pt,
        align: horizon,
        [
          #block(fill: rgb("#f2f5f7"), stroke: 0.7pt + rgb("#cbd7de"), inset: 6pt, radius: 5pt)[
            *Ramo neural (contexto)* \
            #text(size: 0.78em)[
              3 sementes do BERTimbau-base \
              (sementes 42, 1, 2; tamanho máx. 512; lr 2e-5) \
              $arrow.r$ Média das probabilidades: $P_("BERT")$
            ]
          ]
        ],
        [
          #text(weight: "bold", size: 1.3em, fill: rgb("#3d332a"))[+]
        ],
        [
          #block(fill: rgb("#fbf4ec"), stroke: 0.7pt + rgb("#e8d5bf"), inset: 6pt, radius: 5pt)[
            *Ramo linear (termos)* \
            #text(size: 0.78em)[
              TF-IDF (1 e 2-gramas; min_df 2; sublinear) \
              ComplementNB ($alpha=0.3$) \
              #text(fill: rgb("#9c4826"), weight: "bold")[Calibração de prior ($w$)] $arrow.r P_("CNB")^("calib")$
            ]
          ]
        ]
      )
      #v(4pt)
      #line(length: 60%, stroke: 0.8pt + rgb("#d8d0c2"))
      #v(3pt)
      #text(weight: "bold", size: 0.98em, fill: rgb("#2d2621"))[
        Votação suave ponderada: \
        $P_("final") = 0,30 dot P_("BERT") + 0,70 dot P_("CNB")^("calib")$
      ]
    ]
  )
]

#v(4pt)
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #card(title: [Diversidade de representação])[
      - O BERTimbau capta relações sintáticas e semânticas globais
      - O ComplementNB reage à presença de palavras específicas de cada domínio
      - Os dois ramos apresentam correlação baixa de erros
    ]
  ],
  [
    #card(title: [Fusão calibrada])[
      - Reduz a variância inerente a sementes neurais individuais
      - Estabiliza predições nas fronteiras de decisão
      - Pesos validados pelo protocolo em metades no holdout
    ]
  ]
)

== Calibração de prior no ComplementNB

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  [
    #card(title: [Distorção empírica do estimador])[
      - O ComplementNB estima parâmetros usando o complemento das classes
      - Na tarefa do e-SIC, esse cálculo gerou uma distorção:
        - Previu c5 em 46% dos casos no conjunto de teste
        - Reduziu fortemente a revocação da classe central c234
    ]
    #v(4pt)
    #card(title: [Ajuste multiplicativo de prior], fill: rgb("#f2f5f1"), stroke: 0.7pt + rgb("#d4ded1"), title-color: rgb("#2c5737"))[
      Multiplicação pontual pelo vetor de calibração $w$:
      $ P_("CNB")^("calib")(c) = frac(w_c dot P_("CNB")(c), sum_j w_j dot P_("CNB")(j)) $
      com $w = [1,2, quad 1,1, quad 0,7]$ para `{c1, c234, c5}`.
    ]
  ],
  [
    #card(title: [Efeito isolado no teste 1], fill: rgb("#f7f4ee"))[
      #align(center)[
        #table(
          columns: (1.4fr, 1fr),
          stroke: 0.5pt + rgb("#ded8cb"),
          inset: 6pt,
          table.header([*Versão do ComplementNB*], [*Acurácia*]),
          [Sem calibração], [0,4711],
          [*Com calibração de prior*], [*0,5000*],
          [Saldo], [#text(fill: rgb("#2c5737"), weight: "bold")[+26 acertos]]
        )
      ]
      #v(2pt)
      - Conteve a superestimação da classe c5
      - Elevou o número de acertos em c234 de 109 para 120
      - Superou o modelo do EP1 de forma isolada
    ]
  ]
)

== Validação em metades do holdout

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #card(title: [Prevenção contra sobreajuste nos pesos])[
      - Avaliar dezenas de combinações de pesos no holdout inteiro pode gerar estimativas otimistas
      - O holdout estratificado de 20% (4.019 instâncias) foi dividido em duas metades independentes (A e B)
      - Pesos ajustados na metade A e medidos na metade B: 0,4818
      - Pesos ajustados na metade B e medidos na metade A: 0,4701
      - Ambas as metades selecionaram a mesma composição (0,30 BERT e 0,70 CNB)
    ]
  ],
  [
    #card(title: [Treinamento para a submissão])[
      - Validação do BERT: 10% do treino serviu exclusivamente para early stopping
      - Modelo final: retreinado nos 20.092 exemplos de treino
      - Teste 1: avaliado uma única vez para o teste de McNemar
      - Teste 2: aplicado diretamente para produzir a planilha oficial de 897 instâncias, sem ajustes posteriores
    ]
  ]
)

== Resultados no teste 1 (900 exemplos)

#align(center)[
  #table(
    columns: (2.3fr, 1.1fr, 1.1fr, 1.5fr),
    stroke: (x, y) => if y == 0 { (bottom: 2pt + rgb("#2d2621")) } else { 0.5pt + rgb("#e8e2d5") },
    fill: (x, y) => if y == 0 { rgb("#f2eee6") } else if y == 6 { rgb("#f2f5f1") } else if calc.even(y) { rgb("#f9f7f2") } else { white },
    inset: 6.5pt,
    table.header([*Abordagem avaliada*], [*Acurácia*], [*F1-macro*], [*Ganho frente ao baseline*]),
    [Classe majoritária (c5)], [0,3333], [–], [-13,34 p.p.],
    [Baseline oficial (RegLog + TF-IDF)], [0,4667], [–], [Referência],
    [ComplementNB sem calibração], [0,4711], [0,4621], [+0,44 p.p.],
    [BERTimbau único (EP1)], [0,4867], [0,4839], [+2,00 p.p.],
    [ComplementNB com calibração], [0,5000], [0,4938], [+3,33 p.p.],
    [*Ensemble calibrado (EP2)*], [*0,5078*], [*0,5056*], [#text(fill: rgb("#2c5737"), weight: "bold")[*+4,11 p.p.*]],
  )
]

#v(4pt)
#grid(
  columns: (1fr, 1fr, 1fr),
  gutter: 10pt,
  metric-badge("+4,11 p.p.", [Acima do baseline oficial]),
  metric-badge("+2,11 p.p.", [Acima do modelo do EP1]),
  metric-badge("0,5056", [F1-macro equilibrado], fill: rgb("#fbf6ee"), stroke: 0.7pt + rgb("#ebdcc3"), text-color: rgb("#785213"))
)

== Comparação estatística com o EP1

#grid(
  columns: (1fr, 1.1fr),
  gutter: 12pt,
  [
    #card(title: [Tabela de contingência 2x2 (teste 1)])[
      #align(center)[
        #table(
          columns: (1.2fr, 1fr, 1fr),
          stroke: 0.5pt + rgb("#ded8cb"),
          inset: 6pt,
          table.header([], [*EP2 certo*], [*EP2 errado*]),
          [*EP1 certo*], [389], [49],
          [*EP1 errado*], [*68*], [394]
        )
      ]
      #v(2pt)
      - 117 casos discordantes no total:
        - 68 em que apenas o EP2 acertou
        - 49 em que apenas o EP1 acertou
      - Saldo líquido de 19 acertos a favor do EP2
    ]
  ],
  [
    #card(title: [Teste de McNemar], fill: rgb("#f7f4ee"))[
      - Estatística com correção de continuidade:
        $chi^2 = 2,7692$ (1 gl), com $p = 0,0961$
      - Teste binomial exato: $p = 0,0957$
      - Significância marginal a 10%
      
      #v(2pt)
      #text(size: 0.82em, fill: rgb("#61574d"))[
        Antes da calibração de prior, o p-valor era de 0,2765. O ajuste reduziu o valor para menos de 0,10.
      ]
    ]
    #v(4pt)
    #card(title: [Queda em erros opostos], fill: rgb("#f2f5f1"), stroke: 0.7pt + rgb("#d4ded1"), title-color: rgb("#2c5737"))[
      Erros que trocavam c1 e c5 diretamente caíram de 86 para 53 casos (redução de 38%).
    ]
  ]
)

== Pesos globais e identificadores de órgão

#text(size: 0.85em)[Termos mais influentes na regressão logística sobre TF-IDF (1 e 2-gramas):]

#table(
  columns: (0.7fr, 3.3fr),
  stroke: 0.5pt + rgb("#ded8cb"),
  inset: 5.5pt,
  fill: (x, y) => if calc.even(y) { rgb("#f9f7f2") } else { white },
  table.header([*Classe*], [*Termos de maior peso positivo*]),
  [`c1`], [luis (+1,29), demanda (+1,28), solicitação (+1,27), essa informação (+1,12), sr luis (+1,06), nup (+0,96)],
  [`c234`], [sistemas (+1,40), prezado em (+1,21), territorial (+1,01), projetos (+0,96), extensão (+0,96), reitoria (+0,94)],
  [`c5`], [dnpm (+2,07), questionário (+1,89), sic dnpm (+1,61), solicitação encaminhamos (+1,55), odila (+1,19)],
)

#v(4pt)
#card(title: [O classificador aprendeu procedência institucional], fill: rgb("#fbf5f2"), stroke: 0.7pt + rgb("#ebd9ce"), title-color: rgb("#a3432a"))[
  - Os termos de maior peso são siglas de órgãos (DNPM = Departamento Nacional de Produção Mineral), domínios temáticos (reitoria, concurso, territorial) e nomes de solicitantes ou servidores (luis, odila).
  - O classificador utiliza a procedência como indicador indireto: órgãos públicos apresentam qualidades habituais distintas em suas respostas.
  - Isso explica por que retirar saudações e rodapés prejudicou os resultados: o cabeçalho é justamente onde a identidade do órgão está registrada.
]

== Interpretação local e análise de erros

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #card(title: [Comportamento local das predições])[
      - As contribuições individuais de palavras são baixas (valores abaixo de 0,13) e espalhadas por termos funcionais (de, que, uma)
      - A decisão não se concentra em palavras-chave específicas de clareza
      - Confiança média do modelo: 0,534 nos acertos e 0,486 nos erros
      - O tamanho do texto não discrimina acertos e erros: média de 952 caracteres nos acertos e 950 nos erros
    ]
  ],
  [
    #card(title: [Casos concretos do teste 1])[
      #text(size: 0.8em)[
        - *Linha 6 (c5 previsto como c234):* resposta com link direto da UFF atendendo à solicitação. O cidadão deu nota máxima, mas o modelo hesitou por fragmentos da URL (uff br, pdf).
        - *Linha 0 (c5 previsto como c234):* explicação detalhada do IBGE sobre o Censo. O modelo reagiu ao tema censitário.
        - *Linha 1 (c1 previsto como c234):* resposta protocolar do Ministério do Planejamento. O texto é gramaticalmente correto, mas não resolveu a demanda.
      ]
    ]
  ]
)

#v(4pt)
#card(fill: rgb("#fbf6ee"), stroke: 0.7pt + rgb("#ebdcc3"), title-color: rgb("#8c601b"))[
  Sem a pergunta original e sem saber se a solicitação foi atendida, o texto da resposta sozinho não traz elementos para separar a nota 1 da nota 5.
]

== Submissão do teste 2 e conclusões

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  [
    #card(title: [Arquivo entregue: test2_ep2.xlsx])[
      - 897 instâncias rotuladas sem valores ausentes
      - Distribuição das predições geradas:
        - c1: 36,0% (323 instâncias)
        - c234: 23,5% (211 instâncias)
        - c5: 40,5% (363 instâncias)
      - A calibração de prior evitou o esvaziamento da classe c234 e conteve a concentração em c5
    ]
    #v(4pt)
    #card(title: [Código e reprodutibilidade], fill: rgb("#f7f4ee"))[
      - Scripts e notebooks disponíveis em: \
        #text(fill: rgb("#a3432a"))[`github.com/clebenjr/ep1-ipln`]
    ]
  ],
  [
    #card(title: [Conclusões do estudo], fill: rgb("#f2f5f1"), stroke: 0.7pt + rgb("#d4ded1"), title-color: rgb("#2c5737"))[
      1. A clareza de uma resposta depende da pergunta feita e da expectativa do cidadão. Classificar apenas a resposta parte de uma entrada incompleta.
      2. O teto de desempenho decorre do ruído humano na escala subjetiva (respostas com texto idêntico no treino receberam notas diferentes).
      3. O ganho de 0,4667 para 0,5078 derivou da união de paradigmas complementares (BERT e n-gramas) com ajuste estatístico de prior.
    ]
  ]
)

== Perguntas e encerramento

#align(center + horizon)[
  #text(weight: "bold", size: 1.8em, fill: rgb("#2d2621"))[Obrigado!]
  
  #v(10pt)
  #text(size: 1.05em, fill: rgb("#4a3e35"))[
    *EP2: Classificação de clareza no e-SIC* \
    ACH2118 - Introdução ao Processamento de Língua Natural \
    EACH - Universidade de São Paulo (USP)
  ]
  
  #v(12pt)
  #block(
    fill: rgb("#f7f4ee"),
    stroke: 0.8pt + rgb("#ded8cb"),
    inset: (x: 18pt, y: 10pt),
    radius: 7pt,
    [
      #text(size: 0.88em)[
        *Integrantes:* \
        Cleben Junior Cosendey Garcia (14691220) \
        João Pedro Oliveira de Paula Marcondes (14582570) \
        Luis Henrique Moraes (14615590) \
        Renan Bezerra Guimarães (14605640)
      ]
      #v(5pt)
      #text(size: 0.82em, fill: rgb("#a3432a"))[
        Repositório: `https://github.com/clebenjr/ep1-ipln`
      ]
    ]
  )
]
