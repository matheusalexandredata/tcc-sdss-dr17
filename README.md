# Classificação de objetos astronômicos no SDSS DR17

Experimento do trabalho de graduação de Matheus Castro Alexandre, do curso de Ciência de Dados da Fatec Jundiaí, orientado pelo Prof. Me. Ricardo Leardini Lobo.

A comparação usa regressão logística multinomial, árvore de decisão, Random Forest e MLP para classificar estrelas, galáxias e quasares. São avaliadas quatro cores fotométricas com e sem o redshift espectroscópico.

## Como executar

Com Python 3.12.14, abra um terminal na pasta do repositório e execute:

```bash
python -m pip install -r requirements.txt
python main.py
```

O script baixa a [versão 1 do conjunto de dados no Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/versions/1). Se o download automático falhar, crie a pasta `dados` e salve o ZIP baixado do Kaggle como `dados/sdss17.zip`. O script confere o conteúdo do CSV antes de treinar.

## O que o script faz

1. Lê o recorte do [SDSS DR17](https://www.sdss4.org/dr17/), remove a linha com magnitude inválida e mantém o primeiro registro da entrada fotométrica repetida. Restam 99.998 registros.
2. Calcula u−g, g−r, r−i e i−z. Separa 30.000 registros para teste. Dos 69.998 de treino, usa 55.998 para ajuste e 14.000 para validação. As duas divisões são estratificadas e usam a seed 42.
3. Seleciona as configurações pelo F1 macro na validação e treina novamente com o conjunto completo de treino. Na análise sem redshift, repete o treinamento com a configuração mantida e com uma nova seleção.
4. Avalia os modelos nas mesmas 30.000 observações. Grava métricas, predições, partições, histórico da MLP e matrizes de confusão em `resultados/`.

## Cenários de teste

As quatro cores de entrada são u−g, g−r, r−i e i−z. O script avalia os quatro modelos nos mesmos 30.000 registros de teste em três cenários.

| Cenário | Entradas | Configuração usada |
| --- | --- | --- |
| Com redshift | Quatro cores e redshift espectroscópico | Selecionada na validação com as cinco entradas |
| Só cores, configuração mantida | Quatro cores | Hiperparâmetros escolhidos com redshift. A MLP mantém também o número de épocas |
| Só cores, nova seleção | Quatro cores | Selecionada novamente na validação usando só as cores |

Há também um controle para a MLP com redshift. Ele mantém a configuração e o número de épocas da MLP selecionada e usa padronização direta, sem a transformação asinh.

## Opções testadas

| Modelo | Alternativas avaliadas na validação | Com redshift | Só cores |
| --- | --- | ---: | ---: |
| Regressão logística multinomial | Regularização C de 0,1, 1 e 10. Padronização, método L-BFGS e limite de 3.000 iterações em todas as opções | 3 | 3 |
| Árvore de decisão | Profundidade máxima de 2, 3, 4, 5, 8 ou 12, além de uma opção sem limite de profundidade | 7 | 7 |
| Random Forest | 100 ou 200 árvores, cada opção combinada com profundidade máxima de 12, 20 ou sem limite | 6 | 6 |
| MLP | Seis arquiteturas com duas preparações do redshift, mais as seis combinações da tabela abaixo | 18 | 12 |

Para as primeiras 12 opções da MLP, as camadas ocultas tinham (32), (64), (128), (64, 32), (128, 64) ou (64, 32, 16) unidades. Cada arquitetura foi avaliada com padronização direta e com `asinh(redshift / 0,001)` antes da padronização. Nessas opções, usaram-se ReLU, Adam, alpha de 0,0001, taxa inicial de 0,001, lotes de 256 e nenhuma ponderação de classe.

### Seis opções adicionais da MLP

| Camadas ocultas | Ativação | Alpha | Taxa inicial | Lote | Preparação do redshift | Pesos por classe |
| --- | --- | ---: | ---: | ---: | --- | --- |
| (64, 32) | Tanh | 0,0001 | 0,001 | 256 | Padronização direta | Não |
| (128, 64) | Tanh | 0,0001 | 0,001 | 256 | Asinh e padronização | Não |
| (64, 32) | ReLU | 0,001 | 0,003 | 512 | Asinh e padronização | Não |
| (128, 64) | ReLU | 0,00001 | 0,003 | 256 | Asinh e padronização | Não |
| (64, 32) | ReLU | 0,0001 | 0,001 | 256 | Asinh e padronização | Sim |
| (128, 64, 32) | ReLU | 0,001 | 0,003 | 512 | Asinh e padronização | Sim |

As seis combinações adicionais também usam Adam. No cenário com quatro cores, a MLP padroniza as entradas. Como seis pares de configurações diferiam apenas pela transformação do redshift, restam 12 configurações distintas.

A seleção usa F1 macro na validação. Para a MLP, a avaliação é feita a cada cinco épocas. Todas as opções passam por 30 épocas e as três melhores de cada cenário podem seguir até 200. O refinamento para após 30 épocas sem melhora superior a 0,0001 no F1 macro. As configurações escolhidas e o número de épocas usado no treinamento final são gravados em `resultados/resultados.json`.

## F1 macro esperado no teste

| Modelo | Com redshift | Só cores, configuração mantida | Só cores, nova seleção |
| --- | ---: | ---: | ---: |
| Regressão logística multinomial | 94,99% | 61,63% | 61,62% |
| Árvore de decisão | 97,10% | 80,45% | 80,90% |
| Random Forest | 97,51% | 83,14% | 83,18% |
| MLP | 97,01% | 83,74% | 83,55% |

O arquivo `resultados/metricas.csv` tem os valores sem arredondamento. Os rótulos aparecem na ordem GALAXY, QSO e STAR nas matrizes de confusão.
