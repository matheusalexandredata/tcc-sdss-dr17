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
4. Avalia os modelos nas mesmas 30.000 observações. Grava métricas, predições, partições, histórico da MLP e matrizes de confusão em resultados/.

## F1 macro esperado no teste

| Modelo | Com redshift | Só cores, configuração mantida | Só cores, nova seleção |
| --- | ---: | ---: | ---: |
| Regressão logística multinomial | 94,99% | 61,63% | 61,62% |
| Árvore de decisão | 97,10% | 80,45% | 80,90% |
| Random Forest | 97,51% | 83,14% | 83,18% |
| MLP | 97,01% | 83,74% | 83,55% |

O arquivo `resultados/metricas.csv` tem os valores sem arredondamento. Os rótulos aparecem na ordem GALAXY, QSO e STAR nas matrizes de confusão.
