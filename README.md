# Análise de Perfis de Consumo e Saúde - SuperVida

Este projeto contém a análise de dados de clientes da rede de supermercados SuperVida, com o objetivo de identificar perfis de consumo e saúde para subsidiar o lançamento de um programa de alimentação saudável.

## Estrutura do Projeto

- mineracaov2.ipynb → Notebook Python com o código da análise

- Pós Graduação em Ciência de Dados.docx → Relatório gerencial completo

- *.png → Visualizações geradas pela análise

- dados_processados_final.csv → Dataset original

- dataset_limpo.csv → Dataset processado após todas as transformações

## OBejtivos

- Identificar perfis de consumo e saúde a partir de dados reais

- Realizar análises descritivas (EDA) com variáveis demográficas e de hábitos

- Segmentar consumidores com clustering (KMeans)

- Prever consumidores naturais com classificação (Árvore de Decisão e KNN)

- Extrair regras de associação (Apriori) entre categorias de produtos

- Gerar recomendações de Business Intelligence (BI) para campanhas e fidelização

## Requisitos

Para executar o código, você precisará das seguintes bibliotecas Python:

```
pandas
matplotlib
seaborn
numpy
scikit-learn
mlxtend

```

## Instalação

Para instalar as dependências necessárias, execute:

```bash
pip install pandas matplotlib seaborn numpy scikit-learn mlxtend

```

from google.colab import drive

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

## Execução

Para executar a análise completa:

1. Certifique-se de que o arquivo `dados_processados_final` está no mesmo diretório do script
2. Execute o script Python:

```bash
python refactored_mining.py
```

O script irá:
- Carregar e pré-processar os dados
- Realizar engenharia de features
- Executar análises descritivas
- Aplicar técnicas de clustering e classificação
- Gerar visualizações em formato PNG
- Salvar o dataset processado como `dataset_limpo.csv`

## Visualizações

As seguintes visualizações são geradas:

- `distribuicao_perfis_saudaveis.png`: Distribuição de perfis saudáveis
- `consumo_medio_por_faixa_etaria.png`: Consumo médio por faixa etária
- `percentual_saudaveis_por_sexo.png`: Percentual de perfis saudáveis por sexo
- `consumo_medio_por_escolaridade.png`: Consumo médio por escolaridade
- `distribuicao_consumo.png`: Distribuição geral de consumo
- `consumo_natural_faixa_etaria_otimizada.png`: Consumo natural por faixa etária otimizada
- `silhouette_score.png`: Análise de Silhouette Score para validação de clusters
- `perfis_consumidores_clustering.png`: Visualização dos clusters de consumidores
- `matriz_confusao_arvore_decisao.png`: Matriz de confusão do modelo de árvore de decisão
- `matriz_confusao_knn.png`: Matriz de confusão do modelo K-NN


## Visualizações Geradas

distribuicao_perfis_saudaveis.png

consumo_medio_por_faixa_etaria.png

percentual_saudaveis_por_sexo.png

consumo_medio_por_escolaridade.png

- distribuicao_consumo.png

- consumo_natural_faixa_etaria_otimizada.png

- silhouette_score.png

- perfis_consumidores_clustering.png

- matriz_confusao_arvore_decisao.png

matriz_confusao_knn.png

## Resultados Principais

- 37,37% dos clientes possuem perfil saudável de consumo

- Identificação de 3 clusters:

- Industrializado (37%)

- Saudável (27%)

- Misto (36%)

- Regras de associação revelaram forte vínculo entre naturais, orgânicos e snacks saudáveis

- Modelos de classificação atingiram 100% de acurácia (necessário validar overfitting)

## Relatório e Apresentação

- O relatório gerencial completo está disponível em `Pós Graduação em Ciência de Dados`


## Contato

Para mais informações ou dúvidas sobre o projeto, entre em contato com a aluna Thamiriz Costa.

