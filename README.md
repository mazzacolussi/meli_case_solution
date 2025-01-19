Case de Logística - Meli
==================================================

Este projeto visa resolver uma entre quatro problemáticas propostas em um case de logística para cientistas de dados do Mercado Livre. A solução abrange todo o processo, desde a criação da base de dados bruta, extraída via API, até a construção dos artefatos necessários para um eventual deploy em produção.

A problemática escolhida consiste em desenvolver um modelo capaz de classificar se um determinado produto deve ser colocado em promoção e, além disso, quantificar o valor ideal do desconto. O contexto do case é descrito da seguinte forma:

> En la vertical de pricing están interesados en dar sugerencias de descuentos para los
> ítems del Marketplace. Actualmente, tienen un equipo experto en jación de precios,
> los cuales revisan ítems manualmente para encontrar si el ítem requiere un descuento
> y cuál sería el descuento adecuado para generar un aumento en las ventas en el
> corto plazo. Este equipo busca disminuir las cargas manuales que tienen sus
> colaboradores para que puedan dedicarse a otras actividades más rentables para el
> negocio.



# Etapas de desenvolvimento

Para garantir o bom funcionamento do desenvolvimento, recomenda-se que instale os pacotes necessário:

```bash
pip install -r requirements.txt
```


## 1. Geração da base raw

A primeira etapa do desenvolvimento consiste na amostragem via API fornecida neste case. Para isso, foi iterado para cada `site_id` (`MLA`, `MLB`, etc) existente em que a resposta (`result`) não fosse vazia, os arquivos de apoio se encontram na pasta [src/config](src/config). Também, como proposto, iterou-se sobre os offsets discretizados (offset $\in \{0, 50, 100, ..., 950\}$) e para todas as categorias de produtos existentes no marketplace da respectiva região da iteração.

```bash
python src/data/download_raw_data.py
```


## 2. Análise exploratória de dados

A análise exploratória dos dados pode ser consultada em [notebook/01-EDA.ipynb](notebooks/01-EDA.ipynb). Iniciou-se por uma análise de volumetria e univariada da base raw coletada, observando possíveis amostras repetidas, features pouco informativas e possibilidade de criação de novas features a partidas das já existentes. Posteriormente, após a criação das targets, iniciou-se uma análise bivariada, bem como possíveis eventos de data leaky que poderiam enviesar os modelos. Assim, essa análise foi de grande utilidade para a construção da base processada (próxima etapa).


## 3. Geração da base processada

A finalidade dessa etapa é a construção de uma base pronta para o início do desenvolvimento do modelo, semelhante ao conceito de bases SOT. logo, há o desempacotamento de colunas potenciais a serem utilizadas nos modelos, conforme a breve análise exploratória realizada na etapa anterior. Então, o processo resultante proporcionará uma base pronta para consumo, removendo possíveis duplicadas, features não informativas e criação das targets dos modelos.

Para a efetivação dessa etapa, basta executar:

```bash
python src/data/basic_process.py
```

A tabela resultante estará na pasta `data/processed`.


## 4. Geração da base interim

O processo de criação de features será realizado nessa etapa, produzindo uma base pronta para a modelagem em `data/interim`. Para tal, o script de apoio está em [src/utils/transformers.py](src/utils/transformers.py), este servirá como um pacote interno do projeto, onde há classes auxiliares para a construção da pipeline completa de modelagem.

Execute:

```bash
python src/features/build_features.py
```


## 5. Split das bases de treino e teste

Para realizar o split das bases de treino e teste, execute o comando abaixo. Nesse processo, serão criadas bases específicas para o modelo de classificação (responsável por prever se o produto é uma promoção) e para o modelo de regressão (destinado a prever o valor do desconto). Para o modelo de regressão, optou-se por utilizar apenas dados históricos de produtos que efetivamente foram promocionais, já que produtos não promocionais não possuem descontos.

```bash
python src/data/train_test_split.py
```
As bases estarão disponíveis na pasta `data/train_test`.


## 6. Feature Selection

No processo de feature selection foi utilizado o Boruta. Execute os comandos abaixos para realizar a seleção das variáveis para cada um dos modelos. Ao finalizar a execução será gerado um arquivo das features selecionadas e rejeitadas (Classificação: [features_selected.yaml](src/features/selected/features_selected.yaml) e Regressão: [features_selected_regression.yaml](src/features/selected/features_selected_regression.yaml))


> Classificação:

```bash
python src/features/feature_selection_clf.py
```

> Regressão:

```bash
python src/features/feature_selection_reg.py
```


## 7. Create encoders

Os encoders serão gerados nessa etapa na pasta `models`, assim como as bases encodadas para o início da modelagem. Esse procedimento é essencial para ter controle nos inputs dos modelos, como: preenchimentos de dados vazios, correção de strings, agrupamento de variáveis de alta cardinalidade e padronização de tipagem.

> Classificação:

```bash
python src/features/create_encoder.py
```
> Regressão:

```bash
python src/features/create_encoder_regression.py
```


## 8. Model selection



```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```


## 9. Tunning

```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```


## 10. Model training

```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```


## 11. Geração dos artefatos

```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```


## 12. Análise dos resultados

Os resultados obtidos podem ser consultados em [notebook/10-Resultados.ipynb](notebook/10-Resultados.ipynb), bem como breves análises a respeito das métricas e interpretações.



Organização do projeto

------------

    ├── README.md                       <- README do projeto para guiar a sua execução.
    │
    ├── data
    │   ├── interim                     <- Dados intermediários, transformados a partir dos dados raw.
    │   ├── processed                   <- Dados finais: dados canônicos para o processo de modelagem.
    │   └── raw                         <- Dados raw, originais e imutáveis.
    │
    ├── notebooks                       <- Jupyter notebooks para análise descritiva, 
    │                                      acompanhamento do desenvolvimento e processo interativos.
    │
    ├── reports                         <- Relatórios do desenvolvimento.
    │   └── logs                        <- Logs dos scripts executáveis.
    │   └── visualization               <- Gráficos, imagens, evidências.
    │
    ├── requirements.txt                <- Requirements do projeto para o controle de versão dos pacotes.
    │                                      Gerado de forma análoga a `pip freeze > requirements.txt`.
    │
    ├── pyproject.toml                  <- Torna o projeto instalável (`pip install -e .`). 
    │                                      Permite que src possa ser importado sem conflitos de diretórios.
    │
    │
    ├── src                             <- Códigos fontes para uso neste projeto.
    │   │
    │   ├── __init__.py                 <- Torna src um módulo Python.
    │   │
    │   ├── data                        <- Scripts de download da base raw e processamentos intermediários.
    │   │   └── download_raw_data.py
    │   │   └── basic_process.py
    │   │
    │   ├── features                    <- Scripts de processamento, criação e seleção de features.
    │   │    └── build_features.py
    │   │    └── create_encoders.py
    │   │    └── feature_selection.py
    │   │    └── selected               <- Contém o arquivo de features selecionadas.
    │   │  
    │   │    
    │   ├──  models                     <- Scripts relacionados aos modelos.
    │   │   └── generate_artifacts.py
    │   │   └── model_selection.py
    │   │   └── tunning.py
    │   │
    │   └── utils                       <- Pasta de pacotes internos do projeto.
    │
    └── models                          <- Modelos treinados e serializados 
        └── encoders                       (encoders, predictor, artefato final).
        └── predictos
        └── wrapped

--------

 

<p><small>Projeto baseado em <a target="_blank" href=https://drivendata.github.io/cookiecutter-data-science/>cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>