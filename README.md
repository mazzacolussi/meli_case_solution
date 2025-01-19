Case de Logística - Meli
==================================================

Neste case foi proposto a escolha de 1 entre 4 problemáticas. Sendo assim, optou-se pela primeira proposta:

"""
En la vertical de pricing están interesados en dar sugerencias de descuentos para los
ítems del Marketplace. Actualmente, tienen un equipo experto en jación de precios,
los cuales revisan ítems manualmente para encontrar si el ítem requiere un descuento
y cuál sería el descuento adecuado para generar un aumento en las ventas en el
corto plazo. Este equipo busca disminuir las cargas manuales que tienen sus
colaboradores para que puedan dedicarse a otras actividades más rentables para el
negocio.
"""


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

 [notebook/01-Analise_ingestao.ipynb]


## 3. Geração da base processada

```bash
python src/data/basic_process.py
```

## 4. Split das bases de treino e teste

```bash
python src/data/train_test_split.py
```


## 5. Split das bases de treino e teste

```bash
python src/data/train_test_split.py
```


## 6. Feature Selection

```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```


## 7. Create encoders

```bash
python src/features/feature_selection_clf.py
```

```bash
python src/features/feature_selection_reg.py
```



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
    ├── requirements.txt                <- Os requirements do projeto para o controle de versão dos pacotes.
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
    │   ├── data                        <- Scripts para download da base raw e processamentos intermediários.
    │   │   └── download_raw_data.py
    │   │   └── basic_process.py
    │   │
    │   ├── features                    <- Scripts que envolvem processamento, criação e seleção de features.
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