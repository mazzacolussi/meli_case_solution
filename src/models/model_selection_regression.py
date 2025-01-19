import os
import pandas as pd
import numpy as np
import yaml
import logging
import click
import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from utils.training_utils import find_specific_variables

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--configfile', default='feature_config.yaml', help='Arquivo descritivo das features', type=str)
@click.option('--dataset_name', default='train_encoded_reg.parquet', help='Nome do dataset de treino', type=str)
def main(configfile, dataset_name):

    logger = logging.getLogger(__name__)

    logger.info('Iniciando a seleção do modelo')

    df = pd.read_parquet(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Tabela de treino lida. Shape: {df.shape}')

    features = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))
    feature_target = find_specific_variables(features, 'target_reg', specific_value=True)

    seletor = pickle.load(
        open(os.path.join('models', 'encoders_reg', 'seletor_2.pkl'), 'rb')
    )
    
    models = {
        'RF': RandomForestRegressor(random_state=98),
        'XGBoost': XGBRegressor(random_state=98),
        'LGBM': LGBMRegressor(random_state=98)
    }

    scoring = make_scorer(mean_squared_error, greater_is_better=False)

    results = {}

    for name, model in models.items():
        kf = KFold(n_splits=5, random_state=98, shuffle=True)
        scores = cross_val_score(
            model, 
            df[seletor.features], 
            df[feature_target], 
            cv=kf, 
            scoring=scoring
        )
        
        results[name] = -scores
        logger.info(f'{name}: {round(np.mean(-scores), 4)} +/- {round(np.std(-scores), 4)}')

    logger.info('Sucesso! Construindo o gráfico dos boxplots dos desempenho dos modelos')


    fig = plt.figure(figsize=(12, 8))

    fig.suptitle('Dispersão e avaliação dos modelos')
    ax = fig.add_subplot(111)
    plt.boxplot(results.values())
    plt.ylabel('MSE')
    plt.xlabel('Modelos')
    ax.set_xticklabels(models.keys())

    plt.savefig(os.path.join('reports', 'visualization', 'Regression_models.png'))

    logger.info('Sucesso! Etapa finalizada!')


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()