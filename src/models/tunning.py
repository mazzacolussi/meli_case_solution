import pandas as pd
import yaml
import os
import pickle
import logging
import click
import optuna

from utils.optuna_settings import objective_lgbm, objective_xgboost
from utils.training_utils import find_specific_variables

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--configfile', default='feature_config.yaml', help='Arquivo descritivo das features', type=str)
@click.option('--dataset_name', default='train_encoded_clf.parquet', help='Nome do dataset de treino encodado', type=str)
def main(configfile, dataset_name):
    """Encontra os melhores hiperparâmetros para o treinamento do modelo."""

    logger = logging.getLogger(__name__)

    logger.info('Iniciando o tunning')

    logger.info('Lendo a base de treino encodada')
    df = pd.read_parquet(os.path.join('data', 'train_test', dataset_name))
    logger.info(f'Sucesso! Base de treino lida. Shape: {df.shape}')

    features = yaml.safe_load(open(os.path.join('src', 'config', configfile), 'r'))
    feature_target = find_specific_variables(features, 'target_clf', specific_value=True)

    try:
        pass
    except:
        logger.error('Não foi possível ler o arquivo de saída da etapa de Feature Selection.')
        raise


    seletor2 = pickle.load(
        open('models/encoders/seletor_2.pkl', 'rb')
    )

    logger.info('Iniciando o Optuna')
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=12), study_name='XGBoost Classifier')

    func = lambda trial: objective_xgboost(
        trial,
        df[seletor2.features].values,
        df[feature_target].values,
    )

    study.optimize(
        func,
        n_trials=40,
        n_jobs=1,
        show_progress_bar=True
    )

    df_metrics_results = study.trials_dataframe()

    with open(os.path.join('models', 'df_metrics_results_tunning.pkl'), 'wb') as output:
        pickle.dump(df_metrics_results, output)

    logger.info('Sucesso! Tunning Finalizado.')


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()