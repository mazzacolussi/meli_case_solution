import os
import pickle
import logging
import click 
from utils.transformers import CriaFeatures, Json2DF
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


@click.command()
@click.option('--clf_mode', default=True, help='Define se os artefatos exportados serão do modelo de classificação ou regressão', type=bool)
def main(clf_mode):
    """Cria os binários do modelo no formato do deploy."""

    logger = logging.getLogger(__name__)
    
    logger.info('Lendo binários.')

    json_to_df = Json2DF()
    cria_features = CriaFeatures()

    if clf_mode:
        encoders_path_folder = 'models/encoders'

        logger.info('Iniciando a criação da pipeline do modelo de classificação.')

        seletor_1 = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'seletor_1.pkl'), 'rb')
        )

        seletor_2 = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'seletor_1.pkl'), 'seletor_2.pkl', 'rb')
        )

        fill_null = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'features_fill_null.pkl'), 'rb')
        )

        bool_handler = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'bool_handler.pkl'), 'rb')
        )

        fill_string_missing = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','fill_string_missing.pkl'), 'rb')
        )

        normalize_lower_string = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'normalize_lower_string.pkl'), 'rb')
        )

        grouper_object_exists = os.path.exists(os.path.join(f'{encoders_path_folder}', 'grouper.pkl'))
        if grouper_object_exists:
            grouper = pickle.load(
                open(os.path.join(f'{encoders_path_folder}','grouper.pkl'), 'rb')
            )

        encoder = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','encoder.pkl'), 'rb')
        )

        conversor_float = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','conversor_float.pkl'), 'rb')
        )

        logger.info('Sucesso! Binários lidos.')

        for model_file in os.listdir('models/predictors'):
            if model_file.startswith('model_clf') and model_file.endswith('.pkl'):
                nome_modelo = model_file[len('model_'):-len('.pkl')]

                modelo = pickle.load(
                    open(os.path.join('models', 'predictors', model_file), 'rb')
                )

                logger.info(f'Criando pipeline do modelo {nome_modelo}')

                pipeline_list = [
                    ('json_to_df', json_to_df),
                    ('seletor_1', seletor_1),
                    ('cria_features', cria_features),
                    ('seletor_2', seletor_2),
                    ('fill_null', fill_null),
                    ('bool_encoder', bool_handler),
                    ('fill_string_missing', fill_string_missing),
                    ('normalize_lower_string', normalize_lower_string),
                    #('agrupador', grouper),
                    ('encoder', encoder),
                    ('conversor_float', conversor_float),
                    ('seletor_3', seletor_2),
                    ('modelo', modelo)
                ]

                if grouper_object_exists:
                    pipeline_list.insert(pipeline_list.index(('encoder', encoder)), ('agrupador', grouper))

                pipeline_prod = Pipeline(steps=pipeline_list)

                logger.info(f'Exportando o modelo {nome_modelo} final')

                pickle.dump(
                    pipeline_prod,
                    open(os.path.join('models', 'wrapped', f'model_{nome_modelo}_pipeline_prod.pkl'), 'wb')
                )

    else:
        encoders_path_folder = 'models/encoders_reg'

        logger.info('Iniciando a criação da pipeline do modelo de regressão.')

        seletor_1 = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'seletor_1.pkl'), 'rb')
        )

        seletor_2 = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'seletor_2.pkl'), 'rb')
        )

        fill_null = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'features_fill_null.pkl'), 'rb')
        )

        bool_handler = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'bool_handler.pkl'), 'rb')
        )

        fill_string_missing = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','fill_string_missing.pkl'), 'rb')
        )

        normalize_lower_string = pickle.load(
            open(os.path.join(f'{encoders_path_folder}', 'normalize_lower_string.pkl'), 'rb')
        )

        grouper_object_exists = os.path.exists(os.path.join(f'{encoders_path_folder}', 'grouper.pkl'))
        if grouper_object_exists:
            grouper = pickle.load(
                open(os.path.join(f'{encoders_path_folder}','grouper.pkl'), 'rb')
            )

        encoder = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','encoder.pkl'), 'rb')
        )

        conversor_float = pickle.load(
            open(os.path.join(f'{encoders_path_folder}','conversor_float.pkl'), 'rb')
        )

        logger.info('Sucesso! Binários lidos.')

        for model_file in os.listdir('models/predictors'):
            if model_file.startswith('model_reg') and model_file.endswith('.pkl'):
                nome_modelo = model_file[len('model_'):-len('.pkl')]

                modelo = pickle.load(
                    open(os.path.join('models', 'predictors', model_file), 'rb')
                )

                logger.info(f'Criando pipeline do modelo {nome_modelo}')

                pipeline_list = [
                    ('json_to_df', json_to_df),
                    ('seletor_1', seletor_1),
                    ('cria_features', cria_features),
                    ('seletor_2', seletor_2),
                    ('fill_null', fill_null),
                    ('bool_encoder', bool_handler),
                    ('fill_string_missing', fill_string_missing),
                    ('normalize_lower_string', normalize_lower_string),
                    #('agrupador', grouper),
                    ('encoder', encoder),
                    ('conversor_float', conversor_float),
                    ('seletor_3', seletor_2),
                    ('modelo', modelo)
                ]

                if grouper_object_exists:
                    pipeline_list.insert(pipeline_list.index(('encoder', encoder)), ('agrupador', grouper))

                pipeline_prod = Pipeline(steps=pipeline_list)

                logger.info(f'Exportando o modelo {nome_modelo} final')

                pickle.dump(
                    pipeline_prod,
                    open(os.path.join('models', 'wrapped', f'model_{nome_modelo}_pipeline_prod.pkl'), 'wb')
                )

    logger.info('Sucesso! Artefatos exportados!')


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()