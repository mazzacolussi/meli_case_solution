import pandas as pd
import numpy as np

import yaml
import os

import logging

from utils.training_utils import get_features_attribute

import warnings
warnings.filterwarnings('ignore')


def unpack_feature(col, feature):
    return col.get(feature) if col else None


def attribute_extract(attributes, key):
    
    if not isinstance(attributes, (list, tuple, np.ndarray)):
        return None
    
    for attribute in attributes:
        if isinstance(attribute, dict) and attribute.get('id') == key:
            return attribute.get('value_name')

    return None


def main():
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação da base interim.')

    logger.info('Leitura da base raw.')
    df = pd.read_parquet(os.path.join('data', 'raw', 'amostra_raw.parquet'))
    logger.info(f'Sucesso! Base lida. Shape da base: {df.shape}')

    features_config = yaml.safe_load(open(os.path.join('src', 'config', 'feature_config.yaml'), "r"))

    logger.info('Tratando dados duplicados: excluindo linhas de mesmo `id`.')
    df = df.drop_duplicates(subset='id', keep='first')
    logger.info(f'Linhas duplicadas removidas. Shape: {df.shape}.')

    logger.info('Iniciando o processo de desempacotamento das features `struct`.')

    features_sale_price = ['conditions', 'metadata', 'payment_method_type', 'type']
    mapping_sale_price_features = {
        'conditions': ['start_time', 'end_time', 'eligible'],
        'metadata': ['promotion_type']
    }

    for feature in features_sale_price:
        logger.info(f'----> feature: sale_price > {feature}.')
        df[f'sale_price_{feature}'] = df['sale_price'].apply(lambda x: unpack_feature(x, feature))

        sub_features = mapping_sale_price_features.get(feature)
        if sub_features:
            for sub_feature in sub_features:
                logger.info(f'└── {sub_feature}')
                df[f'sale_price_{feature}_{sub_feature}'] = df[f'sale_price_{feature}'].apply(lambda x: unpack_feature(x, sub_feature))


    df['sale_price_conditions_start_time'] = pd.to_datetime(df['sale_price_conditions_start_time'], errors = 'coerce')
    df['sale_price_conditions_end_time'] = pd.to_datetime(df['sale_price_conditions_end_time'], errors = 'coerce')
    df['stop_time'] = pd.to_datetime(df['stop_time'], errors = 'coerce')



    features_attribute = ['brand', 'color', 'main_color', 'weight']
    for feature in features_attribute:
        logger.info(f'----> feature: attributes > {feature}.')
        df[f'attributes_{feature}_value'] = df['attributes'].apply(lambda x: attribute_extract(x, feature.upper()))


    features_shipping = ['free_shipping', 'logistic_type']
    for feature in features_shipping:
        logger.info(f'----> feature: shipping > {feature}.')
        df[f'shipping_{feature}'] = df['shipping'].replace({np.nan: None}).apply(lambda x: unpack_feature(x, feature))


    features_address = ['city_name', 'state_name']
    for feature in features_address:
        logger.info(f'----> feature: address > {feature}.')
        df[f'address_{feature}'] = df['address'].replace({np.nan: None}).apply(lambda x: unpack_feature(x, feature))    


    features_address = ['city_name', 'state_name']
    for feature in features_address:
        logger.info(f'----> feature: address > {feature}.')
        df[f'address_{feature}'] = df['address'].replace({np.nan: None}).apply(lambda x: unpack_feature(x, feature))    



    features_installments = ['quantity', 'metadata']
    mapping_installments_features = {
        'metadata': ['additional_bank_interest', 'meliplus_installments']
    }

    for feature in features_installments:
        logger.info(f'----> feature: installments > {feature}.')
        df[f'installments_{feature}'] = df['installments'].apply(lambda x: unpack_feature(x, feature))


    logger.info('Construção das targets: `promotion_flag` e `discount`.')
    df['promotion_flag'] = (df['sale_price_type'] == 'promotion').astype(int)
    df['discount'] = ((df['original_price'] - df['price'].fillna(0)) / df['original_price']).fillna(0)


    logger.info('Tratando a feature de preço do produto, evitando data leaky em casos que houve promoção.')
    df.loc[(df['promotion_flag'] == 1), 'price_tratado'] = df['original_price']
    df.loc[(df['promotion_flag'] != 1), 'price_tratado'] = df['price']
    logger.info('Sucesso! Tratamento realizado.')

    features_struct = get_features_attribute(features_config, attribute = 'struct')
    features_hard_remove = get_features_attribute(features_config, attribute = 'hard_remove')


    cols_to_remove = \
        list(features_struct.keys()) + \
        list(features_hard_remove.keys()) + \
        [f'sale_price_{name}' for name in mapping_sale_price_features.keys()]     

    logger.info(f'Removendo as features `hard_remove` e `struct`: {cols_to_remove}.')
    df = df.drop(cols_to_remove, axis = 1)
    logger.info('Features removidas.')

    logger.info('Removendo linhas em que não há valor do produto.')
    df = df.dropna(subset='price_tratado')
    logger.info(f'Sucesso! Shape da base: {df.shape}.')


    logger.info("Salvando a base interim.")
    path_output = os.path.join('data', 'interim', f'amostra_interim.parquet')
    df.to_parquet(path_output, index=False, engine='pyarrow')
    logger.info("Sucesso! Base interim salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()