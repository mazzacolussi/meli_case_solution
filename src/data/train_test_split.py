
import os
import pandas as pd
import logging
import click 

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

@click.command()
@click.option('--dataset_name', default='amostra_processada.parquet', help='Nome do dataset processado', type=str)
def main(dataset_name):
    logger = logging.getLogger(__name__)

    logger.info('Iniciando o split da base - Classificação promoção')
    df = pd.read_parquet(os.path.join('data', 'processed', dataset_name))
    logger.info(f'Shape da base: {df.shape}')

    df_train, df_test, y_train, y_test = train_test_split(
        df.drop(['promotion_flag', 'sale_price_metadata_promotion_type', 'discount', 'price', 'original_price'], axis=1), 
        df.promotion_flag, 
        test_size = 0.2, 
        random_state = 98
    )

    df_train = pd.concat([df_train, y_train.rename('promotion_flag')], axis=1)
    df_test = pd.concat([df_test, y_test.rename('promotion_flag')], axis=1)

    logger.info(f'Salvando a base de treino. Shape {df_train.shape}.')
    df_train.to_parquet(os.path.join('data', 'train_test', 'train_promotion_clf.parquet'), index=False)

    logger.info(f'Salvando a base de teste. Shape {df_test.shape}.')
    df_test.to_parquet(os.path.join('data', 'train_test', 'test_promotion_clf.parquet'), index=False)

    logger.info(f'Sucesso! Base para o treino do modelo de promoção salva!')


    logger.info('Iniciando o split da base - Quantificação do desconto.')

    df_promotions = df[df['promotion_flag'] == 1].copy()

    df_train_promo, df_test_promo, y_train_promo, y_test_promo = train_test_split(
        df_promotions.drop(['promotion_flag', 'discount', 'price', 'original_price'], axis=1), 
        df_promotions.discount, 
        test_size = 0.2, 
        random_state = 98
    )

    df_train_promo = pd.concat([df_train_promo, y_train_promo.rename('discount')], axis=1)
    df_test_promo = pd.concat([df_test_promo, y_test_promo.rename('discount')], axis=1)

    logger.info(f'Salvando a base de treino. Shape {df_train_promo.shape}.')
    df_train_promo.to_parquet(os.path.join('data', 'train_test', 'train_promotion_reg.parquet'), index=False)

    logger.info(f'Salvando a base de teste. Shape {df_test_promo.shape}.')
    df_test_promo.to_parquet(os.path.join('data', 'train_test', 'test_promotion_reg.parquet'), index=False)

    logger.info(f'Sucesso! Base para o treino do modelo de quantificação do desconto salva!')


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()