import pandas as pd
import os

import logging
from utils.transformers import CriaFeatures

import warnings
warnings.filterwarnings('ignore')

def main():
    logger = logging.getLogger(__name__)

    logger.info('Iniciando a criação das features.')
    df = pd.read_parquet(os.path.join('data', 'interim', 'amostra_interim.parquet'))
    logger.info(f'Shape da base: {df.shape}.')

    criador_features = CriaFeatures()

    logger.info('Iniciando processamento.')
    df = criador_features.transform(df)
    logger.info(f'Sucesso! Processamento finalizado. Shape da tabela processada: {df.shape}')


    logger.info(f"Salvando a base interim.")
    path_output = os.path.join('data', 'processed', f'amostra_processada.parquet')
    df.to_parquet(path_output, index=False, engine='pyarrow')
    logger.info(f"Sucesso! Base interim salva!")


if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()