import os
import pandas as pd
import yaml
import logging
import requests
from concurrent.futures import ThreadPoolExecutor

from utils.training_utils import get_features_attribute

import warnings
warnings.filterwarnings('ignore')


def main():

    logger = logging.getLogger(__name__)

    country_info = yaml.safe_load(open(os.path.join('src', 'config', 'country_infos.yaml'), "r"))
    features_config = yaml.safe_load(open(os.path.join('src', 'config', 'feature_config.yaml'), "r"))
    features_struct = get_features_attribute(features_config, attribute='struct')

    offsets = [i for i in range(0, 1000, 50)]

    def fetch_category_data(country_id, category_id):
        serial_lst = []
        with requests.Session() as session:
            for offset in offsets:
                url = f"https://api.mercadolibre.com/sites/{country_id}/search?category={category_id}&offset={offset}"
                response = session.get(url)
                items = response.json()
                serial_lst.append(pd.DataFrame(items['results']))
        return pd.concat(serial_lst, ignore_index=True)


    def process_country(country):
        cats = requests.get(f"https://api.mercadolibre.com/sites/{country['id']}/categories")
        category_ids_country = {item['id'] for item in cats.json()}

        with ThreadPoolExecutor() as executor:
            dataframes = list(executor.map(lambda category_id: fetch_category_data(country['id'], category_id), category_ids_country))

        df_country_id = pd.concat(dataframes, ignore_index=True)
        return df_country_id
    
    lst_dfs = []
    for country in country_info['country']:
        logger.info(f"Iniciando a criação da base Raw de {country['id']}.")
        df_parcial = process_country(country)
        logger.info(f"Shape: {df_parcial.shape}.")

        lst_dfs.append(df_parcial)
        del df_parcial

    logger.info(f"Concatenando os DataFrames.")
    df = pd.concat(lst_dfs, ignore_index=True)
    df.drop(columns=df.filter(like='variation').columns, axis=1, inplace=True)
    logger.info(f"Sucesso! {df.shape}.")

    logger.info(f"Salvando a base raw.")
    path_output = os.path.join('data', 'raw', f'amostra_raw.parquet')
    df.to_parquet(path_output, index=False, engine='pyarrow')
    logger.info(f"Sucesso! Base Raw salva!")
    
    
if __name__ == '__main__':

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()