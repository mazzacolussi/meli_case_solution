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
    features_config = yaml.safe_load(open(os.path.join('src', 'config', 'feature_infos.yaml'), "r"))

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
        logger.info(f"{country['id']}:")
        cats = requests.get(f"https://api.mercadolibre.com/sites/{country['id']}/categories")
        category_ids_country = {item['id'] for item in cats.json()}

        with ThreadPoolExecutor() as executor:
            dataframes = list(executor.map(lambda category_id: fetch_category_data(country['id'], category_id), category_ids_country))

        df_country_id = pd.concat(dataframes, ignore_index=True)

        return df_country_id
    
    for country in country_info['country']:
        df = process_country(country)
        df[list(features_struct)] = df[list(features_struct)].apply(lambda x: str(x) if isinstance(x, dict) else x)

        path_output = os.path.join('data', 'raw', f'df_raw_{country["id"]}.parquet')
        df.to_parquet(path_output, index=False, engine='pyarrow')
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'

    handler = logging.FileHandler(os.path.join('reports', 'logs', 'download_raw_data.txt'), encoding='utf-8')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(log_fmt))

    logging.basicConfig(level=logging.INFO, handlers=[handler])

    main()