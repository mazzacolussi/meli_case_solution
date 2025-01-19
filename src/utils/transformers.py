import pandas as pd
import numpy as np
import re
import json

from unidecode import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

def brand_processing(brand):
    if brand is not None:
        brand = brand.lower()
        return re.sub(r'[^\w\s]', '', brand)
    else:
        return None


def color_name_processing(color):
    if color is not None:
        color = color.lower()
        color = re.sub(r'[^\w\s]', '', color)

        color_dict = {
            'black': ['negro', 'preto'],
            'white': ['blanco', 'branco'],
            'gray': ['gris', 'cinza'],
            'dark gray': ['gris oscuro', 'cinza escuro'],
            'blue': ['azul', 'azul marinho', 'marinho'],
            'silver': ['plateado', 'prateado', 'prata', 'plata'],
            'red': ['rojo', 'vermelho'],
            'green': ['verde'],
            'pink': ['rosa'],
            'yellow': ['amarillo', 'amarelo'],
            'orange': ['naranja', 'laranja'],
            'transparent': ['transparente'],
            'multicolor': ['multicolor'],
            'turquoise': ['turquesa'],
            'brown': ['marron', 'marrom'],
            'purple': ['purpura', 'roxo'],
            'gold': ['dorado', 'ouro'],
            'beige': ['beige', 'bege'],
            'maroon': ['granate', 'marrom escuro'],
            'violet': ['violeta'],
            'magenta': ['magenta'],
            'salmon': ['salmao', 'salmon'],
        }

        if color in color_dict:
            return color
        else:
            for col_en, col_esp_pt in color_dict.items():
                if color in col_esp_pt:
                    return col_en
            return '<other>'
    else:
        return None


def weight_processing(weight_str_value) -> float:
    if type(weight_str_value) == str:
        if weight_str_value.strip() != '':
            value, unit = re.match(r'(-?[\d.,]+)\s*([a-zA-Z]+)', weight_str_value.strip().lower(), re.IGNORECASE).groups()
            conversao = {
                'kg': 1000,
                'g': 1,
                'lb': 453.592,
                'oz': 28.3495,
                'mg': 0.001,
                'mcg': 0.000001,
                't': 1000000
            }
            mult = conversao.get(unit, np.nan)
            output_value = float(value)*mult

            if output_value < 0:
                return np.nan
            else:
                output_value
    return np.nan


def ratio_calc(num, denom):
    if np.isnan(denom):
        return np.nan
    else:
        return num/denom


class CriaFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, training=False):
        "Classe usada para processar os dados de entrada, criando novas features a partir das variáveis existentes"
        super().__init__()

        self.training = training
    
    def __repr__(self):
        return "Objeto destinado para criar features"
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        X = self.cria_features(X)
        return X

    def cria_features(self, X):
        def aplica_variaveis(X: pd.DataFrame) -> pd.DataFrame:

            X["category_id_tratado"] = X["category_id"].apply(lambda x: x[3:])
            X['domain_id_tratado'] = X['domain_id'].apply(lambda x: x[4:])

            X['attributes_brand_value'] = X['attributes_brand_value'].apply(brand_processing)
            X['attributes_color_value'] = X['attributes_color_value'].apply(color_name_processing)
            X['attributes_main_color_value'] = X['attributes_main_color_value'].apply(color_name_processing)
            X['attributes_weight_value'] = X['attributes_weight_value'].apply(weight_processing)

            X['address_city_name'] = X['address_city_name'].apply(
                lambda x: unidecode(x).lower() if x is not None else x
            )
            X['address_state_name'] = X['address_state_name'].apply(
               lambda x: unidecode(x).lower() if x is not None else x
            )

            X['installments_price'] = X.apply(lambda x: ratio_calc(x['price_tratado'], x['installments_quantity']), axis=1)

            return X
        
        if self.training:
            return aplica_variaveis(X)
        else:
            return aplica_variaveis(X)


class Json2DF:  
    def __init__(self):
        """Classe que faz o parse do json (payload) para o dataframe"""
    
    def fit(self, X, y):
        pass

    def __repr__(self):
        return "Conversor de JSON"
    
    def transform(self, input_data):
        if type(input_data) == str:
            json_input = str(input_data)
            return pd.json_normalize(json.loads(json_input)).replace(to_replace=[None], value=np.nan)
        elif type(input_data) == pd.DataFrame:
            return input_data


class Selector:
    def __init__(self, features: list, target: str, mode='train'):
        self.features = features
        self.target = target
        self.mode = mode
    
    def __repr__(self):
        return f"Seletor de variáveis. Modo: {self.mode}. Features: {self.features}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        if self.mode == 'train':
            return X[self.features] 
        elif self.mode == 'inference':
            return X[self.features] 

    
class FillStringMissing:
    def __init__(self, cols_to_adjust):
        """ Preenche missing em string com o valor '<vazio>'"""
        self.cols_to_adjust = cols_to_adjust

    def __repr__(self):
        return f"Preenchedor de missings em string para '<vazio>'. Colunas {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].fillna('<vazio>')
        return X



class NormalizeLowerString:
    def __init__(self, cols_to_adjust):
        self.cols_to_adjust = cols_to_adjust
    
    def normalize(self, x):
        y = x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return y
    
    def __repr__(self):
        return f"Converte string para minúsculo e remove pontuações. Colunas {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].apply(self.normalize)
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].apply(lambda x: x.str.lower())
        return X



class BoolHandler:
    def __init__(self, cols_to_adjust):
        self.cols_to_adjust=cols_to_adjust

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].astype(float).fillna(-1.0)
        return X
    


class FillNull:
    def __init__(self, cols_to_adjust):
        """Preenche missing numéricas específicas com valor -999"""
        self.cols_to_adjust=cols_to_adjust
    
    def __repr__(self):
        return f"Preenche missing numéricas específicas com valor -999. Colunas {self.cols_to_adjust}"
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].fillna(-999)
        return X
    


class Grouper:
    def __init__(self, features_to_group):
        self.features_to_group=features_to_group
        self.categ_features={}

    def fit(self, X, y=None):
        for feature in self.features_to_group:
            self.categ_features[feature] = X[feature].value_counts().index.to_list()[:self.features_to_group[feature]]
        return self

    def transform(self, X):
        for feature in self.features_to_group:
            X.loc[~X[feature].isin(self.categ_features), feature] = '<outros>'
        return X
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)



class ConverteFloat:
    def __init__(self):
        pass

    def __repr__(self):
        return "Conversor de float"
    
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.astype(float)
        return X
