import pandas as pd
import numpy as np
import unicodedata
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score



def trata_ordenacao(words:str) -> str:

    if type(words) == str:
        if words.strip('[]').lower() != 'none':
            words = words.strip('[]').replace("'", "")
            lst = words.split(', ')
            lst = [word.upper() for word in lst]
            lst.sort()
            return str(lst)
        else:
            return np.nan
    else:
        return np.nan


def position_counter(words) -> int:
    if type(words) == str:
        return words.count(',') +1
    else:
        np.nan


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
            X['fecha'] = pd.to_datetime(X['fecha'])
            X['dia_do_mes'] = X['fecha'].dt.day
            X['dia_da_semana'] = X['fecha'].dt.weekday
            X['percentual_do_mes'] = (X['fecha'].dt.day / X['fecha'].dt.days_in_month)
            X['hora_do_dia'] = X['fecha'].dt.hour
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
        X.loc[:, self.cols_to_adjust] = X.loc[:, self.cols_to_adjust].replace({
            'Y': True,
            'Yes': True,
            'N': False,
            'No': False
        })
        
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



class Calibrator(BaseEstimator):
    def __init__(self, model, params):
        self.model=model
        self.params=params

    def predict_proba(self, X, **kwargs):
        """Faz o predict_proba semelhantes aos estimadores base do sklearn"""
        return self.calibrated_model(self.model.predict_proba(X, **kwargs))
    
    def calibrated_model(self, prediction):
        """Calibra o modelo base"""

        prediction = prediction*self.params[0] + (prediction**2)*self.params[1] + (prediction**3)*self.params[2]
        prediction = np.where(prediction < 0, 0, prediction)
        prediction = np.where(prediction > 1, 1, prediction)

        return prediction
    


def custom_auc_eval(y_true, y_pred):

    is_higher_better=True
    auc = roc_auc_score(y_true, y_pred, max_fpr=0.1)

    return 'pAUC-1%', auc, is_higher_better