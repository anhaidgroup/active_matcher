from abc import abstractmethod, ABC, abstractproperty
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.functions import vector_to_array, array_to_vector
from pyspark.ml.linalg import VectorUDT
import numpy as np
import warnings
from typing import Iterator
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from joblib import parallel_backend
from threadpoolctl import ThreadpoolController

class MLModel(ABC):
    
    @abstractproperty
    def nan_fill(self):
        pass

    @abstractproperty
    def use_vectors(self):
        pass

    @abstractproperty
    def use_floats(self):
        pass

    @abstractmethod
    def predict(self, df, vector_col : str, output_col : str):
        pass

    @abstractmethod
    def prediction_conf(self, df, vector_col : str, label_column : str):
        pass

    @abstractmethod
    def entropy(self, df, vector_col : str, output_col : str):
        pass

    @abstractmethod
    def train(self, df, vector_col : str, label_column : str):
        pass

    @abstractmethod
    def params_dict(self) -> dict:
        pass


    def prep_fvs(self, fvs, feature_col='features'):
        if self.nan_fill is not None:
            fvs = fvs.withColumn(feature_col, F.transform(feature_col, lambda x : F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(self.nan_fill)))

        if self.use_vectors:
            fvs = convert_to_vector(fvs, feature_col)
        else:
            fvs = convert_to_array(fvs, feature_col)
            if self.use_floats:
                fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<float>'))
                pass
            else:
                fvs = fvs.withColumn(feature_col, fvs[feature_col].cast('array<double>'))

        return fvs

def convert_to_vector(df, col):
    if not isinstance(df.schema[col].dataType, VectorUDT):
        df = df.withColumn(col, array_to_vector(col))
    return df

_DOUBLE_ARRAY = T.ArrayType(T.DoubleType())
_FLOAT_ARRAY = T.ArrayType(T.FloatType())
_ARRAY_TYPES = {_DOUBLE_ARRAY, _FLOAT_ARRAY}

def convert_to_array(df, col):
    if df.schema[col].dataType not in _ARRAY_TYPES:
        df = df.withColumn(col, vector_to_array(col))
    return df

class SKLearnModel(MLModel):

    def __init__(self, model, nan_fill=None, use_floats=True, **model_args):
        self._model_args = model_args.copy()
        self._model = model
        self._nan_fill = nan_fill
        self._use_floats = use_floats
        self._trained_model = None
        self._vector_buffer = None
    

    def params_dict(self):
        return {
                'model' : str(self._model),
                'nan_fill' : self._nan_fill,
                'model_args' : self._model_args.copy()
        }
    
    def _no_threads(self):
        tpc = ThreadpoolController()
        tpc.limit(limits=1, user_api='openmp')
        tpc.limit(limits=1, user_api='blas')
        pass

    @property
    def nan_fill(self):
        return self._nan_fill

    @property
    def use_vectors(self):
        return False

    @property
    def use_floats(self):
        return self._use_floats

    def get_model(self):
        return self._model(**self._model_args)
        
    def _allocate_buffer(self, nrows, ncols):
        needed_size = nrows * ncols
        if self._vector_buffer is None or self._vector_buffer.size < needed_size:
            self._vector_buffer = np.empty(needed_size, dtype=(np.float32 if self.use_floats else np.float64) )

        return self._vector_buffer[:needed_size].reshape(nrows, ncols)


    def _make_feature_matrix(self, vecs):
        if len(vecs) == 0:
            return None
        buffer = self._allocate_buffer(len(vecs), len(vecs[0]))
        X = np.stack(vecs, axis=0, out=buffer)

        return X

    def _predict(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            yield pd.Series(self._trained_model.predict(X))


    def predict(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        f = F.pandas_udf(self._predict, T.DoubleType())
        return df.withColumn(output_col, f(vector_col))
    
    def _prediction_conf(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            yield pd.Series(probs.max(axis=1))

    def prediction_conf(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        f = F.pandas_udf(self._prediction_conf, T.DoubleType())
        return df.withColumn(output_col, f(vector_col))

    def _entropy(self, vec_itr : Iterator[pd.Series]) -> Iterator[pd.Series]:
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        self._no_threads()
        for vecs in vec_itr:
            X = self._make_feature_matrix(vecs.values)
            probs = self._trained_model.predict_proba(X)
            yield pd.Series(np.nan_to_num((-probs * np.log2(probs)).sum(axis=1)))
    
    def entropy(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')
        df = convert_to_array(df, vector_col)
        f = F.pandas_udf(self._entropy, T.DoubleType())
        return df.withColumn(output_col, f(vector_col))

    def train(self, df, vector_col : str, label_column : str):
        df = convert_to_array(df, vector_col)
        df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values
        self._trained_model = self.get_model().fit(X, y)
    
    def cross_val_scores(self, df, vector_col : str, label_column : str):
        df = convert_to_array(df, vector_col)
        df = df.toPandas()
        X = self._make_feature_matrix(df[vector_col].values)
        y = df[label_column].values

        scores = cross_val_score(self.get_model(), X, y, cv=10)
        return scores

class SparkMLModel(MLModel):

    def __init__(self, model, nan_fill = 0.0, **model_args):
        self._model_args = model_args.copy()
        self._model = model
        self._trained_model = None
        self._nan_fill = nan_fill

    @property
    def nan_fill(self):
        return self._nan_fill

    @property
    def use_vectors(self):
        return True

    @property
    def use_floats(self):
        return False

    def get_model(self):
        return self._model(**self._model_args)

    def params_dict(self):
        return {
                'model' : str(self._model),
                'model_args' : self._model_args.copy()
        }

    def prediction_conf(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')

        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.array_max(vector_to_array(F.col(self._trained_model.getProbabilityCol()))).alias(output_col)

        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)

    def predict(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to predict')

        df = convert_to_vector(df, vector_col)
        cols = df.columns
        out = F.col(self._trained_model.getPredictionCol()).alias(output_col)

        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, out)
    
    def _entropy_component(self, p_col, idx):
        return F.when(p_col.getItem(idx) != 0.0, -p_col.getItem(idx) * F.log2(p_col.getItem(idx))).otherwise(0.0)

    def _entropy_expr(self, probs, classes=2):
        p_col = F.col(probs)

        e = self._entropy_component(p_col, 0)
        for i in range(1, classes):
            e = e + self._entropy_component(p_col, i)

        return e


    def entropy(self, df, vector_col : str, output_col : str):
        if self._trained_model is None:
            raise RuntimeError('Model must be trained to compute entropy')
        df = convert_to_vector(df, vector_col)
        prob_col = self._trained_model.getProbabilityCol()
        prob_array = 'prob_array'
        cols = df.columns
        return self._trained_model.setFeaturesCol(vector_col)\
                                    .transform(df)\
                                    .select(*cols, vector_to_array(prob_col).alias(prob_array))\
                                    .withColumn(output_col, self._entropy_expr(prob_array))\
                                    .drop(prob_array)
        
    def train(self, df, vector_col : str, label_column : str):
        df = convert_to_vector(df, vector_col)
        self._trained_model = self.get_model().setFeaturesCol(vector_col)\
                                            .setLabelCol(label_column)\
                                            .fit(df)\
                                            .setPredictionCol('__PREDICTION_TMP')\
                                            .setProbabilityCol('__PROB_TMP')\
                                            .setRawPredictionCol('__RAW_PREDICTION_TMP')

