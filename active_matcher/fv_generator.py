import pyspark.sql.functions as F
import pyspark.sql.types as T
from active_matcher.utils import  get_logger, repartition_df
import pickle
from active_matcher.storage import MemmapDataFrame
from active_matcher.feature_selector import FeatureSelector
from active_matcher.utils import compress
from pyspark.ml.functions import array_to_vector
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threading import Lock

log = get_logger(__name__)


class BuildCache:
    def __init__(self):
        self._cache = []
        self._lock = Lock()

    def add_or_get(self, builder):
        with self._lock:
            try:
                builder = self._cache[self._cache.index(builder)]
            except ValueError:
                self._cache.append(builder)

        return builder

    def clear(self):
        with self._lock:
            self._cache.clear()
                



class FVGenerator:

    def __init__(self, features, fill_na=None):
        # columns to generate features
        self._features = features

        self._fill_na = float(fill_na) if fill_na is not None else fill_na
        # preprocessed table a and table b
        self._table_a_preproc = None
        self._table_b_preproc = None
        self._projected_columns = list(set([f.a_attr for f in features] + [f.b_attr for f in features]))
        self._preprocess_chunk_size = 100
    
    @property
    def features(self):
        return self._features

    @property
    def feature_names(self):
        return [str(f) for f in self._features]

    def _generate_feature_vectors_inner(self, rec, recs):
        f_cols = [f(rec, recs) for f in self._features]
        f_mat = np.stack(f_cols, axis=-1).astype(np.float32)

        if self._fill_na is not None:
            f_mat = np.nan_to_num(f_mat, copy=False, nan=self._fill_na)

        return f_mat
    

    def _generate_feature_vectors(self, df_itr):
        table_a = self._table_a_preproc
        table_b = self._table_b_preproc
        table_a.init()
        table_b.init()

        for df in df_itr:
            # filter out null pairs
            df = df.loc[df['id1_list'].apply(lambda x : x is not None and len(x) != 0)]
            b_recs = table_b.fetch(df['id2'].values)

            for idx, row in df.iterrows():
                b_rec = b_recs.loc[row.id2]
                # for high arity data memory can be a issue
                # fetch records lazily without caching to reduce memory pressure
                a_recs = table_a.fetch(row.id1_list)
                f_mat = self._generate_feature_vectors_inner(b_rec, a_recs)

                row['fv'] = list(f_mat)
                row.rename(index={'id1_list' : 'id1'}, inplace=True)
                yield pd.DataFrame(row.to_dict())
    
    def _gen_fvs(self, pairs, output_col='features'):
        if self._table_a_preproc is None:
            raise RuntimeError('FVGenerator must be built before generating feature vectors')
        
        fields = pairs.drop('id1_list').schema.fields
        for i, f in enumerate(fields):
            # is an array field
            if hasattr(f.dataType, 'elementType'):
                fields[i] = T.StructField(f.name, f.dataType.elementType)

        schema = T.StructType(fields)\
                        .add('id1', 'long')\
                        .add('fv', T.ArrayType(T.FloatType()))\
        
        log.info(f'schema of fvs {schema}')
        pairs = repartition_df(pairs, 50, 'id2')       
        fvs = pairs.mapInPandas(self._generate_feature_vectors , schema=schema)\
                    .withColumn('_id', F.monotonically_increasing_id())\
                    .withColumnRenamed('fv', output_col)
                    
        return fvs
    
    
    def _preprocess_data(self, data, pp_for_a, pp_for_b):
        if pp_for_a:
            for f in self._features:
                data = f.preprocess(data, True)
        if pp_for_b:
            for f in self._features:
                data = f.preprocess(data, False)


        return data

    def _get_processing_columns(self, df, pp_for_a, pp_for_b):

        data = df.limit(5).toPandas().set_index('_id')
        data = self._preprocess_data(data, pp_for_a, pp_for_b)
        return data.columns

    def _preprocess(self, df_itr, pp_for_a, pp_for_b):
        for dataframe in df_itr:
            for start in range(0, len(dataframe), self._preprocess_chunk_size):
                if start >= len(dataframe):
                    break
                end = min(start + self._preprocess_chunk_size, len(dataframe))
                df = dataframe.iloc[start:end].set_index('_id')
                df = self._preprocess_data(df, pp_for_a, pp_for_b)

                df = df.apply(lambda x : MemmapDataFrame.compress(pickle.dumps(x.values)), axis=1)\
                        .to_frame(name='pickle')\
                        .reset_index(drop=False)

                yield df
            

    def _create_sqlite_df(self, df, pp_for_a, pp_for_b):
        
        if not pp_for_a and not pp_for_b:
            raise RuntimeError('preprocessing must be done for a and/or b')

        schema = T.StructType([
            df.schema['_id'],
            T.StructField('pickle',  T.BinaryType())
        ])
        
        log.info('preprocesing data')
        # project out unused columns
        df = df.select('_id', *self._projected_columns)
        cols = self._get_processing_columns(df, pp_for_a, pp_for_b)
        df = df.mapInPandas(lambda x : self._preprocess(x, pp_for_a, pp_for_b), schema)
        
        log.info('constructing sqlite df')
        sqlite_df = MemmapDataFrame.from_spark_df(df, 'pickle', cols)

        return sqlite_df
    
    def _prepreprocess_table(self, df):
        part_size = 5000
        df = repartition_df(df, part_size, '_id')\
                .select('_id', *[F.col(c).cast('string') for c in df.columns if c != '_id'])
        
        return df
    
    def build(self, A, B=None):
        A = self._prepreprocess_table(A).persist()
            
        if B is not None:
            B = self._prepreprocess_table(B).persist()
        
        log.info('building features')
        cache = BuildCache()
        pool = Parallel(n_jobs=-1, backend='threading')
        pool(delayed(f.build)(A, B, cache) for f in self._features)
        cache.clear()
        
        if B is not None:
            delayed_build = delayed(self._create_sqlite_df)
            self._table_a_preproc, self._table_b_preproc = pool([delayed_build(A, True, B is None), delayed_build(B, False, True)])
            self._table_b_preproc.to_spark()
        else:
            self._table_a_preproc = self._create_sqlite_df(A, True, B is None)
            self._table_b_preproc = self._table_a_preproc

        self._table_a_preproc.to_spark()
        
        A.unpersist()
        if B is not None:
            B.unpersist()
        

    def generate_fvs(self, pairs):

        log.info('generating features')
        
        fvs = self._gen_fvs(pairs)

        return fvs

    def generate_and_score_fvs(self, pairs):

        log.info('generating features')
        
        fvs = self._gen_fvs(pairs)

        log.info('scoring records')
        positively_correlated = self._get_pos_cor_features()

        fvs = self._score_fvs(fvs, positively_correlated)
        return fvs

    def _get_pos_cor_features(self):
        positively_correlated_features = {
            'exact_match',
            'needleman_wunch',  # TODO: double check this may be a typo
            'smith_waterman',
            'jaccard',
            'overlap_coeff',
            'cosine',
            'monge_elkan_jw',
            'tf_idf',
            'sif'
        }
        return [1 if any(str(f).startswith(prefix) for prefix in positively_correlated_features) else 0
                for f in self._features]
    
    def _score_fvs(self, fvs, positively_correlated):
        pos_cor_array = F.array(*[F.lit(x) for x in positively_correlated])

        return (fvs.withColumn("score", F.aggregate(
            F.zip_with("features", pos_cor_array, 
                       lambda x, y: F.nanvl(x, F.lit(0.0)) * y),
                       F.lit(0.0), 
                       lambda acc, x: acc + x)
                       )
                )

    def release_resources(self):
        if self._table_a_preproc is not None:
            self._table_a_preproc.delete()
            self._table_a_preproc = None

        if self._table_b_preproc is not None:
            self._table_b_preproc.delete()
            self._table_b_preproc = None




        


