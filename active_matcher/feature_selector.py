from active_matcher.tokenizer import AlphaNumericTokenizer, NumericTokenizer, QGramTokenizer, StrippedQGramTokenizer, WhiteSpaceTokenizer, StrippedWhiteSpaceTokenizer, ShingleTokenizer
from active_matcher.feature import ExactMatchFeature, TFIDFFeature, JaccardFeature, OverlapCoeffFeature, SIFFeature, CosineFeature
from active_matcher.feature import EditDistanceFeature, MongeElkanFeature, SmithWatermanFeature, NeedlemanWunschFeature, RelDiffFeature
import pyspark.sql.functions as F
import pyspark.sql.types as T
import pandas as pd
from active_matcher.utils import get_logger, repartition_df, type_check

log = get_logger(__name__)

class FeatureSelector:

    TOKENIZERS = [
            StrippedWhiteSpaceTokenizer(),
            NumericTokenizer(),
            QGramTokenizer(3),
    ]

    EXTRA_TOKENIZERS = [
            AlphaNumericTokenizer(),
            QGramTokenizer(5),
            StrippedQGramTokenizer(3),
            StrippedQGramTokenizer(5),
    ]

    TOKEN_FEATURES = [
        TFIDFFeature,
        JaccardFeature, 
        SIFFeature,
        OverlapCoeffFeature, 
        CosineFeature,
    ]

    EXTRA_TOKEN_FEATURES = [

    ]



    def __init__(self, extra_features=False):
        """
        Parameters
        ----------

        extra_features : bool
            flag to enable extra features, this will GREATLY increase the number of features output

        """
        type_check(extra_features, 'extra_features', bool)

        self._extra_features = extra_features
        
        self._tokenizers = FeatureSelector.TOKENIZERS.copy()
        self._token_features = FeatureSelector.TOKEN_FEATURES.copy()
        if self._extra_features:
            self._tokenizers += FeatureSelector.EXTRA_TOKENIZERS
            self._token_features += FeatureSelector.EXTRA_TOKEN_FEATURES

        self._numeric_features = [
                RelDiffFeature
        ]

        self._sequence_features = [

        ]
        self.projected_columns_ = None

    
    @staticmethod
    def _tokenize_and_count(df_itr, token_col_map):

        for df in df_itr:
            yield pd.DataFrame({
                    col : df[t[1]].apply(t[0].tokenize) for col, t in token_col_map.items()
                    }).applymap(lambda x : len(x) if x is not None else None)

    
    def _drop_nulls(self, df, threshold):
        null_percent = df.select(*[(df[c].isNull() | F.isnan(c)).cast('int').alias(c) for c in df.columns])\
                        .agg(*[F.mean(c).alias(c) for  c in df.columns])\
                        .toPandas()\
                        .iloc[0]

        cols = null_percent.index[null_percent.lt(threshold)]
        return df.select(*cols)

    def select_features(self, A, B, null_threshold=.5):
        """
        Parameters
        ----------

        A : pyspark.sql.DataFrame
            the raw data that will have FVS generated for it

        B : pyspark.sql.DataFrame or None
            the raw data that will have FVS generated for it

        null_threshold : float
            the portion of values that must be null in order for the column to be dropped and 
            not considered for feature generation

        """
        df = A
        if B is not None:
            df = df.unionAll(B)
        
        n_start_cols = len(df.columns)
        df = repartition_df(df, 5000, [])
        df = self._drop_nulls(df, null_threshold)
        self.projected_columns_ = df.columns

        log.info(f'{n_start_cols - len(df.columns)} columns dropped because more than {null_threshold * 100}% of values were null or nan')
        log.info(f'columns after dropping nulls :\n {df.columns}')

        numeric_cols = [c.name for c in df.schema if c.dataType in {T.IntegerType(), T.LongType(), T.FloatType(), T.DoubleType()}]
        log.info(f'numeric columns in dataframe :\n {numeric_cols}')
        # cast everything to a string
        df = df.select(*[F.col(c).cast('string') for c in df.columns])

        token_cols = {}
        for t in self._tokenizers:
            for c in df.columns:
                cname = t.out_col_name(c)
                token_cols[cname] = (t, c)

        schema = T.StructType([T.StructField(c, T.IntegerType()) for c in token_cols])
        df = df.mapInPandas(lambda x : self._tokenize_and_count(x, token_cols), schema=schema)
        #record_count = df.count()

        avg_counts = df.agg(*[F.mean(c).alias(c) for c in token_cols])\
                        .toPandas().iloc[0]
        # not used right now
        #unique_counts = {}
        #for c in token_cols:
        #    unique_counts[c] = df.select(F.explode(c)).distinct().count()

        features = []
        
        for c in self.projected_columns_:
            features.append(ExactMatchFeature(c, c))

        for c in numeric_cols:
            features.append(RelDiffFeature(c, c))

        for token_col_name, p in token_cols.items():
            tokenizer, column_name = p  
            avg_count = avg_counts[token_col_name]
            #unique_count = unique_counts[token_col_name]

            if avg_count >= 3:
                features += [f(column_name, column_name, tokenizer=tokenizer) for f in self._token_features]

            if str(tokenizer) == AlphaNumericTokenizer.NAME:
                if avg_count <= 10:
                    features.append(MongeElkanFeature(column_name, column_name, tokenizer=tokenizer))
                    features.append(EditDistanceFeature(column_name, column_name))
                    features.append(SmithWatermanFeature(column_name, column_name))

        df.unpersist()
        return features



            




        
        





        


        
