import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
from active_matcher.utils import type_check
from active_matcher.labeler import Labeler



def down_sample(fvs, percent, score_column, search_id_column='id2', bucket_size=25000):
    """
    down sample fvs by score_column, producing fvs.count() * percent rows

    Parameters
    ----------

    fvs : pyspark.sql.DataFrame
        the feature vectors to be down sampled

    percent : float
        the portion of the vectors to be output, (0.0, 1.0]

    score_column : str, pyspark.sql.Column
        the column that will be used to score the vectors, should be positively correlated with 
        the probability of the pair being a match

    search_id_column : str, pyspark.sql.Column
        the column that will be used to hash the vectors into buckets, if fvs 
        are the output of top-k blocking this should be the id of the serach record.

    bucket_size : int
        the size of the buckets that the vectors will be hashed into for sampling
    """
    type_check(fvs, 'fvs', pyspark.sql.DataFrame)
    type_check(percent, 'percent', float)
    type_check(bucket_size, 'bucket_size', int)
    type_check(score_column, 'score_column', (str, pyspark.sql.Column))
    type_check(score_column, 'search_id_column', (str, pyspark.sql.Column))

    if bucket_size < 1000:
        raise ValueError('bucket_size must be >= 1000')

    if percent <= 0 or percent > 1.0:
        raise ValueError('percent must be in the range (0.0, 1.0]')

    if isinstance(score_column, str):
        score_column = F.col(score_column)

    # temp columns for sampling
    percentile_col = '_PERCENTILE'
    hash_col = '_HASH'

    window = Window().partitionBy(hash_col).orderBy(score_column.desc())
    nparts = max(fvs.count() // bucket_size, 1)
    fvs = fvs.withColumn(hash_col, F.xxhash64(search_id_column) % nparts)\
                    .select('*', F.percent_rank().over(window).alias(percentile_col))\
                    .filter(F.col(percentile_col) <= percent)\
                    .drop(percentile_col, hash_col)
                    
    return fvs


def select_seeds(fvs, score_column, nseeds, labeler):
    """
    down sample fvs by score_column, producing fvs.count() * percent rows

    Parameters
    ----------

    fvs : pyspark.sql.DataFrame
        the feature vectors to be down sampled

    score_column : str, pyspark.sql.Column
        the column that will be used to score the vectors, should be positively correlated with 
        the probability of the pair being a match

    nseeds : int
        the number of seeds to be selected
        
    labeler : Labeler
        the labeler that will be used to label the seeds

    """
    type_check(fvs, 'fvs', pyspark.sql.DataFrame)
    type_check(score_column, 'score_column', (str, pyspark.sql.Column))
    type_check(nseeds, 'nseeds', int)
    type_check(labeler, 'labeler', Labeler)

    if isinstance(score_column, str):
        score_column = F.col(score_column)

    # TODO handle edge cases
    fvs = fvs.filter((~F.isnan(score_column)) & (score_column.isNotNull()))
    # lowest scoring vectors
    maybe_neg = fvs.sort(score_column, ascending=True)\
                    .limit(nseeds)\
                    .toPandas()\
                    .iterrows()

    # highest scoring vectors
    maybe_pos = fvs.sort(score_column, ascending=False)\
                    .limit(nseeds)\
                    .toPandas()\
                    .iterrows()
    pos_count = 0
    neg_count = 0
    seeds = []
    # iteratively label vectors, attempt to produce a 
    # set 50% positive 50% negative set 
    for i in range(nseeds):
        idx, ex = next(maybe_pos) if pos_count <=neg_count else next(maybe_neg)
        label = labeler(ex['id1'], ex['id2'])
        if label:
            pos_count += 1
        else:
            neg_count += 1

        ex['label'] = label
        seeds.append(ex)


    return pd.DataFrame(seeds)
