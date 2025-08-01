import pyspark
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
from active_matcher.utils import (
    type_check, save_training_data_streaming, load_training_data_streaming
)
from active_matcher.labeler import Labeler, WebUILabeler
import logging

log = logging.getLogger(__name__)

def down_sample(fvs, percent, score_column='score', search_id_column='id2', bucket_size=25000):
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


def select_seeds(fvs, nseeds, labeler, score_column='score', 
                parquet_file_path='active-matcher-training-data.parquet'):
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

    parquet_file_path : str
        path to save/load training data parquet file

    """
    type_check(fvs, 'fvs', pyspark.sql.DataFrame)
    type_check(score_column, 'score_column', (str, pyspark.sql.Column))
    type_check(nseeds, 'nseeds', int)
    type_check(labeler, 'labeler', Labeler)
    type_check(parquet_file_path, 'parquet_file_path', str)

    if isinstance(score_column, str):
        score_column = F.col(score_column)

    # Load existing training data if available
    existing_training_data = load_training_data_streaming(parquet_file_path, log)
    
    if existing_training_data is not None:
        log.info(f'Found {len(existing_training_data)} existing labeled examples')
        # Check if we already have enough seeds
        if len(existing_training_data) >= nseeds:
            log.info(f'Already have {len(existing_training_data)} labeled examples, returning existing data')
            return existing_training_data.head(nseeds)
        else:
            log.info(f'Using {len(existing_training_data)} existing examples, need {nseeds - len(existing_training_data)} more')
            seeds = existing_training_data.to_dict('records')
            pos_count = existing_training_data['label'].sum()
            neg_count = len(existing_training_data) - pos_count
    else:
        seeds = []
        pos_count = 0
        neg_count = 0

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

    # iteratively label vectors, attempt to produce a 
    # set 50% positive 50% negative set 
    i = 0
    if isinstance(labeler, WebUILabeler):
        log.warning(f"Ready for seeds to be labeled.")
    
    while pos_count + neg_count < nseeds and i < nseeds * 2:
        try:
            idx, ex = next(maybe_pos) if pos_count <= neg_count else next(maybe_neg)
            label = float(labeler(ex['id1'], ex['id2']))
            
            if label == -1.0:  # User requested to stop
                log.info("User stopped labeling seeds")
                break
            elif label == 2.0:  # User marked as unsure
                log.info("Skipping unsure example")
                continue
            elif label == 1.0:  # Positive match
                pos_count += 1
            else:  # label == 0.0, Negative match
                neg_count += 1

            ex['label'] = label
            seeds.append(ex)
            
            # Save the newly labeled seed immediately
            new_seed_df = pd.DataFrame([ex])
            save_training_data_streaming(new_seed_df, parquet_file_path, log)
            
        except StopIteration:
            log.warning("Ran out of examples before reaching nseeds")
            break
        i += 1
    if not seeds:
        raise RuntimeError("No seeds were labeled before stopping")
    if isinstance(labeler, WebUILabeler):
        log.warning("Seed selection is complete.")
    log.info(f'seeds: pos_count = {pos_count} neg_count = {neg_count}')
    return pd.DataFrame(seeds)
