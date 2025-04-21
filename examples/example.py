import sys
sys.path.append('.')
sys.path.append('..')
import shutil
from sklearn.metrics import f1_score
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from active_matcher.active_learning import EntropyActiveLearner
from active_matcher.fv_generator import FVGenerator
from active_matcher.feature_selector import FeatureSelector
from active_matcher.ml_model import  SKLearnModel, SparkMLModel
from active_matcher.labeler import  GoldLabeler
from active_matcher.algorithms import select_seeds
from xgboost import XGBClassifier
import pandas as pd
from warnings import simplefilter
from pathlib import Path
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
simplefilter(action="ignore", category=FutureWarning)


spark =  SparkSession.builder\
                        .master('local[*]')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

#shutil.make_archive('active_matcher', 'zip', '../')
#spark.sparkContext.addPyFile('active_matcher.zip')

data_dir = Path('./data/dblp_acm/')
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))

A.show()

cand.show()

gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)

model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)

selector = FeatureSelector(extra_features=False)

features = selector.select_features(A.drop('_id'), B.drop('_id'))

fv_gen = FVGenerator(features)
fv_gen.build(A, B)
fvs = fv_gen.generate_fvs(cand)
fvs = model.prep_fvs(fvs, 'features')

fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
seeds = select_seeds(fvs, 'score', 50, labeler)

active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)

fvs = trained_model.predict(fvs, 'features', 'prediction')
fvs = trained_model.prediction_conf(fvs, 'features', 'confidence')

res = fvs.toPandas()

predicted_matches = set(res.loc[res['prediction'].eq(1.0)][['id1', 'id2']].itertuples(name=None, index=False))

true_positives = len(gold & predicted_matches)
precision = true_positives / len(predicted_matches)
recall = true_positives / len(gold)
f1 = (precision * recall * 2) / (precision + recall)

print(
f'''
{true_positives=}
{precision=}
{recall=}
{f1=}
'''
)
