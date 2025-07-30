# This example shows how to use Active Matcher in the basic mode for a cluster of machines
# Step 3: Import the Dependencies
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

# Step 4: Initialize Spark
spark =  SparkSession.builder\
                        .master('{url of Spark Master}')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

# Step 5: Reading the Data
#path to the data
data_dir = Path(__file__).resolve().parent
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))

# Step 6: Specifying a Labeler
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)

# Step 7: Creating a Machine Learning Model to Serve as the Matcher
model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)

# Step 8: Creating Features for the ML Model
selector = FeatureSelector(extra_features=False)
features = selector.select_features(A.drop('_id'), B.drop('_id'))

# Step 9: Creating the Feature Vectors
fv_gen = FVGenerator(features) 
fv_gen.build(A, B) 
fvs = fv_gen.generate_fvs(cand) 
fvs = model.prep_fvs(fvs, 'features') 

# Step 10: Scoring the Feature Vectors
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))

# Step 11: Selecting Seeds
seeds = select_seeds(fvs, 50, labeler, 'score')

# Step 12: Using Active Learning to Train the Matcher
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)

# Step 13: Applying the Trained Matcher
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

"""
Step 14: Running the Python Program
You have finished writing a Python program for matching with ActiveMatcher, using a cluster.
In order to run this on a cluster, we can use the following command from the root directory (you can always get to the root directory by typing cd into the terminal).

Note: This command assumes that the directory structure is the same as ours, and if you followed our installation guide, it will be the same. 
Otherwise you should change the directory /home/ubuntu/dblp_acm/am_cluster_example.py specified below.

spark/bin/spark-submit \
  --master {url of Spark Master} \
  --deploy-mode client
  /home/ubuntu/dblp_acm/am_cluster_example.py

"""
