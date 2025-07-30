# This example shows how to use Active Matcher with a cluster of machines.
# Here we use both sampling and continuous learning.
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

# Down Sampling Step: Sampling the Feature Vectors
from active_matcher.algorithms import down_sample
# This sample has 10% of the feature vectors (0.1)
sampled_fvs = down_sample(fvs, 0.1, 'score')

# Step 11: Selecting Seeds
seeds = select_seeds(sampled_fvs, 50, labeler, 'score')

# Step 12: Using Continuous Active Learning to Train the Matcher
"""
In the following, we will use Continuous Active Learning rather than Batch Active Learning..
Continous Active Learning does not require the user to wait to label data as the Machine Learning model is training.
Here, we will specify two parameters:
max_labeled and on_demand_stop.

max_labeled: This parameter specifies the maximum number of labeled examples to use for training.
on_demand_stop: This parameter specifies that we should continue labeling until the user manually stops the labeling process.

For simplicity, in this example, we will use max_labeled=500 and on_demand_stop=False, so the algorithm will complete after 500 examples have been labeled.
"""
from active_matcher.active_learning import ContinuousEntropyActiveLearner
active_learner = ContinuousEntropyActiveLearner(model, labeler, max_labeled=500, on_demand_stop=False)
trained_model = active_learner.train(sampled_fvs, seeds)

# Step 13: Applying the Trained Matcher to ALL feature vectors
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