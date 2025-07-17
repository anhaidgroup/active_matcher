# Step-by-step guide to running ActiveMatcher on a Single Machine

This guide is a step-by-step guide to running the active matcher. For this guide, we will assume that you have already installed everything from the appropriate [Single Machine Installation Guide](https://github.com/anhaidgroup/active_matcher/blob/docs/doc/installation-guides/install-single-machine.md).

## Step One: Download datasets

To begin, we need to download the datasets from the GitHub. Navigate to the dblp_acm folder here: https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm. Then, click on 'cand.parquet' and click the download icon at the top. Repeat this for 'gold.parquet', 'table_a.parquet', and 'table_b.parquet'. Now, using your file manager on your computer, move these all into one folder called 'dblp_acm'.

## Step Two: Create Python file

Within the 'dblp_acm' directory, create a file called 'example.py'. We will use this Python file to write the code.

## Step Three: Import dependencies

Now, we can open up the 'example.py' file. Before we begin, we first need to import all of the necessary packages that we will use.

```
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
```

## Step Four: Initialize Spark

Next we need to initialize Spark, for this example we are just going to run in local mode, however ActiveMatcher can also run on a cluster seemlessly.

```
spark =  SparkSession.builder\
                        .master('local[*]')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

```

## Step Five: Reading in Data

Once we have the SparkSession initialized, we can read in the raw data along with our candidate set.

```
data_dir = Path('./')
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))
```

Our candidate set is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the record in table B and the ids in cand['id1_list'] refer to the records in table A with ids A['_id']. We use this format for improving effeciency of generating feature vectors, especially when cand is produced by a top-k blocking algorithm.

Next we can create a labeler, for this example, we use gold data to create an automatic labeler, however the Labeler class can be subclassed to add a human in the loop.

```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```

## Step Six: Creating a Model

Next we can choose a model to train. In this example we are using XGBClassifier, which exposes an SKLearn model interface. To read about SKLearn model options, please visit their documentation [here](https://scikit-learn.org/stable/supervised_learning.html). To read about SparkML model options, please visit their documentation [here](https://spark.apache.org/docs/latest/ml-classification-regression.html). As noted above, XGBClassifier exposes an SKLearn model interface, but it is not included in the SKLearn package. To read about the XGBoost model, please visit their documentation [here](https://xgboost.readthedocs.io/en/stable/index.html). Notice that we pass the type of model, not a model instance. Additionally, we can pass model specific keyword args as we would when constructing the model normally, in this case we passed,

eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42
Note that while we use XGBClassifier in this example, most any model that exposes an SKLearn or SparkML model interface should work. For models which expose an SKLearn interface, their are two important caveats.

Model Training and Inference Time
First, for each iteration in active learning, requries training a new model and then applying the model to each feature vector we are doing active learning on. This means that if model training and/or inference are slow, the active learning process will be very slow.

Model Threading
Second, many algorithms use multiple threads for training and inference. Since training takes place on the spark driver node, it is okay if model training with multiple threads. For inference the model should not use multiple threads as it will cause significant over subscription of the processor and lead to extremely slow model inference times (including during active learning). Fortunately, sklearn provides an easy way to disable threading using threadpoolctl, SKLearnModel automatically disables threading for inference using threadpoolctl meaning that sklearn models shouldn't require any modification and can be passed to SKLearnModel unchanged.

```
model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)
```

## Step Seven: Selecting Features

With all of that set up, we can now select features that we will use to generate feature vectors for each pair in cand. Here we use the default typical set of features, however extra_features can be set to True which will cause the code to generate significantly more features, and likely improve model accuracy at the cost of increased runtime for feature vector generation and active learning.

```
selector = FeatureSelector(extra_features=False)

features = selector.select_features(A.drop('_id'), B.drop('_id'))
```

## Step Eight: Generating Feature Vectors

Now that we have selected features, we can generate feature vectors for each pair in cand. First we need to build the features and then we can generate the actual feature vectors.

```
fv_gen = FVGenerator(features)
fv_gen.build(A, B)
fvs = fv_gen.generate_fvs(cand)
fvs = model.prep_fvs(fvs, 'features')
```

## Step Nine: Selecting Seeds

Once we have the feature vectors, we can select seeds for active learning, for this operation we need to score each pair which is positively correlated with being a match. That is the higher the score for the pair the more likely it is to be a match. In this example, we just take the sum of all the components of the feature vector for each pair.

```
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
seeds = select_seeds(fvs, 50, labeler, 'score')
```

## Step Ten: Training the Model with Active Learning

Next we run active learning, for at most 50 iterations with a batch size of 10. This process will then output a trained model.

```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)
```

## Step Eleven: Applying the Trained Model

We can then apply the trained model to the feature vectors, outputting the binary prediction into a fvs['prediction'] and the confidence of the prediction to fvs['condifidence'].

```
fvs = trained_model.predict(fvs, 'features', 'prediction')
fvs = trained_model.prediction_conf(fvs, 'features', 'confidence')
```

Finally, we can compute precision, recall, and f1 of the predictions made by the model.

```
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
```

### Step Twelve: Running the Python Program

Congratulations. You have finished writing a Python program for matching with ActiveMatcher, and now you can run the program. To do so, open a terminal and navigate to the directory that you wrote your 'example.py' file in. Finally, run the following command, and once the program is finished, it will output true_positives, 'precision', 'recall', and 'f1':

```
python3 example.py
```
