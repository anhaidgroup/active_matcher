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

In this example, the provided datasets, table_a and table_b, have the same schema. ActiveMatcher requires that the datasets which are being matched have the same schema. Additionaly, ActiveMatcher requires that both A and B have an id column. An id column is a column where all of the values are unique integers. Since A and B are required to have the same schema, the id column in both datasets needs to have the same name as well. In this case, the id columns in A and B are both named '_id'.

Our candidate set is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the record in table B and the ids in cand['id1_list'] refer to the records in table A with ids A['_id']. We use this format for improving effeciency of generating feature vectors, especially when cand is produced by a top-k blocking algorithm.

Next we can create a labeler, for this example, we use gold data to create an automatic labeler, however the Labeler class can be subclassed to add a human in the loop. 

```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```

Additionally, if you would like to label the data with a command-line interface because you do not already have a gold dataset, we provide a class called CLILabeler. Here is an example for how you would use the CLILabeler rather than the GoldLabeler:

```
from active_matcher.labeler import CLILabeler

labeler = CLILabeler(a_df=A, b_df=B, id_col:'_id')
```

where id_col is the name of the id column in your data. 

There are two steps where ActiveMatcher requires the user to label data if a simulated (GoldLabeler) is not being used. The first is when seeds are being selected and the second is during the Active Learning process. In either of these cases, using the CLILabeler will provide an interactive labeler using the command-line. Within the command-line, the user will see two records side-by-side. They will be asked if the records match, and will be prompted to input 'y' if they do match, 'n' if they do not match, and 'u' if they are unsure. 

## Step Six: Creating a Model

Next we can choose a model to train. In this example we are using XGBClassifier, which exposes an SKLearn model interface. However, the user is free to select a model that they believe will fit their data well. The user has the option to select a model that exposes an SKLearn model interface or a SparkML model interface. 

To read about SKLearn model options, please visit their documentation [here](https://scikit-learn.org/stable/supervised_learning.html). To read about SparkML model options, please visit their documentation [here](https://spark.apache.org/docs/latest/ml-classification-regression.html). Finally, as noted above, XGBClassifier exposes an SKLearn model interface, but it is not included in the SKLearn package. To read about the XGBoost model, please visit their documentation [here](https://xgboost.readthedocs.io/en/stable/index.html). 

Now, we will demonstrate how to create the model object.

```
model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)
```

Notice that we pass the type of model (XGBClassifier), not a model instance. Additionally, we can pass model specific keyword args as we would when constructing the model normally, in this case we passed
```
eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42
```

Additionally, we want to provide two important notes for the model process:

### Model Training and Inference Time
First, each iteration in active learning requries training a new model and then applying the model to each feature vector we are doing active learning on. This means that if model training and/or inference are slow, the active learning process will be very slow.

### Model Threading
Second, many algorithms use multiple threads for training and inference. Since training takes place on the Spark driver node, it is okay if model training with multiple threads. 

However, the inference process is distributed across workers which each have tasks that run on threads. Then, the sklearn model also tries to use multiple threads. This can cause more threads to be running than CPU cores availabe. Therefore, for inference the model should not use multiple threads as it will cause significant over subscription of the processor and lead to extremely slow model inference times (including during active learning). 

Fortunately, SKLearn provides an easy way to disable threading using threadpoolctl, SKLearnModel automatically disables threading for inference using threadpoolctl meaning that sklearn models shouldn't require any modification and can be passed to SKLearnModel unchanged. If you are interested in reading more about oversubscription with sklearn, please check out their documentation [here](https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads).

The model threading issue discussed above is specific to SKLearn models and does not affect to SparkML models.

## Step Seven: Selecting Features

With all of that set up, we can now select features that we will use to generate feature vectors for each pair in cand. Here we use the default typical set of features, however extra_features can be set to True which will cause the code to generate significantly more features, and likely improve model accuracy at the cost of increased runtime for feature vector generation and active learning.

```
selector = FeatureSelector(extra_features=False)

features = selector.select_features(A.drop('_id'), B.drop('_id'))
```

## Step Eight: Generating Feature Vectors

Now that we have selected features, we can generate feature vectors for each pair in cand. First we need to build the features and then we can generate the actual feature vectors.

```
fv_gen = FVGenerator(features) # This creates an FVGenerator object with the features selected in Step Seven.
fv_gen.build(A, B) # This creates a binary representation of the DataFrame and stores it on disk. This is a memory optimization to avoid the large dataframes being kept in memory.
fvs = fv_gen.generate_fvs(cand) # generate_fvs creates feature vectors between candidate records in the 'cand' dataset.
fvs = model.prep_fvs(fvs, 'features') # This ensures that fvs is the correct datatype (vector or array), fills in NaN values, and saves the feature fectors (fvs) in a column called 'features'.
```

## Step Nine: Scoring the Feature Vectors

Once we have the feature vectors we need to score each pair such that the higher the score a pair recieves, the more likely it is to be a match. In this example, we just take the sum of all the components of the feature vector for each pair. This step is important for the optional down sampling and for selecting seeds. The score here serves as a heuristic to evaluate if a pair is likely a match or likely a non-match.

```
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
```

## Optional Step: Down sampling
In some cases where the number of records is very large (over one million), it can be beneficial to take a sample of the data. ActiveMatcher provides a method to strategically choose data for the sample. In many cases of matching, the number of non-matches will significantly outweigh the number of matches. Thus, random sampling could lead to a skewed sample of non-matches. Therefore, ActiveMatcher uses a heuristic to ensure the sample contains both non-matches and matches. Here is an example of how to use the down sampling method:

```
from active_matcher.algorithms import down_sample

sampled_fvs = down_sample(fvs, percent = .1, score_column= 'score')
```

By setting percent equal to .1, we are telling the down_sample method to give us a sample of the feature vectors with a size of |fvs|*.1, so we will end up with 10% of the original fvs.

## Step Ten: Selecting Seeds

Once we have the feature vectors, we can select seeds for active learning, for this operation we need to score each pair which is positively correlated with being a match. That is the higher the score for the pair the more likely it is to be a match. In this example, we just take the sum of all the components of the feature vector for each pair.

```
seeds = select_seeds(fvs, 50, labeler, 'score')
```

When selecting seeds, you may choose to use the sample rather than the full dataset. If you choose to do so, 'fvs' would be changed to 'sampled_fvs', so the full line would look like this:

```
seeds = select_seeds(sampled_fvs, 50, labeler, 'score')
```

## Step Eleven: Training the Model with Active Learning

Next we run active learning, for at most 50 iterations with a batch size of 10. This process will then output a trained model.

```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)
```

Additionally, active learning can be run in a 'continuous' mode. The 'continuous' mode will allow the user to keep labeling data while new models are being trained in a different thread. This process can significantly reduce the amount of time a user is waiting during the active learning process since data to be labeled will be supplied continuously. If you would prefer to perform continous active learning and you are using a GoldLabeler, your code would look like this:
```
from active_matcher.active_learning import ContinuousActiveLearning

active_learner = ContinuousEntropyActiveLearner(model, labeler, max_labeled=550, on_demand_stop=False)
trained_model = active_learner.train(fvs, seeds)
```

The max_labeled parameter is the number of examples (including the number of seeds) that the program should label. The on_demand_stop parameter tells the program that it should continue labeling until it reaches max_labeled. If you are using the CLILabeler, and you are going to label for a set period of time, or until you decide you don't want to label anymore, your call will look like this:

```
from active_matcher.active_learning import ContinuousActiveLearning

active_learner = ContinuousEntropyActiveLearner(model, labeler, on_demand_stop=True)
trained_model = active_learner.train(fvs, seeds)
```

Here, we do not need to set max_labeled parameter. The active learner will wait until they recieve a stop signal from the command line interface (an input of 's', for stop, from the user).


When training the model, the user may choose to use the sample rather than the full dataset. If you choose to do so, 'fvs' would be changed to 'sampled_fvs', so the second line (the train line) would look like this:

```
trained_model = active_learner.train(sampled_fvs, seeds)
```
## Step Twelve: Applying the Trained Model

We can then apply the trained model to the feature vectors, outputting the binary prediction into a fvs['prediction'] and the confidence of the prediction to fvs['condifidence']. 

_If you used downsampling: At this point, since we are applying the trained model to the feature vectors, we should use the full feature vectors ('fvs'), rather than the sample ('sampled_fvs')._

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

## Step Thirteen: Running the Python Program

Congratulations. You have finished writing a Python program for matching with ActiveMatcher, and now you can run the program. To do so, open a terminal and navigate to the directory that you wrote your 'example.py' file in. Finally, run the following command, and once the program is finished, it will output true_positives, 'precision', 'recall', and 'f1':

```
python3 example.py
```
