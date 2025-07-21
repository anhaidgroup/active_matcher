## Running ActiveMatcher on a Single Machine

Here we will walk through an example of running ActiveMatcher on a single machine. In particular, we will show you how to create a Python program step by step, then execute it at the end of the walkthrough. We assume you have already installed ActiveMatcher on a single machine, using [this guide](https://github.com/anhaidgroup/active_matcher/blob/docs/doc/installation-guides/install-single-machine.md).

### Step 1: Download the Datasets

To begin, we need to download the datasets from GitHub. Navigate to the [dblp_acm folder](https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm) then click 'cand.parquet' and click the download icon at the top. Repeat this for 'gold.parquet', 'table_a.parquet', and 'table_b.parquet'. Now move all these into a local folder called 'dblp_acm'. To explain these files: 
* The files 'table_a.parquet' and 'table_b.parquet' contain the tuples of Table A and Table B, respectively.
* The file 'cand.parquet' contains candidate tuple pairs that are output by the blocker. Each tuple pair is of the form (x,y) where x is a tuple in A and y is a tuple in B. The goal of ActiveMatcher is to predict for each such tuple pair whether it is a match or non-match.
* The file 'gold.parquet' contains the gold matches, that is, the IDs of all tuple pairs that are matches between Tables A and B. This file is used here only to compute the accuracy of the matching step. 

### Step 2: Create a Python File

Within the 'dblp_acm' directory, create a file called 'example.py'. As we walk through the subsequent steps, we will add code to this file. 

### Step 3: Import the Dependencies

Now we add the following code to the Python file to import all of the necessary packages that we will use.

```
import sys
sys.path.append('.')
sys.path.append('..')
import shutil
from sklearn.metrics import f1_scoremodel
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

### Step 4: Initialize Spark

Next we initialize Spark, which runs in the local mode (that is, on your local machine) in this example.

```
spark =  SparkSession.builder\
                        .master('local[*]')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

```

### Step 5: Reading the Data

Once we have the SparkSession initialized, we read in the tables along with our candidate set.

```
data_dir = Path('./')
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))
```

Here the provided datasets, table_a and table_b, have the same schema. ActiveMatcher requires that the datasets (that is, tables) being matched have the same schema. This schema must also contain an ID column. Note that each tuple (that is, record) must have a value for this ID column and all values (across the tuples) must be different. Here the ID columns for both table_a and table_b are named '_id'.

The candidate set file 'cand.parquet' is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the records in Table B and the ids in cand['id1_list'] refer to the records in Table A with ids A['_id']. This is an efficient way to store and manipulate a large number of candidate tuple pairs. 

### Step 6: Specifying a Labeler

ActiveMatcher uses a labeler to label a candidate tuple pair as match or non-match. It does this in the step to create a set of seeds for the active learning process and in the step of active learning itself (as we describe soon). 

#### Using the Command-Line Interface Labeler

We have provided a labeler that operates within the command-line interface (CLI). To specify this labeler, you should put the following code into the Python file: 
```
from active_matcher.labeler import CLILabeler

labeler = CLILabeler(a_df=A, b_df=B, id_col:'_id')
```
Here '_id' is the name of the ID columns for Tables A and B. This labeler will display a pair of tuples (x,y) to the CLI, side by side, then ask you to specify if x and y match, or do not match, or if you are unsure. It then displays the next pair of tuples, and so on. 

#### Using the Gold Labeler

In this example, since we do have access to gold, that is, tuple pairs that are matches between Tables A and B, we will use the gold labeler, by adding the following code to the Python file: 
```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```
Here, if ActiveMatcher wants to know if a pair of tuples (x,y) is a match or non-match, it simply consults 'gold'. Thus, this is a "simulated" active learning process which is completely automatic. It is not a "real" active learning process (like with the CLI Labeler), because it does not require a human user to be in the loop (to label the tuple pairs). 

Such simulated active learning using gold is very useful for code development, debugging, and computing the accuracy of the matching process. For the rest of this example, we will use this gold labeler. 

#### Using Other Labelers

Currently we do not provide more labelers. But you can extend the labeling code in ActiveMatcher to create more powerful labelers, such as one that uses a GUI instead of the command-line interface. You can do this by subclassing the Labeler class. 
 
### Step 7: Creating a Machine Learning Model to Serve as the Matcher

Next we specify a machine learning (ML) classification model to serve as the matcher. Here we will use XGBClassifier, which exposes an SKLearn model interface. In general, you can select any classification model that you believe will fit your data well and exposes an SKLearn or SparkML model interface. 
 
SKLearn model options are described [here](https://scikit-learn.org/stable/supervised_learning.html), and SparkML model options are described [here](https://spark.apache.org/docs/latest/ml-classification-regression.html). Note that even though XGBClassifier exposes an SKLearn model interface, it is not included in the SKLearn package and so is not described there. See instead its documentation [here](https://xgboost.readthedocs.io/en/stable/index.html). 

To continue with our example, the following code specifies the XGBClassifier model: 
```
model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)
```
Note that we pass the type of model (XGBClassifier), not a model instance. Additionally, we pass model-specific keyword args as we would when constructing the model normally. In this case we passed
```
eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42
```
#### Avoid Models with Slow Training and/or Inference
Each iteration in the active learning process requires training a new model and then applying that model to each feature vector we are doing active learning on. So you should avoid using a model where training and/or inference (that is, model application) are slow, otherwise the active learning process will be slow. 

#### Be Careful with Threading (** NOT SURE IF THIS APPLIES TO LOCAL MACHINES? **)
Many models use multiple threads for training and inference. Since training takes place on the Spark driver node, it is okay if model training is done with multiple threads. 

However, the inference process is distributed across workers which each have tasks that run on threads. Then, the sklearn model also tries to use multiple threads. This can cause more threads to be running than CPU cores availabe. Therefore, for inference, the model should not use multiple threads as it will cause significant oversubscription of the processor and lead to extremely slow model inference times (including during active learning). 

Fortunately, SKLearn provides an easy way to disable threading using threadpoolctl, SKLearnModel automatically disables threading for inference using threadpoolctl meaning that sklearn models shouldn't require any modification and can be passed to SKLearnModel unchanged. If you are interested in reading more about oversubscription with sklearn, please check out their documentation [here](https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads).

The model threading issue discussed above is specific to SKLearn models and does not affect SparkML models.

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
fvs = model.prep_fvs(fvs, 'features') # This ensures that fvs is the correct datatype (vector or array), fills in NaN values, and saves the feature vectors (fvs) in a column called 'features'.
```

## Step Nine: Scoring the Feature Vectors

Once we have the feature vectors, we need to score each pair such that the higher the score a pair recieves, the more likely it is to be a match. In this example, we just take the sum of all the components of the feature vector for each pair. This step is important for the optional down sampling and for selecting seeds. The score serves as a heuristic to evaluate if a pair is likely a match or likely a non-match.

```
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
```

## Optional Step: Downsampling
In some cases where the number of records is very large (over one million), it can be beneficial to take a sample of the data for model training. In many cases of matching, the number of non-matches will significantly outweigh the number of matches. Thus, random sampling could lead to a skewed sample of non-matches.  ActiveMatcher provides a method which uses a heuristic to ensure the sample contains both non-matches and matches. Here is an example of how to use the down sampling method:

```
from active_matcher.algorithms import down_sample

sampled_fvs = down_sample(fvs, percent = .1, score_column= 'score')
```

By setting percent equal to .1, we are telling the down_sample method to give us a sample of the feature vectors with a size of |fvs|*.1, so we will end up with 10% of the original size fvs.

## Step Ten: Selecting Seeds

Once we have the feature vectors created ('fvs'), and a score column ('score'), we can select seeds.

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
