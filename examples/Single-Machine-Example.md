## Running ActiveMatcher on a Single Machine

Here we will walk through an example of running ActiveMatcher on a single machine. In particular, we show how to create a Python program step by step, then execute it. We assume you have installed ActiveMatcher on a single machine, using [this guide](https://github.com/anhaidgroup/active_matcher/blob/docs/doc/installation-guides/install-single-machine.md).

### Step 1: Download the Datasets

First we download the datasets from GitHub. Navigate to the [dblp_acm folder](https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm) then click 'cand.parquet' and click the download icon at the top. Repeat this for 'gold.parquet', 'table_a.parquet', and 'table_b.parquet'. Now move all these into a local folder called 'dblp_acm'. To explain these files: 
* The files 'table_a.parquet' and 'table_b.parquet' contain the tuples of Table A and Table B, respectively. Our goal is to match A and B, that is, find matches between them. 
* We assume blocking (e.g., using Sparkly or Delex) has been done. The file 'cand.parquet' contains candidate tuple pairs that are output by the blocker. Each tuple pair is of the form (x,y) where x is a tuple in A and y is a tuple in B. The goal of ActiveMatcher is to predict for each such tuple pair whether it is a match or non-match.
* The file 'gold.parquet' contains the gold matches, that is, the IDs of all tuple pairs that are matches between Tables A and B. This file is used here only to simulate a user's labeling a set of tuple pairs for training a matcher, and to compute the accuracy of the matching step. Obviously when you apply ActiveMatcher "for real", you will not have access to the gold matches. 

### Step 2: Create a Python File

Within the 'dblp_acm' directory, create a file called 'am_local_example.py'. As we walk through the subsequent steps, we will add code to this file. 

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

Here the provided datasets, table_a and table_b, have the same schema. ***ActiveMatcher requires that the datasets (that is, tables) being matched have the same schema. This schema must also contain an ID column.*** Note that each tuple (that is, record) must have a value for this ID column and all values (across the tuples) must be different. Here the ID columns for both table_a and table_b are named '_id'.

The candidate set file 'cand.parquet' is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the records in Table B and the ids in cand['id1_list'] refer to the records in Table A with ids A['_id']. This is an efficient way to store and manipulate a large number of candidate tuple pairs. 

### Step 6: Specifying a Labeler

ActiveMatcher uses a labeler to label a candidate tuple pair as match or non-match. It does this in the step to create a set of seeds for the active learning process and in the step of active learning itself (as we describe soon). 

#### Using the Command-Line Interface (CLI) Labeler

We have provided a labeler that operates within the command-line interface (CLI). To specify this labeler, you should put the following code into the Python file: 
```
from active_matcher.labeler import CLILabeler

labeler = CLILabeler(a_df=A, b_df=B, id_col:'_id')
```
Here '_id' is the name of the ID columns for Tables A and B. This labeler will display a pair of tuples (x,y) to the CLI, side by side, then ask you to specify if x and y match, or do not match, or if you are unsure. It then displays the next pair of tuples, and so on. 

#### Using the Web Labeler

We have provided a Web-based labeler that the user can use to label tuple pairs when running ActiveMatcher. Specifically, when the Spark process underlying ActiveMatcher needs to label tuple pairs, it sends these pairs to a Flask-based Web server, which in turn sends these pairs to a Streamlit GUI, where the user can label. The labeled pairs are sent back to the Flaks Web server, which in turn sends them back to the Spark process. 

The Flask-based Web server and the Streamlit GUI are hosted on the users local machine. 

To use this Web labeler, put the following code into the Python file:
```
from active_matcher.labeler import WebUILabeler

labeler = WebUILabeler(a_df=A, b_df=B, id_col:'_id', flask_port=5005, streamlit_port=8501, flask_host='127.0.0.1')
```
To explain the above paramaters: 
* Here '_id' is the name of the ID columns for Tables A and B.
* The 'flask_port' will be the port number for the Flask server to run on. The 'streamlit_port' will be the port number for the Streamlit app to be run on.
* Unless you have other processes running on port 5005 and/or 8501, there should be no need to change the default arguments for 'flask_port' or 'streamlit_port'. It is important that the 'flask_port' and 'streamlit_port' are two distinct values. You may not set them both to the same value.
* Next, 'flask_host' is the IP where the Flask server should be running. By using the default value of '127.0.0.1', we are running the Flask server locally. This means that only processes on the same machine can call the Flask endpoints (which is fine for this example).

The Streamlit UI will be run on 0.0.0.0, and you will be able to access it from your machine.

On your local machine you can open 127.0.0.1:{streamlit_port} in the browser of your choice to see the Web UI.

The Web UI will display a pair of tuples (x,y), side by side, then ask you to specify if x and y match, or do not match, or if you are unsure. It then displays the next pair of tuples, and so on. 

#### Using the Gold Labeler

*In this example, since we do have access to gold, that is, tuple pairs that are matches between Tables A and B, we will use the gold labeler,* by adding the following code to the Python file: 
```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```
Here, if ActiveMatcher wants to know if a pair of tuples (x,y) is a match or non-match, it simply consults 'gold'. Thus, this is a "simulated" active learning process which is completely automatic. It is not a "real" active learning process (like with the CLI Labeler), because it does not require a human user to be in the loop (to label the tuple pairs). 

Such simulated active learning using gold is very useful for code development, debugging, and computing the accuracy of the matching process. *For the rest of this example, we will use this gold labeler.*

#### Using Other Labelers

Currently we do not provide more labelers. But you can extend the labeling code in ActiveMatcher to create more powerful labelers. You can do this by subclassing the Labeler class (see the CLI Labeler for an example of subclassing). 
 
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
Each iteration in the active learning process requires training a new model and then applying that model to each feature vector we are doing active learning on. So you should avoid using a model where training and/or inference (that is, model application) are slow, otherwise the active learning process will be slow. Use such a model only if you think the benefits (for example, higher accuracy) will outweigh the long runtime. 

#### Avoid Threading for SKLearn Models
You do not need to take any action here. This part is only for your information. Many ML models use multiple threads for inference. However, SKLearn models appear to have a problem using multiple threads for inference. So this should be disabled. 

Fortunately, SKLearn provides an easy way to disable threading using threadpoolctl. In the ActiveMatcher code, SKLearnModel automatically disables threading for inference using threadpoolctl. So SKLearn models do not require any modification and can be passed to SKLearnModel unchanged. If you want to read more about this issue, see [this document](https://scikit-learn.org/stable/computing/parallelism.html#oversubscription-spawning-too-many-threads).

The above threading issue is specific to SKLearn models. It does not affect SparkML models.

### Step 8: Creating Features for the ML Model

We now create a set of features. In the next step we will use these features to convert each pair of tuples (x,y) in the candidate set into a feature vector. We use the following code to create the features: 
```
selector = FeatureSelector(extra_features=False)

features = selector.select_features(A.drop('_id'), B.drop('_id'))
```
The above code snippet will create features that compute similarity scores between the attributes of Table A and Table B. For example, a feature may compute the Jaccard score between A.name and B.name, after the names have been tokenized into sets of 3-grams. Another feature may compute the TF/IDF score between A.address and B.address, and so on. *ActiveMatcher uses heuristics to examine the attributes of Tables A and B and automatically generate these features.*

Note that in the above code snippet, we pass 'extra_features=False' to FeatureSelector. If we set 'extra_features=True', ActiveMatcher will generate even more features. This may improve the ML model's accuracy, but will increase the time to generate the feature vectors and to perform active learning. 

### Step 9: Creating the Feature Vectors

Now we use the features created in the previous step to convert all tuple pairs in the candidate set into feature vectors:
```
fv_gen = FVGenerator(features) 
fv_gen.build(A, B) 
fvs = fv_gen.generate_fvs(cand) 
fvs = model.prep_fvs(fvs, 'features') 
```
In the above code snippet
* Line 1 creates an FVGenerator object with the features previously created.
* Line 2 creates a binary representation of the DataFrame 'cand' and stores it on disk. This is a memory optimization to avoid the large dataframes being kept in memory.
* Line 3 creates a feature vector for each tuple pair in the cand set.
* Line 4 ensures that fvs is the correct datatype (vector or array), fills in NaN values, and saves the feature vectors (fvs) in a column called 'features'.

### Step 10: Scoring the Feature Vectors

Next we compute a score for each feature vector, such that the higher the score, the more likely that it is a match. Later we will use these scores to select a set of seeds for active learning, and optionally to obtain a sample of the candidate set for active learning. 

Here we compute the score of each feature vector to be the sum of all components of that vector. This is based on the heuristic that each component of a vector is a similarity score (such as Jaccard, cosine), so the higher the sum of these similarity scores, the more likely that the feature vector is a match (that is, the tuple pair corresponding to this vector is a match): 
```
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
```

### Step 11: Selecting Seeds

Next we select a small set of tuple pairs that we will label. This set of tuple pairs will serve as "seeds" to start the active learning process. Specifically, we will use these seeds to train an initial matcher. Then we use the matcher to look for unlabeled "informative" tuple pairs, then we ask the user to label those pairs and retrain the matcher, and so on. 

We select a set of 50 seeds as follows:
```
seeds = select_seeds(fvs, 50, labeler, 'score')
```
Here the scores that we have computed in the previous step are stored in the column 'score'. We select 25 feature vectors that have the highest scores (so they are most likely to be matches) and 25 feature vectors that have the lowest scores (so they are likely to be non-matches). 

### Step 12: Using Active Learning to train the Matcher

We now use active learning to train the matcher by adding the following code to the Python file:  
```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)
```
In the above code 
* We ask the user to label the selected seeds (as matches or non-matches), using the labeler 'labeler'.
* Then we use the labeled seeds to train the matcher specified in 'model' (which is a ML classifier in this case).
* Then we perform up to 'max_iter=50' iterations. In each iteration
  + we apply the trained matcher to all feature vectors (in the candidate set) to predict them as matches/non-matches,
  + use these predictions to select the top 'batch_size=10' most informative tuple pairs,
  + ask the user to label these selected tuple pairs as matches/non-matches,
  + then re-train the matcher using *all* tuple pairs that have been labeled so far.

The above training process stops when we have finished 'max_iter=50' iterations, or when we have run out of tuple pairs to select. In any case, we return the matcher that has been trained with all tuple pairs that have been labeled so far. 
   
### Step 13: Applying the Trained Matcher

We can now apply the trained matcher to the feature vectors in the candidate set, outputting the binary prediction into a fvs['prediction'] and the confidence score of the prediction to fvs['condifidence']. The binary prediction will be either 1.0 or 0.0. 1.0 implies that the model predicts two records are a match, and 0.0 implies that the model predicts two records are not a match. Then, the confidence score is in the range of \[0.50, 1.0\]. The confidence score is the models estimation of the probability that the 'prediction' is correct. For example if 'prediction' is 1.0 and 'confidence' is .85, then the model is 85% confident that two records are a match. On the other hand, if 'prediction' is 0.0 and 'confidence' is .85, then the model is 85% confident that two records do not match.

```
fvs = trained_model.predict(fvs, 'features', 'prediction')
fvs = trained_model.prediction_conf(fvs, 'features', 'confidence')
```

Finally, we can compute precision, recall, and f1 of the predictions made by the matcher:
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

### Step 14: Running the Python Program

You have finished writing a Python program for matching with ActiveMatcher. To run this program, open a terminal and navigate to the directory that you wrote your 'am_local_example.py' file in. Then run the following command, which will output 'true_positives', 'precision', 'recall', and 'f1':

```
python3 am_local_example.py
```
