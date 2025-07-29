## Running ActiveMatcher on a Cluster of Machines

Here we will walk through an example of running ActiveMatcher on a cluster of machines (on AWS). In particular, we show how to create a Python program step by step, then execute it. We assume you have installed ActiveMatcher on a cluster of machines, using [this guide](https://github.com/anhaidgroup/active_matcher/blob/docs/doc/installation-guides/install-cloud-based-cluster.md).

### Step 1: Download the Datasets

First we download the datasets from GitHub. Navigate to the [dblp_acm folder](https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm) then click 'cand.parquet' and click the download icon at the top. Repeat this for 'gold.parquet', 'table_a.parquet', and 'table_b.parquet'. Now move all these into a local folder called 'dblp_acm' on each node in the cluster. That is, each node must have a folder with the same path and that folder must contain the above three files. 

To explain these files: 
* The files 'table_a.parquet' and 'table_b.parquet' contain the tuples of Table A and Table B, respectively. Our goal is to match A and B, that is, find matches between them. 
* We assume blocking (e.g., using Sparkly or Delex) has been done. The file 'cand.parquet' contains candidate tuple pairs that are output by the blocker. Each tuple pair is of the form (x,y) where x is a tuple in A and y is a tuple in B. The goal of ActiveMatcher is to predict for each such tuple pair whether it is a match or non-match.
* The file 'gold.parquet' contains the gold matches, that is, the IDs of all tuple pairs that are matches between Tables A and B. This file is used here only to simulate a user's labeling a set of tuple pairs for training a matcher, and to compute the accuracy of the matching step. Obviously when you apply ActiveMatcher "for real", you will not have access to the gold matches. 

### Step 2: Create a Python File

On the master node, in the 'dblp_acm' directory, create a file called 'cluster_am_example.py'. As we walk through the subsequent steps, we will add code to this file. 

### Step 3: Import the Dependencies

Now we add the following code to the Python file to import all of the necessary packages that we will use.

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

### Step 4: Initialize Spark

Next we initialize Spark, which runs on a cluster for this example.
```
spark =  SparkSession.builder\
                        .master('{url of Spark Master}')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

```

### Step 5: Reading the Data

Once we have the SparkSession initialized, we can read in the tables along with our candidate set.

```
data_dir = Path('/home/ubuntu/dblp_acm')
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))
```
Here the provided datasets, table_a and table_b, have the same schema. ***ActiveMatcher requires that the datasets (that is, tables) being matched have the same schema. This schema must also contain an ID column.*** Note that each tuple (that is, record) must have a value for this ID column and all values (across the tuples) must be different. Here the ID columns for both table_a and table_b are named '_id'.

The candidate set file 'cand.parquet' is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the records in Table B and the ids in cand['id1_list'] refer to the records in Table A with ids A['_id']. This is an efficient way to store and manipulate a large number of candidate tuple pairs. 

### Step 6: Specifying a Labeler

ActiveMatcher uses a labeler to label a candidate tuple pair as match or non-match. It does this in the step to create a set of seeds for the active learning process and in the step of active learning itself (as we describe soon). Currently ActiveMatcher provides a command-line interface (CLI) labeler, a Web-based labeler, and a gold labeler. Among these, the CLI labeler can only be used when running ActiveMatcher on a single local machine. In what follows we discuss using the Web-based labeler and the gold labeler. 

#### Using the Web Labeler

We have provided a Web-based labeler that the user can use to label tuple pairs when running ActiveMatcher on a cluster of machines. Specifically, when the Spark process underlying ActiveMatcher needs to label tuple pairs, it sends these pairs to a Flask-based Web server, which in turn sends these pairs to a Streamlit GUI, where the user can label. The labeled pairs are sent back to the Flaks Web server, which in turn sends them back to the Spark process. 

The Flask-based Web server and the Streamlit GUI are hosted on the machine on which the user originally submitted the Spark job embodying ActiveMatcher. In this example, this machine is the Spark master node. But in theory, the Flask Web server and the Streamlit GUI can be hosted on any network-accessible machine. 

To use this Web labeler, put the following code into the Python file:
```
from active_matcher.labeler import WebUILabeler

labeler = WebUILabeler(a_df=A, b_df=B, id_col:'_id', flask_port=5005, streamlit_port=8501, flask_host='127.0.0.1')
```
To explain the above paramaters: 
* Here '_id' is the name of the ID columns for Tables A and B.
* The 'flask_port' will be the port number for the Flask server to run on. The 'streamlit_port' will be the port number for the Streamlit app to be run on.
* Unless you have other processes running on port 5005 and/or 8501, there should be no need to change the default arguments for 'flask_port' or 'streamlit_port'. It is important that the 'flask_port' and 'streamlit_port' are two distinct values. You may not set them both to the same value.
* Next, 'flask_host' is the IP where the Flask server should be running. ***By using the default value of '127.0.0.1', we are running the Flask server locally on the node where you submitted the Spark job (which is the master node in this example). This means that only processes on the same node can call the Flask endpoints (which is fine for this example).***

The Streamlit UI will be run on 0.0.0.0 from the master node. This makes the UI accessible via the master node's public IP. (Again, we assume here that the master node is where you submitted your Spark job.) 

We now discuss how to get the public IP of any instance (that is, node) of your cluster using the AWS dashboard. We assume that you have set up your cluster using our instructions [here](https://github.com/anhaidgroup/active_matcher/blob/main/doc/installation-guides/install-cloud-based-cluster.md). 
* If so, you should navigate to the 'EC2' section of the dashboard. Then, you can select an instance from the list of instances by clicking on the checkbox to the left of its name.
* When you select an instance on the instance page, an informational panel will appear at the bottom of the page. Switch to the ‘details’ tab and record public IPv4 address of the instance. 

Once you have obtained the public IP address of the master node, on your local machine you can open {public ip address}:{streamlit_port} in the browser of your choice to see the Web UI. For example, if the public IP address of the master node is 1.2.3.4 and you use the default Streamlit port 8501, you can enter 1.2.3.4:8501 into your browser to view the Web UI.

The Web UI will display a pair of tuples (x,y), side by side, then ask you to specify if x and y match, or do not match, or if you are unsure. It then displays the next pair of tuples, and so on. 

Finally, we note that it is okay that the Streamlit UI is accessible via the master node's public IP. This should not present a security issue, because when installing the Spark cluster, you should have set up a security group that only allows specific IPs (like your local machine) to access the nodes. So random users should not be able to access the Web UI.

#### Using the Gold Labeler

*In this example, since we do have access to gold, that is, tuple pairs that are matches between Tables A and B, we will use the gold labeler,* by adding the following code to the Python file: 
```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```
Here, if ActiveMatcher wants to know if a pair of tuples (x,y) is a match or non-match, it simply consults 'gold'. Thus, this is a "simulated" active learning process which is completely automatic. It is not a "real" active learning process (like with the Web-based Labeler), because it does not require a human user to be in the loop (to label the tuple pairs). 

Such simulated active learning using gold is very useful for code development, debugging, and computing the accuracy of the matching process. For the rest of this example, we will use this gold labeler. 

#### Using Other Labelers

Currently we do not provide more labelers. But you can extend the labeling code in ActiveMatcher to create more powerful labelers. You can do this by subclassing the Labeler class (see the Web Labeler for an example of subclassing). 

### Step 7: Creating a Model

Next we can choose a model to train. In this example we are using XGBClassifier. Notice that we pass the type of model, not a model instance. Additionally, we can pass model specific keyword args as we would when constructing the model normally, in this case we passed,

eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42
Note that while we use XGBClassifier in this example, most any model that exposes a scikit-learn interface should work with two important caveats.

Model Training and Inference Time
First, for each iteration in active learning, requries training a new model and then applying the model to each feature vector we are doing active learning on. This means that if model training and/or inference are slow, the active learning process will be very slow.

Model Threading
Second, many algorithms use multiple threads for training and inference. Since training takes place on the spark driver node, it is okay if model training with multiple threads. For inference the model should not use multiple threads as it will cause significant over subscription of the processor and lead to extremely slow model inference times (including during active learning). Fortunately, sklearn provides an easy way to disable threading using threadpoolctl, SKLearnModel automatically disables threading for inference using threadpoolctl meaning that sklearn models shouldn't require any modification and can be passed to SKLearnModel unchanged.

```
model = SKLearnModel(XGBClassifier, eval_metric='logloss', objective='binary:logistic', max_depth=6, seed=42)
```

## Step 8: Selecting Features

With all of that set up, we can now select features that we will use to generate feature vectors for each pair in cand. Here we use the default typical set of features, however extra_features can be set to True which will cause the code to generate significantly more features, and likely improve model accuracy at the cost of increased runtime for feature vector generation and active learning.

```
selector = FeatureSelector(extra_features=False)

features = selector.select_features(A.drop('_id'), B.drop('_id'))
```

## Step 9: Generating Feature Vectors

Now that we have selected features, we can generate feature vectors for each pair in cand. First we need to build the features and then we can generate the actual feature vectors.

```
fv_gen = FVGenerator(features)
fv_gen.build(A, B)
fvs = fv_gen.generate_fvs(cand)
fvs = model.prep_fvs(fvs, 'features')
```

## Step 10: Selecting Seeds

Once we have the feature vectors, we can select seeds for active learning, for this operation we need to score each pair which is positively correlated with being a match. That is the higher the score for the pair the more likely it is to be a match. In this example, we just take the sum of all the components of the feature vector for each pair.

```
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + F.when(x.isNotNull() & ~F.isnan(x), x).otherwise(0.0) ))
seeds = select_seeds(fvs, 50, labeler, 'score')
```

## Step 11: Training the Model with Active Learning

Next we run active learning, for at most 50 iterations with a batch size of 10. This process will then output a trained model.

```
active_learner = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = active_learner.train(fvs, seeds)
```

## Step 12: Applying the Trained Model

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

## Step 13: Running on a Cluster

In order to run this on a cluster, we can use the following command from the root directory (you can always get to the root directory by typing `cd` into the terminal). 

**Note**: This command assumes that the directory structure is the same as ours, and if you followed our installation guides, it will be the same.

```
spark/bin/spark-submit \
  --master {url of Spark Master} \
  /home/ubuntu/dblp_acm/example.py
```
