# Step-by-step guide to running ActiveMatcher on a Cloud-Based Cluster

This guide is a step-by-step guide to running the active matcher. For this guide, we will assume that you have already installed everything from the [Cloud Based Cluster Guide](https://github.com/anhaidgroup/active_matcher/blob/docs/doc/installation-guides/install-cloud-based-cluster.md) and created a Spark cluster.

## Step 1: Download datasets --- We should change when the datasets are hosted to make it easier using wget

To begin, we need to download the datasets from the GitHub. Navigate to the dblp_acm folder here: https://github.com/anhaidgroup/active_matcher/tree/main/examples/data/dblp_acm. Then, click on 'cand.parquet' and click the download icon at the top. Repeat this for 'gold.parquet', 'table_a.parquet', and 'table_b.parquet'. Now, using your file manager on your computer, move these all into one folder called 'dblp_acm'. (if the data is not hosted somewhere, we will need to add instructions about using scp). This should be done on all of the nodes.

## Step 2: Create Python file

On the master node, in the 'dblp_acm' directory, create a file called 'example.py'. We will use this Python file to write the code.

## Step 3: Import dependencies

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

## Step 4: Initialize Spark

Next we need to initialize Spark and for this example we will run on a cluster.

```
spark =  SparkSession.builder\
                        .master('{url of Spark Master}')\
                        .config('spark.sql.execution.arrow.pyspark.enabled',  'true')\
                        .getOrCreate()

```

## Step 5: Reading in Data

Once we have the SparkSession initialized, we can read in the raw data along with our candidate set.

```
data_dir = Path('/home/ubuntu/dblp_acm')
A = spark.read.parquet(str(data_dir / 'table_a.parquet'))
B = spark.read.parquet(str(data_dir / 'table_b.parquet'))
cand = spark.read.parquet(str(data_dir / 'cand.parquet'))
```

Our candidate set is a set of rolled up pairs, where cand['id2'] refers to the B['_id'] of the record in table B and the ids in cand['id1_list'] refer to the records in table A with ids A['_id']. We use this format for improving effeciency of generating feature vectors, especially when cand is produced by a top-k blocking algorithm.

## Step 6: Specifying a Labeler

ActiveMatcher uses a labeler to label a candidate tuple pair as match or non-match. It does this in the step to create a set of seeds for the active learning process and in the step of active learning itself (as we describe soon). 

### Using the Web-based Labeler

We have provided a labeler that operates within a web-based environment. The web-based labeler will set up a Flask server with endpoints for fetching examples to be labeled and for submitting the labeled examples. It will also create a Streamlit UI that the user can access from their local machine to label the data. To specify this labeler, you should put the following code into the Python file: 
```
from active_matcher.labeler import WebUILabeler

labeler = WebUILabeler(a_df=A, b_df=B, id_col:'_id', flask_port=5005, streamlit_port=8501, flask_host='127.0.0.1')
```
Here '_id' is the name of the ID columns for Tables A and B. The 'flask_port' will be the port number for the Flask server to run on. The 'streamlit_port' will be the port number for the Streamlit app to be run on. Unless you have other processes running on port 5005 and/or 8501, there should be no need to change the default arguments for 'flask_port' or 'streamlit_port'. It is important that the 'flask_port' and 'streamlit_port' are two distinct values. You may not set them both to the same value. Next, 'flask_host' is the ip where the flask server should be running. By using the default value of '127.0.0.1', it is saying to run the server locally on the instance where you submitted the Spark job. This means that only processes on the same instance can call the Flask endpoints, which is correct for this use case.

The Streamlit UI will be run on 0.0.0.0 which will make it accessible via the instances public ip. Here, the instance refers to the instance where you submitted your Spark job. We will now discuss how to get the public ip from the instance using the AWS dashboard. The following assumes that you have set up your cluster using our instructions [here]() on AWS. If this is true, you should navigate to the 'EC2' section of the dashboard. Then, you can select an instance from the list of instances by clicking on the checkbox to the left of its name. When you select an instance on the instance page, an informational panel will appear at the bottom of the page. Switch to the ‘details’ tab and record public IPv4 address of the instance running your Spark job. It is okay that the Streamlit UI is accessible via the instances public ip because in the installation, you would have set up a security group that only allows specific ip's (like your local machine) to access the instance at all. So, even though it is avaiable via the public ip, random users should not be able access the site.

Once you get the public ip address, on your local machine you can open {public ip address}:{streamlit_port} in the browser of your choice to see the Web UI. If your public ip address happened to be 1.2.3.4 and you used the default streamlit port 8501, then you would enter 1.2.3.4:8501 into your browser to view the UI.

This labeler will display a pair of tuples (x,y) to the web interface, side by side, then ask you to specify if x and y match, or do not match, or if you are unsure. It then displays the next pair of tuples, and so on. 

### Using the Gold Labeler

*In this example, since we do have access to gold, that is, tuple pairs that are matches between Tables A and B, we will use the gold labeler,* by adding the following code to the Python file: 
```
gold_df = pd.read_parquet(data_dir / 'gold.parquet')
gold = set(zip(gold_df.id1, gold_df.id2))
labeler = GoldLabeler(gold)
```
Here, if ActiveMatcher wants to know if a pair of tuples (x,y) is a match or non-match, it simply consults 'gold'. Thus, this is a "simulated" active learning process which is completely automatic. It is not a "real" active learning process (like with the CLI Labeler), because it does not require a human user to be in the loop (to label the tuple pairs). 

Such simulated active learning using gold is very useful for code development, debugging, and computing the accuracy of the matching process. For the rest of this example, we will use this gold labeler. 

#### Using Other Labelers

Currently we do not provide more labelers. But you can extend the labeling code in ActiveMatcher to create more powerful labelers, such as one that uses a GUI instead of the command-line interface. You can do this by subclassing the Labeler class. 
 

## Step 7: Creating a Model

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
