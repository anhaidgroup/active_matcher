![license](https://img.shields.io/github/license/anhaidgroup/sparkly)

# ActiveMatcher

Welcome to ActiveMatcher! ActiveMatcher is a machine learning system specialized for
entity matching built on top of Apache Spark.


## Quick Start : ActiveMatcher in 30 Seconds


There are five main steps to running ActiveMatcher,

1. Reading Data

```python
spark = SparkSession.builder.getOrCreate()

table_a = spark.read.parquet('./examples/data/abt_buy/table_a.parquet')
table_b = spark.read.parquet('./examples/data/abt_buy/table_b.parquet')
cands = spark.read.parquet('./examples/data/abt_buy/cand.parquet')
```

2. Feature Selector
```python
selector = FeatureSelector(extra_features=False)
features = selector.select_features(A.drop('_id'), B.drop('_id'))
```

3. Feature Vector Generation
```python
fv_gen = FVGenerator(features)
fv_gen.build(A, B)
fvs = fv_gen.generate_fvs(cands_df)
```

4. Select Seeds

```python
fvs = fvs.withColumn('score', F.aggregate('features', F.lit(0.0), lambda acc, x : acc + x))
seeds = select_seeds(fvs, 'score', 50, labeler)
```

5. Active Learning

```python
model = SKLearnModel(HistGradientBoostingClassifier)

al = EntropyActiveLearner(model, labeler, batch_size=10, max_iter=50)
trained_model = al.train(train_fvs, seeds)

fvs = trained_model.predict(fvs, 'features', 'prediction')
fvs.show()
```

## Installing Dependencies 

### Python

Sparkly has been tested for Python 3.10 on Ubuntu 22.04.

### Other Requirements

This repo has a requirements file which will install the 
other dependencies for Sparkly, to install these dependencies simply use pip

`$ python3 -m pip install -r ./requirements.txt`

The requirements file will install PySpark 3.3.1 with pip but any PySpark installation can be used 
as long as version 3.1.2 or greater is used.



## Tutorials

To get started with ActiveMatcher we recommend starting with the IPython notebook included with 
the repository [examples/example.ipynb](https://github.com/derekpaulsen/active_matcher/blob/main/examples/example.ipynb).

